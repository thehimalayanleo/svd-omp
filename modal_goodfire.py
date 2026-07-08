"""Modal app: load Goodfire 67M, extract weights + activations, run all sweeps.

Replaces the manual Colab step. Runs headlessly on Modal with your wandb API
key stored as a Modal secret.

Setup (one-time):
    modal secret create wandb-secret WANDB_API_KEY=<your-key>

Run:
    modal run modal_goodfire.py

This clones goodfire-ai/param-decomp, loads the 67M LlamaSimpleMLP from
wandb, extracts:
    - The 24 target weight matrices (weights/goodfire_67m_weights.pt)
    - A batch of real MLP-input activations (weights/goodfire_67m_activations.pt)

Then runs the three headline sweeps on this real data (Frobenius, causal
downstream, stable rank) and downloads the resulting JSONs and figures back
to the local repo.
"""

import modal

app = modal.App("svd-omp-goodfire")

# Base image with all deps + a shallow clone of goodfire/param-decomp on PYTHONPATH.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.4",
        "wandb",
        "datasets",
        "transformers",
        "jaxtyping",
        "einops",
        "pydantic",
        "python-dotenv",
        "fire",
        "tqdm",
        "scipy",
        "matplotlib",
        "numpy",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/goodfire-ai/param-decomp "
        "/root/param-decomp"
    )
    .env({"PYTHONPATH": "/root/param-decomp:/root/svd-omp"})
    # Mount the local svd-omp source LAST so we can iterate without
    # rebuilding the image. Exclude .venv and large binaries.
    .add_local_dir(
        ".",
        "/root/svd-omp",
        ignore=[".venv", "__pycache__", ".git", ".pytest_cache",
                "*.pt", "*.png", "*.pdf", "notebooks"],
    )
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/volume": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=1800,
)
def load_extract_and_sweep() -> dict:
    """Load Goodfire's 67M model, extract data, run the three sweeps."""
    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/root/svd-omp")
    sys.path.insert(0, "/root/param-decomp")

    import torch
    from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
        LlamaSimpleMLP,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Goodfire 67M model from wandb run t-9d2b8f02...")
    target_model = LlamaSimpleMLP.from_pretrained("goodfire/spd/runs/t-9d2b8f02")
    target_model.eval().to(device)

    from model_config import TARGET_MODULES

    print(f"Extracting {len(TARGET_MODULES)} weight matrices...")
    weights = {
        p: target_model.get_submodule(p).weight.detach().float().cpu()
        for p in TARGET_MODULES
    }
    Path("/volume/weights").mkdir(parents=True, exist_ok=True)
    torch.save(weights, "/volume/weights/goodfire_67m_weights.pt")

    print("Sampling real MLP-input activations...")
    captured = []

    def hook(mod, inp, out):
        captured.append(inp[0].detach().cpu())

    h = target_model.h[2].mlp.c_fc.register_forward_hook(hook)
    torch.manual_seed(0)
    ids = torch.randint(0, target_model.config.vocab_size, (16, 128), device=device)
    with torch.no_grad():
        _ = target_model(ids)
    h.remove()
    activations = torch.cat([c.reshape(-1, c.shape[-1]) for c in captured], dim=0)
    torch.save(activations, "/volume/weights/goodfire_67m_activations.pt")
    print(f"  activations shape: {tuple(activations.shape)}")

    # Now run the three sweeps in-process.
    from block_svd_omp import block_svd_decompose, block_svd_omp_select_vectorized
    from bsf_weights import evaluate_bsf_weights, run_bsf_weights
    from causal_trainable_svd_omp import (
        causal_trainable_select,
        downstream_mse,
        run_causal_trainable_svd_omp,
    )
    from metrics import evaluate_svd_omp, evaluate_vpd
    from model_config import get_C, get_k
    from stable_rank import activation_stable_rank_per_block
    from svd_omp import svd_decompose
    from vpd_baseline import run_vpd

    # ---- (1) Frobenius sweep ----
    print("\n[1/3] Frobenius sweep on real weights...")
    t0 = time.time()
    frob = {}
    for mod_path in TARGET_MODULES:
        W = weights[mod_path].float().to(device)
        C = get_C(mod_path)
        k = get_k(mod_path)

        V, U, S = svd_decompose(W, C)
        m_svd = evaluate_svd_omp(W, V, U, S, k, n_stab_trials=3, batch_size=64)

        Vv, Uv, gv = run_vpd(W, C, k, n=40, seed=0, verbose=False)
        m_vpd = evaluate_vpd(W, Vv, Uv, gv, k, C)

        C_bl = (C // 4) * 4
        Vb, Ub, gb, blb = run_bsf_weights(
            W, C_bl, 4, max(1, k // 4), n=40, seed=0, verbose=False,
        )
        m_bsf_cold = evaluate_bsf_weights(W, Vb, Ub, gb, blb, max(1, k // 4))

        Vw, Uw, gw, blw = run_bsf_weights(
            W, C_bl, 4, max(1, k // 4), n=40, seed=0,
            warm_start_svd=True, verbose=False,
        )
        m_bsf_warm = evaluate_bsf_weights(W, Vw, Uw, gw, blw, max(1, k // 4))

        frob[mod_path] = {
            "svd":       m_svd["sparse_mse"],
            "vpd":       m_vpd["sparse_mse"],
            "bsf_cold":  m_bsf_cold["sparse_mse"],
            "bsf_warm":  m_bsf_warm["sparse_mse"],
        }
    print(f"  Frobenius sweep: {time.time() - t0:.0f}s")

    # ---- (2) Downstream sweep ----
    print("\n[2/3] Downstream (non-Frobenius) sweep...")
    t0 = time.time()
    dnstream = {}
    for i, mod_path in enumerate(TARGET_MODULES):
        down_path = TARGET_MODULES[(i + 1) % len(TARGET_MODULES)]
        W = weights[mod_path].float().to(device)
        W_next = weights[down_path].float().to(device)
        if W_next.shape[1] != W.shape[0]:
            torch.manual_seed(0)
            proj = torch.randn(W_next.shape[1], W.shape[0], device=device) / W.shape[0] ** 0.5
            W_next = W_next @ proj

        C, r, k = 16, 4, 1
        V_a, U_a, S_a, blocks = block_svd_decompose(W, C, r)
        torch.manual_seed(999)
        phi = torch.randn(128, W.shape[1], device=device) * 1.5
        z_a, _, _ = block_svd_omp_select_vectorized(phi, V_a, U_a, S_a, blocks, k)
        mse_a = downstream_mse(phi, W, W_next, z_a, activation="relu")

        V, U, S, ls, b, _ = run_causal_trainable_svd_omp(
            W, W_next, C, r, k, n=200, mode="full", seed=0,
            verbose=False, activation="relu", phi_scale=1.5, faith_anchor=0.0,
        )
        z_t, _, _ = causal_trainable_select(phi, V, U, S, ls, b, blocks, k)
        mse_t = downstream_mse(phi, W, W_next, z_t, activation="relu")

        pct = 0.0 if mse_a == 0 else (mse_a - mse_t) / mse_a * 100
        dnstream[mod_path] = {
            "downstream_mse_analytic": mse_a,
            "downstream_mse_trained":  mse_t,
            "pct_reduction":           pct,
            "downstream_module":       down_path,
        }
    print(f"  Downstream sweep: {time.time() - t0:.0f}s")

    # ---- (3) Stable rank on real activations ----
    print("\n[3/3] Stable-rank sweep on real activations...")
    t0 = time.time()
    W_c_fc = weights["h.0.mlp.c_fc"].float().to(device)
    phi = activations.float().to(device)
    if phi.shape[1] != W_c_fc.shape[1]:
        for path in TARGET_MODULES:
            W_try = weights[path].float()
            if W_try.shape[1] == phi.shape[1]:
                W_c_fc = W_try.to(device)
                break

    BLOCK_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]
    sr = {m: {} for m in ("analytic", "bsf_cold", "bsf_warm")}
    for K in BLOCK_SIZES:
        C = min(K * 8, min(W_c_fc.shape) // K * K)
        if C < K:
            continue
        V_a, _, _, blocks = block_svd_decompose(W_c_fc, C, K)
        sr["analytic"][K] = float(
            sum(activation_stable_rank_per_block(V_a, blocks, phi))
            / len(blocks)
        )
        V_c, _, _, blocks_c = run_bsf_weights(
            W_c_fc, C, K, k_blocks=2, n=60, seed=0,
            warm_start_svd=False, verbose=False,
        )
        sr["bsf_cold"][K] = float(
            sum(activation_stable_rank_per_block(V_c, blocks_c, phi))
            / len(blocks_c)
        )
        V_w, _, _, blocks_w = run_bsf_weights(
            W_c_fc, C, K, k_blocks=2, n=60, seed=0,
            warm_start_svd=True, verbose=False,
        )
        sr["bsf_warm"][K] = float(
            sum(activation_stable_rank_per_block(V_w, blocks_w, phi))
            / len(blocks_w)
        )
    print(f"  Stable rank: {time.time() - t0:.0f}s")

    # Persist JSON alongside weights.
    Path("/volume/results").mkdir(parents=True, exist_ok=True)
    Path("/volume/results/real_frobenius.json").write_text(json.dumps(frob, indent=2))
    Path("/volume/results/real_downstream.json").write_text(json.dumps(dnstream, indent=2))
    Path("/volume/results/real_stable_rank.json").write_text(json.dumps(sr, indent=2))

    volume.commit()

    return {
        "frobenius": frob,
        "downstream": dnstream,
        "stable_rank": sr,
        "activation_shape": list(activations.shape),
    }


@app.local_entrypoint()
def main():
    """Trigger the sweep and pull the artifacts back to the local repo."""
    import json
    from pathlib import Path

    print("Running on Modal — this will take ~5 minutes on a T4...")
    result = load_extract_and_sweep.remote()

    print("\n=== Summary ===")

    # Downstream: mean % reduction.
    dn = result["downstream"]
    mean_pct = sum(r["pct_reduction"] for r in dn.values()) / len(dn)
    wins = sum(1 for r in dn.values() if r["pct_reduction"] > 5)
    print(f"Non-Frobenius trained beats analytic:  mean MSE reduction {mean_pct:.1f}%,  "
          f"substantive wins {wins}/24")

    # Frobenius: pairwise wins.
    frob = result["frobenius"]

    def wins(a, b):
        return sum(1 for r in frob.values() if r[a] < r[b])

    print(f"\nFrobenius pairwise wins on real weights (24 modules):")
    for a in ["svd", "vpd", "bsf_cold", "bsf_warm"]:
        row = " ".join(
            f"{wins(a, b):>4}" if a != b else "   -"
            for b in ["svd", "vpd", "bsf_cold", "bsf_warm"]
        )
        print(f"  {a:<10} {row}")

    # Stable rank plateau.
    sr = result["stable_rank"]
    print(f"\nStable rank K sweep on real Goodfire activations:")
    print(f"  {'K':<4} {'analytic':>10} {'bsf_cold':>10} {'bsf_warm':>10}")
    for K in sorted(sr["analytic"].keys(), key=int):
        Ki = int(K)
        vals = " ".join(f"{sr[m][K]:>10.2f}" for m in ("analytic", "bsf_cold", "bsf_warm"))
        print(f"  {Ki:<4} {vals}")

    # Save locally too.
    Path("results").mkdir(exist_ok=True)
    Path("results/real_frobenius.json").write_text(json.dumps(frob, indent=2))
    Path("results/real_downstream.json").write_text(json.dumps(dn, indent=2))
    Path("results/real_stable_rank.json").write_text(json.dumps(sr, indent=2))
    print("\nWrote:")
    print("  results/real_frobenius.json")
    print("  results/real_downstream.json")
    print("  results/real_stable_rank.json")
