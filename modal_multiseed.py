"""Modal app: multi-seed robustness for the trained baselines.

SVD-OMP is deterministic (no seed). The trained baselines (VPD gate, BSF cold
init) are seed-dependent, and the draft reports single-seed numbers. This runs
the two headline comparisons across 5 seeds on the real Goodfire-67M weights
and reports mean +/- std plus win-count robustness (does SVD-OMP still win
24/24 on every seed?).

Reuses cached weights on the svd-omp-goodfire volume. Run:

    modal run modal_multiseed.py
"""

import modal

app = modal.App("svd-omp-multiseed")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.4", "wandb", "datasets", "transformers", "jaxtyping",
        "einops", "pydantic", "python-dotenv", "fire", "tqdm", "scipy",
        "matplotlib", "numpy",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/goodfire-ai/param-decomp "
        "/root/param-decomp"
    )
    .env({"PYTHONPATH": "/root/param-decomp:/root/svd-omp"})
    .add_local_dir(
        ".",
        "/root/svd-omp",
        ignore=[".venv", "__pycache__", ".git", ".pytest_cache",
                "*.pt", "*.png", "*.pdf", "notebooks"],
    )
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=True)

SEEDS = [0, 1, 2, 3, 4]


@app.function(
    image=image,
    gpu="T4",
    volumes={"/volume": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=3600,
)
def multiseed_sweep() -> dict:
    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/root/svd-omp")
    sys.path.insert(0, "/root/param-decomp")

    import torch

    from bsf_weights import evaluate_bsf_weights, run_bsf_weights
    from causal_ablation import causal_damage
    from metrics import evaluate_svd_omp, evaluate_vpd
    from model_config import TARGET_MODULES, get_C, get_k
    from svd_omp import svd_decompose, svd_omp_select
    from vpd_baseline import run_vpd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wpath = "/volume/weights/goodfire_67m_weights.pt"
    if not Path(wpath).exists():
        raise FileNotFoundError(f"{wpath} not found — run modal_goodfire.py first.")
    weights = torch.load(wpath, map_location=device)
    print(f"Loaded {len(weights)} real Goodfire weight matrices.")
    print(f"Seeds: {SEEDS}\n")

    # SVD-OMP is deterministic: compute its metrics once per module.
    svd_sparse = {}
    svd_causal_local = {}
    svd_causal_dn = {}
    for i, mod_path in enumerate(TARGET_MODULES):
        W = weights[mod_path].float().to(device)
        C, k = get_C(mod_path), get_k(mod_path)
        V, U, S = svd_decompose(W, C)
        m = evaluate_svd_omp(W, V, U, S, k, n_stab_trials=3, batch_size=64)
        svd_sparse[mod_path] = m["sparse_mse"]

        down_path = TARGET_MODULES[(i + 1) % len(TARGET_MODULES)]
        W_down = weights[down_path].float().to(device)
        if W_down.shape[1] != W.shape[0]:
            torch.manual_seed(0)
            proj = torch.randn(W_down.shape[1], W.shape[0], device=device) / W.shape[0] ** 0.5
            W_down = W_down @ proj
        torch.manual_seed(0)
        phi = torch.randn(128, W.shape[1], device=device) * 0.3
        _, sup_svd, _ = svd_omp_select(phi, V, U, S, k)
        svd_causal_local[mod_path] = causal_damage(phi, V, U, sup_svd)
        svd_causal_dn[mod_path] = causal_damage(phi, V, U, sup_svd, W_down=W_down)

    # Per-seed trained baselines.
    # Structure: per_seed[metric][method] = {module: value}
    per_seed = {s: {"sparse": {}, "causal_local": {}, "causal_dn": {}} for s in SEEDS}

    t0 = time.time()
    for s in SEEDS:
        for i, mod_path in enumerate(TARGET_MODULES):
            W = weights[mod_path].float().to(device)
            C, k = get_C(mod_path), get_k(mod_path)

            # VPD (seed-dependent gate).
            Vv, Uv, gv = run_vpd(W, C, k, n=40, seed=s, verbose=False)
            m_vpd = evaluate_vpd(W, Vv, Uv, gv, k, C)

            # BSF cold + warm (cold is seed-dependent init).
            C_bl = (C // 4) * 4
            kb = max(1, k // 4)
            Vb, Ub, gb, blb = run_bsf_weights(W, C_bl, 4, kb, n=40, seed=s,
                                              warm_start_svd=False, verbose=False)
            m_cold = evaluate_bsf_weights(W, Vb, Ub, gb, blb, kb)
            Vw, Uw, gw, blw = run_bsf_weights(W, C_bl, 4, kb, n=40, seed=s,
                                              warm_start_svd=True, verbose=False)
            m_warm = evaluate_bsf_weights(W, Vw, Uw, gw, blw, kb)

            per_seed[s]["sparse"][mod_path] = {
                "vpd": m_vpd["sparse_mse"],
                "bsf_cold": m_cold["sparse_mse"],
                "bsf_warm": m_warm["sparse_mse"],
            }

            # VPD causal ablation (seed-dependent).
            down_path = TARGET_MODULES[(i + 1) % len(TARGET_MODULES)]
            W_down = weights[down_path].float().to(device)
            if W_down.shape[1] != W.shape[0]:
                torch.manual_seed(0)
                proj = torch.randn(W_down.shape[1], W.shape[0], device=device) / W.shape[0] ** 0.5
                W_down = W_down @ proj
            torch.manual_seed(0)
            phi = torch.randn(128, W.shape[1], device=device) * 0.3
            topk_v = torch.argsort(torch.sigmoid(gv), descending=True)[:k]
            sup_vpd = topk_v.unsqueeze(0).expand(phi.shape[0], -1)
            per_seed[s]["causal_local"][mod_path] = causal_damage(phi, Vv, Uv, sup_vpd)
            per_seed[s]["causal_dn"][mod_path] = causal_damage(phi, Vv, Uv, sup_vpd, W_down=W_down)
        print(f"  seed {s} done ({time.time() - t0:.0f}s elapsed)")

    out = {
        "seeds": SEEDS,
        "svd_sparse": svd_sparse,
        "svd_causal_local": svd_causal_local,
        "svd_causal_dn": svd_causal_dn,
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
    }
    Path("/volume/results").mkdir(parents=True, exist_ok=True)
    Path("/volume/results/real_multiseed.json").write_text(json.dumps(out, indent=2))
    volume.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    import statistics as st
    from pathlib import Path

    print("Running multi-seed robustness on Modal T4 (real Goodfire weights)...")
    r = multiseed_sweep.remote()

    seeds = r["seeds"]
    mods = list(r["svd_sparse"].keys())
    n = len(mods)

    def mean_std(vals):
        return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)

    # --- Sparse reconstruction: SVD-OMP win-rate robustness ---
    print("\n=== Sparse reconstruction: SVD-OMP vs trained, across seeds ===")
    for method in ["vpd", "bsf_cold", "bsf_warm"]:
        # For each seed, count modules where SVD-OMP beats the trained method.
        per_seed_wins = []
        for s in seeds:
            sp = r["per_seed"][str(s)]["sparse"]
            wins = sum(1 for m in mods if r["svd_sparse"][m] < sp[m][method])
            per_seed_wins.append(wins)
        print(f"  vs {method:<9}: SVD-OMP wins per seed = {per_seed_wins}  "
              f"(min {min(per_seed_wins)}/{n}, max {max(per_seed_wins)}/{n})")

    # --- Causal ablation: SVD-OMP win-rate robustness (local + downstream) ---
    print("\n=== Causal ablation: SVD-OMP vs VPD, across seeds ===")
    for metric, sv_key in [("local", "svd_causal_local"), ("downstream", "svd_causal_dn")]:
        key = "causal_local" if metric == "local" else "causal_dn"
        per_seed_wins = []
        ratios = []
        for s in seeds:
            cv = r["per_seed"][str(s)][key]
            wins = sum(1 for m in mods if r[sv_key][m] > cv[m])
            per_seed_wins.append(wins)
            ratios.append(st.mean(r[sv_key][m] / max(cv[m], 1e-9) for m in mods))
        rm, rs = mean_std(ratios)
        print(f"  {metric:<10}: SVD-OMP wins per seed = {per_seed_wins}  "
              f"(min {min(per_seed_wins)}/{n})  mean ratio {rm:.1f}x +/- {rs:.1f}")

    # --- Mean +/- std of the trained metrics themselves (for the paper) ---
    print("\n=== Trained-method sparse MSE, mean +/- std across seeds (24-module avg) ===")
    for method in ["vpd", "bsf_cold", "bsf_warm"]:
        per_seed_means = []
        for s in seeds:
            sp = r["per_seed"][str(s)]["sparse"]
            per_seed_means.append(st.mean(sp[m][method] for m in mods))
        m, sd = mean_std(per_seed_means)
        print(f"  {method:<9}: {m:.4f} +/- {sd:.4f}")
    svd_mean = st.mean(r["svd_sparse"][m] for m in mods)
    print(f"  {'svd_omp':<9}: {svd_mean:.4f} (deterministic)")

    Path("results").mkdir(exist_ok=True)
    Path("results/real_multiseed.json").write_text(json.dumps(r, indent=2))
    print("\nWrote results/real_multiseed.json")
