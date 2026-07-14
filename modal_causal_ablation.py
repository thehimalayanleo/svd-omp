"""Modal app: causal-ablation head-to-head, SVD-OMP vs VPD, on real Goodfire weights.

This is the decisive experiment for the paper's framing. The Frobenius/coherence
metrics are reconstruction/geometry axes a VPD author would say VPD does not
optimize. This runs the CAUSAL axis: for each input and each selected component,
ablate it and measure output change (local + downstream), plus redundancy of the
selected support. If SVD-OMP's per-input supports are at least as causally
irreplaceable as VPD's trained static support, the dominance claim is earned on
VPD's own turf.

Reuses the cached Goodfire-67M weights on the Modal volume (written by
modal_goodfire.py). Run:

    modal run modal_causal_ablation.py
"""

import modal

app = modal.App("svd-omp-causal-ablation")

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


@app.function(
    image=image,
    gpu="T4",
    volumes={"/volume": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=1800,
)
def causal_sweep() -> dict:
    import json
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/root/svd-omp")
    sys.path.insert(0, "/root/param-decomp")

    import torch

    from causal_ablation import causal_damage, redundancy
    from model_config import TARGET_MODULES, get_C, get_k
    from svd_omp import svd_decompose, svd_omp_select
    from vpd_baseline import run_vpd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    wpath = "/volume/weights/goodfire_67m_weights.pt"
    if not Path(wpath).exists():
        raise FileNotFoundError(
            f"{wpath} not found on volume — run modal_goodfire.py first.")
    weights = torch.load(wpath, map_location=device)
    print(f"Loaded {len(weights)} real Goodfire weight matrices.\n")

    results = {}
    print(f"  {'module':<20} {'metric':<14} {'SVD-OMP':>10} {'VPD':>10}  win")
    print("  " + "-" * 62)
    t0 = time.time()

    for i, mod_path in enumerate(TARGET_MODULES):
        W = weights[mod_path].float().to(device)
        C, k = get_C(mod_path), get_k(mod_path)

        # Downstream matrix = next module, projected to fit if shapes differ.
        down_path = TARGET_MODULES[(i + 1) % len(TARGET_MODULES)]
        W_down = weights[down_path].float().to(device)
        if W_down.shape[1] != W.shape[0]:
            torch.manual_seed(0)
            proj = torch.randn(W_down.shape[1], W.shape[0], device=device) / W.shape[0] ** 0.5
            W_down = W_down @ proj

        V, U, S = svd_decompose(W, C)
        torch.manual_seed(0)
        phi = torch.randn(128, W.shape[1], device=device) * 0.3
        _, sup_svd, _ = svd_omp_select(phi, V, U, S, k)

        Vv, Uv, gv = run_vpd(W, C, k, n=200, seed=0, verbose=False)
        topk_v = torch.argsort(torch.sigmoid(gv), descending=True)[:k]
        sup_vpd = topk_v.unsqueeze(0).expand(phi.shape[0], -1)

        # Causal damage: HIGHER = more irreplaceable = better.
        d_loc_svd = causal_damage(phi, V, U, sup_svd)
        d_loc_vpd = causal_damage(phi, Vv, Uv, sup_vpd)
        d_dn_svd = causal_damage(phi, V, U, sup_svd, W_down=W_down)
        d_dn_vpd = causal_damage(phi, Vv, Uv, sup_vpd, W_down=W_down)
        # Redundancy: LOWER = more irreplaceable = better.
        red_svd = redundancy(phi, V, U, sup_svd)
        red_vpd = redundancy(phi, Vv, Uv, sup_vpd)

        results[mod_path] = {
            "local_damage_svd": d_loc_svd, "local_damage_vpd": d_loc_vpd,
            "downstream_damage_svd": d_dn_svd, "downstream_damage_vpd": d_dn_vpd,
            "redundancy_svd": red_svd, "redundancy_vpd": red_vpd,
            "win_local": d_loc_svd > d_loc_vpd,
            "win_downstream": d_dn_svd > d_dn_vpd,
            "win_redundancy": red_svd < red_vpd,
        }
        for name, sv, vp, hi in [
            ("local_damage", d_loc_svd, d_loc_vpd, True),
            ("downstream_dmg", d_dn_svd, d_dn_vpd, True),
            ("redundancy", red_svd, red_vpd, False),
        ]:
            win = (sv > vp) if hi else (sv < vp)
            print(f"  {mod_path:<20} {name:<14} {sv:>10.4f} {vp:>10.4f}  {'*' if win else ' '}")

    print(f"\n  ({time.time() - t0:.0f}s)")
    Path("/volume/results").mkdir(parents=True, exist_ok=True)
    Path("/volume/results/real_causal_ablation.json").write_text(json.dumps(results, indent=2))
    volume.commit()
    return results


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Running causal ablation on Modal T4 (real Goodfire weights)...")
    res = causal_sweep.remote()

    n = len(res)
    win_local = sum(1 for r in res.values() if r["win_local"])
    win_dn = sum(1 for r in res.values() if r["win_downstream"])
    win_red = sum(1 for r in res.values() if r["win_redundancy"])

    print("\n=== Causal ablation: SVD-OMP vs VPD, 24 real Goodfire modules ===")
    print(f"  Local causal damage (higher=better):      SVD-OMP wins {win_local}/{n}")
    print(f"  Downstream causal damage (higher=better):  SVD-OMP wins {win_dn}/{n}")
    print(f"  Redundancy (lower=better):                 SVD-OMP wins {win_red}/{n}")

    # Mean values for the paper table.
    def mean(key):
        return sum(r[key] for r in res.values()) / n
    print("\n  Means:")
    print(f"    local damage      SVD {mean('local_damage_svd'):.4f}  VPD {mean('local_damage_vpd'):.4f}")
    print(f"    downstream damage SVD {mean('downstream_damage_svd'):.4f}  VPD {mean('downstream_damage_vpd'):.4f}")
    print(f"    redundancy        SVD {mean('redundancy_svd'):.4f}  VPD {mean('redundancy_vpd'):.4f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/real_causal_ablation.json").write_text(json.dumps(res, indent=2))
    print("\nWrote results/real_causal_ablation.json")
