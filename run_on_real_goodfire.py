"""End-to-end runner on real Goodfire 67M weights + activations.

Once you've run scripts/save_goodfire_weights.py in Colab and moved the two
files to weights/, this script runs the three headline sweeps on real
Goodfire model data instead of synthetic:

  1. compare_all: analytic vs trained, Frobenius reconstruction, 24 modules
  2. compare_causal: adversarial downstream MSE (uses next-module W_next)
  3. stable_rank on real activations: reproduce BSF's plateau

Usage:
    python run_on_real_goodfire.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from block_svd_omp import block_svd_decompose
from bsf_weights import evaluate_bsf_weights, run_bsf_weights
from causal_trainable_svd_omp import (
    causal_trainable_select,
    downstream_mse,
    run_causal_trainable_svd_omp,
)
from metrics import evaluate_block_svd_omp, evaluate_svd_omp, evaluate_vpd
from model_config import TARGET_MODULES, get_C, get_k
from stable_rank import activation_stable_rank_per_block
from svd_omp import svd_decompose
from vpd_baseline import run_vpd

WEIGHTS_PATH = Path("weights/goodfire_67m_weights.pt")
ACTIVATIONS_PATH = Path("weights/goodfire_67m_activations.pt")


def check_files():
    for p in (WEIGHTS_PATH, ACTIVATIONS_PATH):
        if not p.exists():
            print(f"MISSING: {p}")
            print("  Run scripts/save_goodfire_weights.py in Colab and download the file.")
            return False
    return True


def frobenius_sweep(weights):
    """Reproduces compare_all.py's key numbers on the real weights."""
    print("\n=== Frobenius sweep (SVD-OMP vs VPD vs BSF-W vs BSF-warm) ===")
    results = {}
    for mod_path in TARGET_MODULES:
        W = weights[mod_path].float()
        C = get_C(mod_path)
        k = get_k(mod_path)

        V, U, S = svd_decompose(W, C)
        m_svd = evaluate_svd_omp(W, V, U, S, k, n_stab_trials=3, batch_size=64)

        Vv, Uv, gv = run_vpd(W, C, k, n=40, seed=0, verbose=False)
        m_vpd = evaluate_vpd(W, Vv, Uv, gv, k, C)

        C_bl = (C // 4) * 4
        Vb, Ub, gb, blb = run_bsf_weights(W, C_bl, 4, max(1, k // 4),
                                          n=40, seed=0, verbose=False)
        m_bsf_cold = evaluate_bsf_weights(W, Vb, Ub, gb, blb, max(1, k // 4))

        Vw, Uw, gw, blw = run_bsf_weights(W, C_bl, 4, max(1, k // 4),
                                          n=40, seed=0, warm_start_svd=True, verbose=False)
        m_bsf_warm = evaluate_bsf_weights(W, Vw, Uw, gw, blw, max(1, k // 4))

        results[mod_path] = {
            "svd":       m_svd["sparse_mse"],
            "vpd":       m_vpd["sparse_mse"],
            "bsf_cold":  m_bsf_cold["sparse_mse"],
            "bsf_warm":  m_bsf_warm["sparse_mse"],
        }
        print(f"  {mod_path:<22}  svd={m_svd['sparse_mse']:.5f}  "
              f"vpd={m_vpd['sparse_mse']:.5f}  "
              f"bsf_cold={m_bsf_cold['sparse_mse']:.5f}  "
              f"bsf_warm={m_bsf_warm['sparse_mse']:.5f}")

    # Win counts.
    def wins(a, b):
        return sum(1 for r in results.values() if r[a] < r[b])
    print("\n  Pairwise wins on sparse_mse (lower is better, out of 24):")
    METHODS = ["svd", "vpd", "bsf_cold", "bsf_warm"]
    for a in METHODS:
        row = [f"{wins(a, b):>4}" if a != b else "   -" for b in METHODS]
        print(f"    {a:<12} " + " ".join(row))

    return results


def downstream_sweep(weights):
    """Runs the non-Frobenius / causal-trained comparison on the real weights,
    pairing each module with the next in TARGET_MODULES as W_next.
    """
    print("\n=== Downstream (non-Frobenius) sweep ===")
    results = {}
    for i, mod_path in enumerate(TARGET_MODULES):
        down_path = TARGET_MODULES[(i + 1) % len(TARGET_MODULES)]
        W = weights[mod_path].float()
        W_next = weights[down_path].float()
        # If W_next input dim doesn't match W output dim, take a random-projection
        # of the right shape.
        if W_next.shape[1] != W.shape[0]:
            torch.manual_seed(0)
            proj = torch.randn(W_next.shape[1], W.shape[0]) / W.shape[0] ** 0.5
            W_next = W_next @ proj

        C, r, k = 16, 4, 1
        V_a, U_a, S_a, blocks = block_svd_decompose(W, C, r)
        torch.manual_seed(999)
        phi = torch.randn(128, W.shape[1]) * 1.5
        from block_svd_omp import block_svd_omp_select_vectorized
        z_a, _, _ = block_svd_omp_select_vectorized(phi, V_a, U_a, S_a, blocks, k)
        mse_a = downstream_mse(phi, W, W_next, z_a, activation="relu")

        V, U, S, ls, b, _ = run_causal_trainable_svd_omp(
            W, W_next, C, r, k, n=200, mode="full", seed=0,
            verbose=False, activation="relu", phi_scale=1.5, faith_anchor=0.0)
        z_t, _, _ = causal_trainable_select(phi, V, U, S, ls, b, blocks, k)
        mse_t = downstream_mse(phi, W, W_next, z_t, activation="relu")

        pct = 0.0 if mse_a == 0 else (mse_a - mse_t) / mse_a * 100
        results[mod_path] = {
            "downstream_mse_analytic": mse_a,
            "downstream_mse_trained": mse_t,
            "pct_reduction": pct,
            "downstream_module": down_path,
        }
        print(f"  {mod_path:<22}  analytic={mse_a:.3f}  trained={mse_t:.3f}  "
              f"reduction={pct:>5.1f}%")

    mean_pct = sum(r["pct_reduction"] for r in results.values()) / len(results)
    wins = sum(1 for r in results.values() if r["pct_reduction"] > 5)
    print(f"\n  Mean downstream MSE reduction: {mean_pct:.1f}%")
    print(f"  Trained wins substantively (>5%): {wins}/24 modules")
    return results


def stable_rank_on_real_activations(weights, activations):
    """BSF-style K-sweep on real Goodfire 67M activations."""
    print("\n=== Stable rank on real activations ===")
    # Use one representative weight matrix that shares d_in with the activations.
    W = weights["h.0.mlp.c_fc"].float()   # [d_intermediate, d_model]
    d_in = W.shape[1]
    phi = activations.float()
    if phi.shape[1] != d_in:
        print(f"  activation dim {phi.shape[1]} != W d_in {d_in}; searching alternative")
        for path in TARGET_MODULES:
            W_try = weights[path].float()
            if W_try.shape[1] == phi.shape[1]:
                W = W_try
                print(f"  using {path} with d_in={W.shape[1]}")
                break
    print(f"  W shape: {tuple(W.shape)}   activations: {tuple(phi.shape)}")

    BLOCK_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]
    results = {m: {} for m in ("analytic", "bsf_cold", "bsf_warm")}
    for K in BLOCK_SIZES:
        C = min(K * 8, min(W.shape) // K * K)
        if C < K:
            continue
        V_a, _, _, blocks = block_svd_decompose(W, C, K)
        results["analytic"][K] = mean_rank(V_a, blocks, phi)

        V_c, _, _, blocks_c = run_bsf_weights(W, C, K, k_blocks=2, n=60, seed=0,
                                              warm_start_svd=False, verbose=False)
        results["bsf_cold"][K] = mean_rank(V_c, blocks_c, phi)

        V_w, _, _, blocks_w = run_bsf_weights(W, C, K, k_blocks=2, n=60, seed=0,
                                              warm_start_svd=True, verbose=False)
        results["bsf_warm"][K] = mean_rank(V_w, blocks_w, phi)

    print(f"\n  {'K':<4} " + " ".join(f"{m:>15}" for m in results))
    for K in BLOCK_SIZES:
        vals = [results[m].get(K, float("nan")) for m in results]
        print(f"  {K:<4} " + " ".join(f"{v:>15.2f}" for v in vals))

    # Panel plot.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Stable rank vs K on real Goodfire 67M activations", y=1.02)
    for ax, key, title in zip(axes, ("analytic", "bsf_cold", "bsf_warm"),
                              ("Analytic block-SVD-OMP", "BSF-W (cold)", "BSF-W (SVD warm)")):
        Ks = sorted(results[key].keys())
        ys = [results[key][K] for K in Ks]
        ax.plot(Ks, Ks, "k--", alpha=0.4, label="full rank")
        ax.plot(Ks, ys, "-o", color="#4C72B0")
        ax.set_xlabel("group size K"); ax.set_ylabel("Stable Rank"); ax.set_title(title)
        ax.set_xlim(0, 17); ax.set_ylim(0, 17)
    Path("figures").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/stable_rank_real_goodfire.png", bbox_inches="tight", dpi=150)
    plt.savefig("figures/stable_rank_real_goodfire.pdf", bbox_inches="tight", dpi=150)
    return results


def mean_rank(V, blocks, phi):
    ranks = activation_stable_rank_per_block(V, blocks, phi)
    return sum(ranks) / len(ranks)


def main():
    if not check_files():
        return 1

    weights = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    activations = torch.load(ACTIVATIONS_PATH, map_location="cpu", weights_only=False)
    print(f"Loaded {len(weights)} weight matrices")
    print(f"Activations shape: {tuple(activations.shape)}")

    Path("results").mkdir(exist_ok=True)
    t0 = time.time()

    frob = frobenius_sweep(weights)
    Path("results/real_frobenius.json").write_text(json.dumps(frob, indent=2))

    dnstream = downstream_sweep(weights)
    Path("results/real_downstream.json").write_text(json.dumps(dnstream, indent=2))

    sr = stable_rank_on_real_activations(weights, activations)
    Path("results/real_stable_rank.json").write_text(json.dumps(sr, indent=2))

    print(f"\nTotal: {time.time() - t0:.1f}s")
    print("Wrote:")
    print("  results/real_frobenius.json")
    print("  results/real_downstream.json")
    print("  results/real_stable_rank.json")
    print("  figures/stable_rank_real_goodfire.{png,pdf}")


if __name__ == "__main__":
    main()
