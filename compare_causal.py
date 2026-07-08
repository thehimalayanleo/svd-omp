"""Adversarial-construction sweep: analytic SVD-OMP vs causal-trained SVD-OMP.

Builds a bank of adversarial (W, W_next) pairs at production shapes (matches
the 24 target modules of the 67M model) and reports how much of the analytic
downstream MSE the trained method removes.

Adversarial construction (same as tests/test_causal_trainable.py):
  - W has two clearly-separated singular tiers: 4 loud atoms (sigma=10),
    4 quiet atoms (sigma=2), then noise
  - W_next projects onto the QUIET band only, so downstream signal is
    carried entirely by the quiet atoms
  - Analytic top-1-block selection picks the LOUD block (largest projection
    norm), whose atoms are killed by W_next
  - Trained selection can rotate/reweight to pick the QUIET block

Usage:
    python compare_causal.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from block_svd_omp import block_svd_decompose, block_svd_omp_select_vectorized
from causal_trainable_svd_omp import (
    causal_trainable_select,
    downstream_mse,
    run_causal_trainable_svd_omp,
)
from model_config import TARGET_MODULES


def make_adversarial_weight_and_next(d_out: int, d_in: int, d_next: int, seed: int):
    torch.manual_seed(seed)
    U_full = torch.linalg.qr(torch.randn(d_out, d_out))[0]
    V_full = torch.linalg.qr(torch.randn(d_in, d_in))[0]
    S = torch.zeros(min(d_out, d_in))
    S[:4] = 10.0     # loud
    S[4:8] = 2.0     # quiet
    W = U_full[:, :len(S)] @ torch.diag(S) @ V_full[:, :len(S)].T
    W = W + 0.05 * torch.randn(d_out, d_in)
    W_next = torch.zeros(d_next, d_out)
    for i, j in enumerate(range(4, 8)):
        if i < d_next:
            W_next[i] = U_full[:, j] * 3.0
    return W, W_next


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Adversarial causal-training sweep on synthetic weights sized like the 67M model.")
    print(f"Device: {device}\n")

    shape_of = {
        "attn.q_proj":    (768, 768),
        "attn.k_proj":    (768, 768),
        "attn.v_proj":    (768, 768),
        "attn.o_proj":    (768, 768),
        "mlp.c_fc":       (3072, 768),
        "mlp.down_proj":  (768, 3072),
    }

    def shape_for(path):
        for suffix, shape in shape_of.items():
            if path.endswith(suffix):
                return shape
        raise ValueError(path)

    results = {}
    t0 = time.time()

    for i, path in enumerate(TARGET_MODULES):
        d_out, d_in = shape_for(path)
        # Use d_out as the downstream dim so W_next is sized to route the quiet band.
        d_next = min(d_out, 32)
        W, W_next = make_adversarial_weight_and_next(d_out, d_in, d_next, seed=1000 + i)
        W = W.to(device)
        W_next = W_next.to(device)

        C, r, k = 16, 4, 1
        PHI_SCALE = 1.5

        # Analytic block-SVD-OMP.
        V_a, U_a, S_a, blocks = block_svd_decompose(W, C, r)
        torch.manual_seed(999)
        phi = torch.randn(128, W.shape[1], device=device) * PHI_SCALE
        z_a, sup_a, _ = block_svd_omp_select_vectorized(phi, V_a, U_a, S_a, blocks, k)
        mse_a = downstream_mse(phi, W, W_next, z_a, activation="relu")

        # Trained full-mode.
        V, U, S, ls, b, _ = run_causal_trainable_svd_omp(
            W, W_next, C, r, k, n=300, mode="full", seed=0,
            verbose=False, activation="relu", phi_scale=PHI_SCALE, faith_anchor=0.0)
        z_t, sup_t, _ = causal_trainable_select(phi, V, U, S, ls, b, blocks, k)
        mse_t = downstream_mse(phi, W, W_next, z_t, activation="relu")

        pct = 0.0 if mse_a == 0 else (mse_a - mse_t) / mse_a * 100.0
        fraction_block0_analytic = float((sup_a == 0).float().mean().item())
        fraction_block0_trained = float((sup_t == 0).float().mean().item())
        results[path] = {
            "downstream_mse_analytic": mse_a,
            "downstream_mse_trained": mse_t,
            "pct_reduction": pct,
            "fraction_loud_block_analytic": fraction_block0_analytic,
            "fraction_loud_block_trained": fraction_block0_trained,
        }
        print(f"  [{i+1:>2}/24] {path:<22}  "
              f"analytic={mse_a:.3f}  trained={mse_t:.3f}  "
              f"reduction={pct:>5.1f}%  "
              f"loud-picks: analytic {fraction_block0_analytic:.2f} -> trained {fraction_block0_trained:.2f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    mean_pct = sum(r["pct_reduction"] for r in results.values()) / len(results)
    wins = sum(1 for r in results.values() if r["pct_reduction"] > 5.0)
    print(f"\nSummary:")
    print(f"  Mean downstream MSE reduction:  {mean_pct:.1f}%")
    print(f"  Trained wins substantively (> 5% reduction): {wins}/24 modules")

    Path("results/compare_causal.json").parent.mkdir(parents=True, exist_ok=True)
    Path("results/compare_causal.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote results/compare_causal.json")


if __name__ == "__main__":
    main()
