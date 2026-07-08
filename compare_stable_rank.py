"""BSF-style stable-rank sweep on our block methods.

Reproduces the panel from BSF-Vision by sweeping block size K from 1..16
and measuring the mean activation stable rank per block for four methods:

    1. Analytic block-SVD-OMP    (SVD basis, no training)
    2. BSF-W random init         (learned block dictionary)
    3. BSF-W SVD warm-start      (SVD init, then trained)
    4. Trainable-SVD-OMP scaffold (SVD basis + learned per-block scale/bias)

If any method's stable rank plateaus around 4 like BSF's plot, we have
evidence that the "concepts are 2-4 dimensional" finding is not an
artifact of BSF's specific training recipe.

Usage:
    python compare_stable_rank.py                       # synthetic weights
    python compare_stable_rank.py --module h.0.attn.q_proj
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from block_svd_omp import block_svd_decompose
from bsf_weights import run_bsf_weights
from stable_rank import activation_stable_rank_per_block
from trainable_svd_omp import run_trainable_svd_omp

BLOCK_SIZES = [1, 2, 3, 4, 6, 8, 12, 16]


def synth_weight(d_out=768, d_in=768, seed=0):
    """Low-rank + noise + some structure — mimics a real transformer weight."""
    torch.manual_seed(seed)
    rank = 32
    A = torch.randn(d_out, rank)
    B = torch.randn(d_in, rank)
    W = A @ B.T + 0.05 * torch.randn(d_out, d_in)
    return W / W.norm() * (d_in * d_out) ** 0.25


def synth_activations(d_in, batch_size, seed=0):
    """Structured activations that aren't isotropic Gaussian — a few dominant
    directions plus noise, roughly like real transformer activations."""
    torch.manual_seed(seed)
    # 8 dominant directions.
    U_act = torch.linalg.qr(torch.randn(d_in, d_in))[0][:, :8]
    coefs = torch.randn(batch_size, 8) * 2.0  # heavy tail
    return coefs @ U_act.T + 0.1 * torch.randn(batch_size, d_in)


def measure_stable_rank(V, blocks, phi):
    ranks = activation_stable_rank_per_block(V, blocks, phi)
    return sum(ranks) / len(ranks)


def sweep(W, phi):
    """Return dict of {method: {K: mean_stable_rank}}."""
    results = {m: {} for m in ["analytic", "bsf_cold", "bsf_warm", "trainable"]}
    d_in = W.shape[1]
    d_out = W.shape[0]

    for K in BLOCK_SIZES:
        # C big enough to always allocate at least 4 blocks.
        C = min(K * 8, min(d_out, d_in) // K * K)
        if C < K:
            continue

        # Analytic block-SVD-OMP.
        V_a, _, _, blocks = block_svd_decompose(W, C, K)
        results["analytic"][K] = measure_stable_rank(V_a, blocks, phi)

        # BSF-W cold.
        V_c, _, _, blocks_c = run_bsf_weights(W, C, K, k_blocks=2, n=60, seed=0,
                                              warm_start_svd=False, verbose=False)
        results["bsf_cold"][K] = measure_stable_rank(V_c, blocks_c, phi)

        # BSF-W warm.
        V_w, _, _, blocks_w = run_bsf_weights(W, C, K, k_blocks=2, n=60, seed=0,
                                              warm_start_svd=True, verbose=False)
        results["bsf_warm"][K] = measure_stable_rank(V_w, blocks_w, phi)

        # Trainable-SVD-OMP scaffold: V is frozen at SVD, so stable rank is
        # analytic-identical -- skip retraining, reuse analytic V.
        results["trainable"][K] = results["analytic"][K]

    return results


def main(args):
    W = synth_weight(seed=0)
    phi = synth_activations(W.shape[1], batch_size=256, seed=1)
    print(f"W shape: {tuple(W.shape)}   phi shape: {tuple(phi.shape)}")

    results = sweep(W, phi)

    print(f"\n{'K':<4} " + " ".join(f"{m:>15}" for m in results))
    for K in BLOCK_SIZES:
        vals = [results[m].get(K, float("nan")) for m in results]
        print(f"{K:<4} " + " ".join(f"{v:>15.2f}" for v in vals))

    # Plot: BSF-style panels.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Stable rank vs block size K — analytic SVD-OMP vs BSF-style baselines",
                 y=1.02)

    titles = ["Analytic block-SVD-OMP", "BSF-W (random init)", "BSF-W (SVD warm-start)"]
    keys = ["analytic", "bsf_cold", "bsf_warm"]

    for ax, key, title in zip(axes, keys, titles):
        Ks = sorted(results[key].keys())
        ys = [results[key][K] for K in Ks]
        ax.plot(Ks, Ks, "k--", alpha=0.4, label="full rank (= K)")
        ax.plot(Ks, ys, "-o", color="#4C72B0", label="measured")
        ax.set_xlabel("group size K")
        ax.set_ylabel("Stable Rank")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 17)
        ax.set_ylim(0, 17)
        ax.legend(loc="upper left", fontsize=9)

    Path("figures").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("figures/stable_rank_vs_K.pdf", bbox_inches="tight", dpi=150)
    plt.savefig("figures/stable_rank_vs_K.png", bbox_inches="tight", dpi=150)

    Path("results").mkdir(exist_ok=True)
    Path("results/stable_rank_sweep.json").write_text(json.dumps(results, indent=2))
    print("\nWrote figures/stable_rank_vs_K.{png,pdf} and results/stable_rank_sweep.json")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=None, help="unused for synthetic mode")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
