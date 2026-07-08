"""4-way weight-decomposition comparison: SVD-OMP vs block-SVD-OMP vs VPD vs BSF-W.

The four methods span the 2x2 design space:

                              1D atoms          block atoms
                             +----------------+-----------------+
        analytic / no train  | SVD-OMP        | block-SVD-OMP   |
                             +----------------+-----------------+
        learned / trained    | VPD            | BSF-W           |
                             +----------------+-----------------+

BSF-W = a BSF-style trained baseline applied to weight decomposition
(the original BSF paper decomposes activations, not weights; we adapt it).

Runs each method on the 24 target modules if `weight_matrices.pt` is on disk;
otherwise falls back to synthetic weights sized like the real 67M model so
you can dry-run the full pipeline offline.

Usage:
    python compare_all.py                         # synthetic weights
    python compare_all.py --weights weights/weight_matrices.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from block_svd_omp import block_svd_decompose
from bsf_weights import evaluate_bsf_weights, run_bsf_weights
from metrics import (
    evaluate_block_svd_omp,
    evaluate_svd_omp,
    evaluate_vpd,
)
from model_config import TARGET_MODULES, get_C, get_k
from svd_omp import svd_decompose
from vpd_baseline import run_vpd

# Block config: keep total atoms budget matched between 1D and block variants.
BLOCK_RANK = 4


def synth_weights():
    shape_of = {
        "attn.q_proj": (768, 768),
        "attn.k_proj": (768, 768),
        "attn.v_proj": (768, 768),
        "attn.o_proj": (768, 768),
        "mlp.c_fc":    (3072, 768),
        "mlp.down_proj": (768, 3072),
    }

    def shape_for(path):
        for suffix, shape in shape_of.items():
            if path.endswith(suffix):
                return shape
        raise ValueError(path)

    weights = {}
    for i, path in enumerate(TARGET_MODULES):
        torch.manual_seed(1000 + i)
        d_out, d_in = shape_for(path)
        rank = min(d_out, d_in) // 4
        A = torch.randn(d_out, rank)
        B = torch.randn(d_in, rank)
        W = A @ B.T + 0.05 * torch.randn(d_out, d_in)
        W = W / W.norm() * (d_in * d_out) ** 0.25
        weights[path] = W
    return weights


def block_config_for(mod_path):
    """Match block config (K_blocks * r == k * r_svd_omp) so total active atoms roughly match."""
    k_atoms = get_k(mod_path)
    C = get_C(mod_path)
    # Round C to nearest multiple of BLOCK_RANK.
    C_bl = (C // BLOCK_RANK) * BLOCK_RANK
    k_blocks = max(1, k_atoms // BLOCK_RANK)
    return C_bl, BLOCK_RANK, k_blocks


def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.weights and Path(args.weights).exists():
        weights = torch.load(args.weights, map_location=device)
        source = f"real weights from {args.weights}"
    else:
        weights = synth_weights()
        source = "SYNTHETIC weights (no `--weights` argument or file missing)"

    print(f"Data source: {source}\n")

    results = {}
    t0 = time.time()

    for mod_path in TARGET_MODULES:
        W = weights[mod_path].to(device).float()
        C_1d = get_C(mod_path)
        k_1d = get_k(mod_path)
        C_bl, r, k_blocks = block_config_for(mod_path)

        # SVD-OMP (1D, analytic).
        V1, U1, S1 = svd_decompose(W, C_1d)
        m_svd = evaluate_svd_omp(W, V1, U1, S1, k_1d,
                                 n_stab_trials=args.stab_trials, batch_size=args.batch)

        # block-SVD-OMP (block, analytic).
        V2, U2, S2, blocks = block_svd_decompose(W, C_bl, r)
        m_bs = evaluate_block_svd_omp(W, V2, U2, S2, blocks, k_blocks,
                                      n_stab_trials=args.stab_trials, batch_size=args.batch)

        # VPD (1D, trained).
        Vv, Uv, gv = run_vpd(W, C_1d, k_1d, n=args.vpd_steps, seed=0, verbose=False)
        m_vpd = evaluate_vpd(W, Vv, Uv, gv, k_1d, C_1d)

        # BSF-W (block, trained).
        Vb, Ub, gb, blb = run_bsf_weights(W, C_bl, r, k_blocks,
                                          n=args.vpd_steps, seed=0, verbose=False)
        m_bsf = evaluate_bsf_weights(W, Vb, Ub, gb, blb, k_blocks)

        results[mod_path] = {
            "svd_omp":       {k_: m_svd[k_] for k_ in ("sparse_mse", "faith_mse", "coherence", "stability")},
            "block_svd_omp": {k_: m_bs[k_]  for k_ in ("sparse_mse", "faith_mse", "coherence", "stability")},
            "vpd":           {k_: m_vpd[k_] for k_ in ("sparse_mse", "faith_mse", "coherence")},
            "bsf_w":         {k_: m_bsf[k_] for k_ in ("sparse_mse", "faith_mse", "coherence")},
            "shape": list(W.shape), "C": C_1d, "k": k_1d,
            "C_blocks": C_bl, "block_rank": r, "k_blocks": k_blocks,
        }
        print(f"  {mod_path:<22}  "
              f"svd={m_svd['sparse_mse']:.4f}  "
              f"blk-svd={m_bs['sparse_mse']:.4f}  "
              f"vpd={m_vpd['sparse_mse']:.4f}  "
              f"bsf={m_bsf['sparse_mse']:.4f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # Aggregate: pairwise wins on sparse_mse and coherence.
    def wins(a_key, b_key, metric):
        return sum(1 for r in results.values() if r[a_key][metric] < r[b_key][metric])

    print("\nPairwise wins on sparse_mse (lower is better, out of 24):")
    for a in ["svd_omp", "block_svd_omp", "vpd", "bsf_w"]:
        row = [f"{wins(a, b, 'sparse_mse'):>4}" if a != b else "   -" for b in ["svd_omp", "block_svd_omp", "vpd", "bsf_w"]]
        print(f"  {a:<16} {' '.join(row)}")

    print("\nPairwise wins on coherence (lower is better, out of 24):")
    for a in ["svd_omp", "block_svd_omp", "vpd", "bsf_w"]:
        row = [f"{wins(a, b, 'coherence'):>4}" if a != b else "   -" for b in ["svd_omp", "block_svd_omp", "vpd", "bsf_w"]]
        print(f"  {a:<16} {' '.join(row)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="Optional path to torch.save dict of real weights.")
    ap.add_argument("--out", default="results/compare_all.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--vpd-steps", type=int, default=100)
    ap.add_argument("--stab-trials", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
