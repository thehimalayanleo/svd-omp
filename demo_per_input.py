"""Demonstrate input-dependence: 8 random inputs, show the SVD-OMP support per input.

This is the result VPD's architecture promises (input-dependent support)
but its trained `g` does not deliver -- VPD's support is effectively static.
"""

from __future__ import annotations

import argparse

import torch

from model_config import get_C, get_k
from svd_omp import svd_decompose, svd_omp_select


def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = torch.load(args.weights, map_location=device)

    W = weights[args.module].to(device).float()
    C = get_C(args.module)
    k = get_k(args.module)
    V_dict, U_dict, S = svd_decompose(W, C)

    torch.manual_seed(42)
    phi = torch.randn(args.n, W.shape[1], device=device) * 0.5
    W_hat, support, scores = svd_omp_select(phi, V_dict, U_dict, S, k)

    W_true = phi @ W.T
    err = (W_hat - W_true).pow(2).mean(dim=1)

    print(f"{args.module}  k={k}  C={C}")
    print(f"{'Input':>6}  {'Active components (top-4 of k)':<32}  {'Recon MSE':>10}")
    print("-" * 56)
    for i in range(args.n):
        sup = sorted(support[i].tolist())[:4]
        print(f"{i:>6}  {str(sup):<32}  {err[i].item():>10.5f}")

    sups = [set(support[i].tolist()) for i in range(args.n)]
    print("\nJaccard similarity between input supports:")
    print("     " + "  ".join(f"{i:>4}" for i in range(args.n)))
    for i in range(args.n):
        row = " ".join(
            f"{len(sups[i] & sups[j]) / max(1, len(sups[i] | sups[j])):.2f}"
            for j in range(args.n)
        )
        print(f"  {i:>3}  {row}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/weight_matrices.pt")
    ap.add_argument("--module", default="h.0.attn.q_proj")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
