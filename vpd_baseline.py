"""
VPD baseline: reimplementation of Goodfire's adVersarial Parameter Decomposition,
stripped to the structure required for the SVD-OMP comparison.

Trains a [V, U, g] tuple via stochastic masking + faithfulness + sparsity-on-g.
The selected support is the top-k indices of sigmoid(g).

Reference: Bushnaq et al., "adVersarial Parameter Decomposition" (May 2026).
This is NOT Goodfire's official implementation -- it mirrors the training
recipe described in Section 3 of their paper so we can compare apples to apples.
"""

from __future__ import annotations

import math
import time

import torch

from svd_omp import recon


def run_vpd(
    W: torch.Tensor,
    C: int,
    k: int,
    n: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train the VPD (V, U, g) on one weight matrix.

    Args:
        W: weight matrix, shape [d_out, d_in].
        C: dictionary size.
        k: sparsity (unused during training; the metric loop reads top-k of g).
        n: number of training steps.
        seed: torch RNG seed for the random init.
        verbose: print intermediate loss every 50 steps.

    Returns:
        V, U, g: decomposition tensors after `n` steps. `support = topk(sigmoid(g), k)`.
    """
    d_out, d_in = W.shape
    dev = W.device
    torch.manual_seed(seed)
    V = torch.empty(d_in, C, device=dev).normal_(0, 1 / math.sqrt(d_in)).requires_grad_(True)
    U = torch.empty(C, d_out, device=dev).normal_(0, 1 / math.sqrt(C)).requires_grad_(True)
    g = torch.zeros(C, device=dev).requires_grad_(True)
    opt = torch.optim.AdamW([V, U, g], lr=3e-3)

    t0 = time.time()
    for step in range(n):
        p = max(0.4, 2.0 - 1.6 * (step / n))
        gi = torch.sigmoid(g)
        mask = gi + (1 - gi) * torch.rand(C, device=dev).detach()
        L_r = (recon(V, U, mask) - W).pow(2).mean()
        L_f = (recon(V, U, torch.ones(C, device=dev)) - W).pow(2).mean()
        L_s = (gi + 1e-12).pow(p).mean()
        (L_f + 0.5 * L_r + 0.02 * L_s).backward()
        opt.step()
        opt.zero_grad()
        if verbose and step % 50 == 0:
            print(f"    vpd {step:>3}/{n}  recon={L_r.item():.5f}  ({time.time() - t0:.1f}s)")
    return V.detach(), U.detach(), g.detach()


__all__ = ["run_vpd"]
