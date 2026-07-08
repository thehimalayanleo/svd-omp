"""
Trainable SVD-OMP with a non-Frobenius objective.

The Eckart-Young theorem caps every rank-k reconstruction of W at truncated
SVD. It does NOT cap reconstruction of any downstream composition of W. When
the atoms are selected to preserve a *nonlinearly-composed* signal (e.g.
GELU(W phi) @ W_next^T) rather than W phi itself, the analytic SVD selection
is generally not optimal, and training can beat it.

This module implements a block-TopK SVD-OMP whose training objective is
downstream reconstruction:

    L = || GELU((phi V_masked) U) W_next^T  -  GELU((phi W^T)) W_next^T ||^2

Two modes match the Frobenius trainable-SVD-OMP:
    scaffold  -- freeze V, U at SVD; learn only per-block scale s_b and bias b_b.
    full      -- V, U also trainable (warm-started from SVD).

Selection at inference uses the learned scale + bias on top of the analytic
block-projection score (same closed form as block-SVD-OMP with a learned
scalar correction per block).
"""

from __future__ import annotations

import time
from typing import Literal

import torch
import torch.nn.functional as F

from svd_omp import svd_decompose


def _split_into_blocks(C: int, r: int) -> list[tuple[int, int]]:
    C_bl = (C // r) * r
    return [(i, i + r) for i in range(0, C_bl, r)]


def _downstream(z: torch.Tensor, W_next: torch.Tensor | None, act: str = "gelu") -> torch.Tensor:
    """Compose nonlinearity(z) @ W_next.T. Falls back to identity if W_next is None."""
    if W_next is None:
        return z
    if act == "gelu":
        z = F.gelu(z)
    elif act == "relu":
        z = F.relu(z)
    elif act == "identity":
        pass
    else:
        raise ValueError(act)
    return z @ W_next.T


def run_causal_trainable_svd_omp(
    W: torch.Tensor,
    W_next: torch.Tensor | None,
    C: int,
    r: int,
    k_blocks: int,
    n: int = 200,
    seed: int = 0,
    mode: Literal["scaffold", "full"] = "scaffold",
    verbose: bool = True,
    batch_size: int = 128,
    lr: float = 3e-3,
    faith_anchor: float = 0.1,
    activation: str = "gelu",
    phi_scale: float = 0.5,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, list[tuple[int, int]],
]:
    """Train block-TopK selection to preserve downstream (nonlinear) output.

    Args:
        W: this layer's weight matrix, shape [d_out, d_in].
        W_next: next layer's weight, shape [d_next, d_out]. If None, falls
            back to identity composition (equivalent to Frobenius trainable).
        C: dictionary size (capped internally at min(d_out, d_in) rounded to r).
        r: block rank.
        k_blocks: number of active blocks per input.
        n: training steps.
        seed: RNG seed for phi sampling.
        mode: 'scaffold' (freeze V, U) or 'full' (also fine-tune V, U).
        faith_anchor: weight on the anchor term keeping V, U close to SVD
            (only used in 'full' mode).

    Returns:
        V, U, S, ln_scale, bias, blocks.
    """
    assert C >= r
    d_out, d_in = W.shape
    dev = W.device
    C_requested = (C // r) * r
    max_svd = min(d_out, d_in)
    C_eff = min(C_requested, (max_svd // r) * r)
    blocks = _split_into_blocks(C_eff, r)
    K = len(blocks)

    V_svd, U_svd, S = svd_decompose(W, C_eff)
    if mode == "scaffold":
        V = V_svd.detach()
        U = U_svd.detach()
        params_V, params_U = None, None
    else:
        V = V_svd.clone().detach().requires_grad_(True)
        U = U_svd.clone().detach().requires_grad_(True)
        params_V, params_U = V, U
    S = S.detach()

    ln_scale = torch.zeros(K, device=dev, requires_grad=True)
    bias = torch.zeros(K, device=dev, requires_grad=True)

    params = [ln_scale, bias]
    if params_V is not None:
        params += [params_V, params_U]

    opt = torch.optim.AdamW(params, lr=lr)
    torch.manual_seed(seed)
    t0 = time.time()

    for step in range(n):
        phi = torch.randn(batch_size, d_in, device=dev) * phi_scale

        # Ground-truth downstream signal.
        y_true = _downstream(phi @ W.T, W_next, act=activation)     # [B, d_next or d_out]

        # Block scores.
        projs = phi @ V                                             # [B, C_eff]
        weighted = (projs * S.unsqueeze(0)).view(batch_size, K, r)
        block_norms = weighted.norm(dim=2)                           # [B, K]
        scale = torch.exp(ln_scale).unsqueeze(0)                     # [1, K]
        scores = scale * block_norms + bias.unsqueeze(0)             # [B, K]

        # STE block-TopK.
        topk_idx = scores.topk(k_blocks, dim=1).indices
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, topk_idx, 1.0)
        soft = scores.softmax(dim=1)
        block_gate = hard_mask + (soft - soft.detach())              # [B, K]
        atom_mask = block_gate.repeat_interleave(r, dim=1)          # [B, C_eff]

        # Reconstructed downstream signal.
        z_hat = (projs * atom_mask) @ U                              # [B, d_out]
        y_hat = _downstream(z_hat, W_next, act=activation)          # [B, d_next or d_out]

        loss = (y_hat - y_true).pow(2).mean()

        # Anchor V, U to the SVD basis in full mode.
        if mode == "full" and faith_anchor > 0:
            atom_ones = torch.ones(C_eff, device=dev)
            L_faith = ((V @ (atom_ones.unsqueeze(1) * U)).T - W).pow(2).mean()
            loss = loss + faith_anchor * L_faith

        loss.backward()
        opt.step()
        opt.zero_grad()

        if verbose and step % 50 == 0:
            print(f"    causal-svd-omp {step:>3}/{n} ({mode})  "
                  f"loss={loss.item():.5f}  ({time.time() - t0:.1f}s)")

    return (
        V.detach() if isinstance(V, torch.Tensor) else V,
        U.detach() if isinstance(U, torch.Tensor) else U,
        S,
        ln_scale.detach(),
        bias.detach(),
        blocks,
    )


def causal_trainable_select(
    phi_batch: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    ln_scale: torch.Tensor,
    bias: torch.Tensor,
    blocks: list[tuple[int, int]],
    k_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic block-TopK using the learned scale + bias."""
    B = phi_batch.shape[0]
    K = len(blocks)
    r = blocks[0][1] - blocks[0][0]

    projs = phi_batch @ V
    weighted = (projs * S.unsqueeze(0)).view(B, K, r)
    block_norms = weighted.norm(dim=2)
    scale = torch.exp(ln_scale).unsqueeze(0)
    scores = scale * block_norms + bias.unsqueeze(0)

    topk = scores.topk(k_blocks, dim=1)
    support = topk.indices
    mask = torch.zeros_like(scores)
    mask.scatter_(1, support, 1.0)
    atom_mask = mask.repeat_interleave(r, dim=1)
    z_hat = (projs * atom_mask) @ U
    return z_hat, support, scores


def downstream_mse(
    phi_batch: torch.Tensor,
    W: torch.Tensor,
    W_next: torch.Tensor | None,
    z_hat: torch.Tensor,
    activation: str = "gelu",
) -> float:
    """Mean-squared error of act(z_hat) @ W_next^T against the ground truth
    act(phi @ W.T) @ W_next^T. If W_next is None, falls back to intermediate MSE.
    """
    y_true = _downstream(phi_batch @ W.T, W_next, act=activation)
    y_hat = _downstream(z_hat, W_next, act=activation)
    return (y_hat - y_true).pow(2).mean().item()


__all__ = [
    "run_causal_trainable_svd_omp",
    "causal_trainable_select",
    "downstream_mse",
]
