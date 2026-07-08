"""
Stable-rank analysis of block dictionaries.

Reproduces the BSF stable-rank plot: for a block of size K, measure the
effective number of dimensions that the block actually uses on activation
projections.

    stable_rank(M) = || M ||_F^2 / || M ||_2^2

If all rows of M point in the same direction, stable_rank = 1.
If rows use all r dimensions equally, stable_rank = r.

BSF (Bricken et al., Goodfire 2026) reports stable rank plateauing at ~4
regardless of K on vision models -- evidence that vision concepts are
naturally 2-to-4-dimensional. Our analytic SVD blocks are full-rank by
construction, so the interesting quantity is the activation stable rank:

    A_b(phi_batch) = phi_batch @ V_b       # [B, K]  (projections onto block)
    activation_stable_rank_b = stable_rank(A_b)

This tells us how many dimensions of the block subspace the activations
actually populate. If it's ~4 regardless of K, we've independently
reproduced BSF's finding from an analytic dictionary.
"""

from __future__ import annotations

import torch


def stable_rank(M: torch.Tensor) -> float:
    """Stable rank of a matrix M: || M ||_F^2 / || M ||_2^2."""
    if M.numel() == 0:
        return 0.0
    fro_sq = M.pow(2).sum()
    op_sq = torch.linalg.svdvals(M).max() ** 2
    return (fro_sq / op_sq.clamp(min=1e-12)).item()


def activation_stable_rank_per_block(
    V: torch.Tensor,
    blocks: list[tuple[int, int]],
    phi_batch: torch.Tensor,
) -> list[float]:
    """Per-block stable rank of activations projected into each block subspace.

    Args:
        V: block dictionary of shape [d_in, C] whose columns partition into
           blocks.
        blocks: (start, end) index ranges into the second axis of V.
        phi_batch: activations, shape [B, d_in].

    Returns:
        list of length len(blocks) giving the stable rank of the B x r
        matrix of projections for each block.
    """
    ranks = []
    for s, e in blocks:
        # A_b = phi_batch @ V_b  -- shape [B, e-s]
        A_b = phi_batch @ V[:, s:e]
        ranks.append(stable_rank(A_b))
    return ranks


__all__ = ["stable_rank", "activation_stable_rank_per_block"]
