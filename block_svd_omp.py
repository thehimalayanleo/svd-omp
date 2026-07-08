"""
Block-SVD-OMP: block-sparse extension of SVD-OMP.

Motivation: Goodfire's BSF (Block-Sparse Featurizers, 2026) argues that
concepts in vision models are multi-dimensional (2-4 D) rather than single
directions, and that block-level sparsity recovers them faithfully. BSF pays
for this by training an encoder-decoder per layer.

Block-SVD-OMP is the training-free analog on weight matrices. Group the C
SVD components of W into K = ceil(C / r) contiguous blocks. Per input, score
each block by the L2 norm of that block's contribution to `W @ phi`:

    score_b(phi) = || diag(S_b) @ V_b^T phi ||_2

Select top-k_blocks blocks per input, reconstruct using only those. Because
SVD blocks are orthogonal in *both* V-space and U-space, block-OMP reduces
to closed-form top-k -- no residual updates needed, just like the 1D case.

Two selection modes:
  block-top-k    : one block group at a time (analogous to BSF's block TopK)
  block-lasso-approx : soft threshold on block norms, then top-k

We ship block-top-k here; the lasso variant is a 5-line change if needed.
"""

from __future__ import annotations

import torch
from torch import Tensor


def block_svd_decompose(
    W: Tensor,
    C: int,
    r: int,
) -> tuple[Tensor, Tensor, Tensor, list[tuple[int, int]]]:
    """SVD dictionary partitioned into contiguous rank-`r` blocks.

    Args:
        W: weight matrix, shape [d_out, d_in].
        C: number of singular components to keep.
        r: rank per block. C need not be divisible by r; the last block may
            be smaller.

    Returns:
        V_dict: right singular vectors,           shape [d_in, C].
        U_dict: left singular vectors * sigmas,   shape [C, d_out].
        S:      singular values,                  shape [C].
        blocks: list of (start, end) index ranges into the C-dim axis.
                len(blocks) == ceil(C / r).
    """
    from svd_omp import svd_decompose

    V_dict, U_dict, S = svd_decompose(W, C)
    C_eff = V_dict.shape[1]
    blocks = [(i, min(i + r, C_eff)) for i in range(0, C_eff, r)]
    return V_dict, U_dict, S, blocks


def block_svd_omp_select(
    phi_batch: Tensor,
    V_dict: Tensor,
    U_dict: Tensor,
    S: Tensor,
    blocks: list[tuple[int, int]],
    k_blocks: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-input block-sparse selection on the SVD block dictionary.

    Args:
        phi_batch: activations, shape [B, d_in].
        V_dict, U_dict, S: from `block_svd_decompose`.
        blocks: list of (start, end) index ranges.
        k_blocks: number of blocks to keep per input.

    Returns:
        W_hat:   sparse output reconstructions, shape [B, d_out].
        support: top-k block indices per input, shape [B, k_blocks].
        scores:  block scores, shape [B, K].
    """
    B = phi_batch.shape[0]
    K = len(blocks)
    dev = phi_batch.device

    # Precompute all sigma-weighted projections in one matmul.
    projs = phi_batch @ V_dict           # [B, C]
    weighted = projs * S.unsqueeze(0)    # [B, C], entry (i,c) = sigma_c * v_c^T phi_i

    # Compute per-block L2 norm scores.
    scores = torch.zeros(B, K, device=dev)
    for b, (s, e) in enumerate(blocks):
        scores[:, b] = weighted[:, s:e].norm(dim=1)

    topk = scores.topk(k_blocks, dim=1)
    support = topk.indices                    # [B, k_blocks]

    # Reconstruct: for each input, sum contributions of the selected blocks.
    W_hat = torch.zeros(B, U_dict.shape[1], device=dev)
    for i in range(B):
        for b in support[i].tolist():
            s, e = blocks[b]
            # (V_b^T phi_i) has shape [e-s], multiply by U_b (shape [e-s, d_out]).
            W_hat[i] += projs[i, s:e] @ U_dict[s:e]
    return W_hat, support, scores


def block_svd_omp_select_vectorized(
    phi_batch: Tensor,
    V_dict: Tensor,
    U_dict: Tensor,
    S: Tensor,
    blocks: list[tuple[int, int]],
    k_blocks: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Vectorized version of `block_svd_omp_select`. Requires all blocks to
    have equal size (`r`); if the last block is smaller, this falls back to
    the Python loop.
    """
    sizes = {e - s for s, e in blocks}
    if len(sizes) > 1:
        return block_svd_omp_select(phi_batch, V_dict, U_dict, S, blocks, k_blocks)

    r = next(iter(sizes))
    K = len(blocks)
    B = phi_batch.shape[0]
    dev = phi_batch.device

    projs = phi_batch @ V_dict           # [B, C]
    weighted = projs * S.unsqueeze(0)    # [B, C]
    # Reshape into blocks: [B, K, r].
    weighted_blocks = weighted.view(B, K, r)
    scores = weighted_blocks.norm(dim=2)    # [B, K]

    topk = scores.topk(k_blocks, dim=1)
    support = topk.indices                    # [B, k_blocks]

    # Build a boolean mask over C atoms selected by the top-k blocks.
    block_mask = torch.zeros(B, K, dtype=torch.bool, device=dev)
    block_mask.scatter_(1, support, True)
    atom_mask = block_mask.repeat_interleave(r, dim=1)   # [B, C]

    active_projs = projs * atom_mask         # zero out unselected atoms
    W_hat = active_projs @ U_dict            # [B, d_out]
    return W_hat, support, scores


def block_reconstruction(
    V_dict: Tensor,
    U_dict: Tensor,
    blocks: list[tuple[int, int]],
    active_blocks: list[int],
    d_out: int | None = None,
) -> Tensor:
    """Reconstruct W using only the specified blocks (input-agnostic mode).

    Corresponds to `recon(V, U, ones)` restricted to a subset of blocks.
    """
    from svd_omp import recon

    C = V_dict.shape[1]
    w = torch.zeros(C, device=V_dict.device)
    for b in active_blocks:
        s, e = blocks[b]
        w[s:e] = 1.0
    return recon(V_dict, U_dict, w)


__all__ = [
    "block_svd_decompose",
    "block_svd_omp_select",
    "block_svd_omp_select_vectorized",
    "block_reconstruction",
]
