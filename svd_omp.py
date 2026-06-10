"""
SVD-OMP: training-free parameter decomposition via the SVD basis.

Given a weight matrix W of shape [d_out, d_in], we use its SVD
    W = U S V^T
as a deterministic, orthogonal dictionary of rank-1 atoms
    {sigma_c * u_c v_c^T},  c = 1..C.

Two selection rules are exposed:

  1. Static top-k (Eckart-Young):
        support = argsort(S, descending=True)[:k]
     Optimal weight-level rank-k approximation, independent of input.

  2. Per-input OMP on the SVD basis:
        score_c(phi) = sigma_c * |v_c^T phi|
        support(phi) = top-k indices of score_c
     Because the SVD basis is orthogonal, OMP reduces in closed form
     to this top-k. The sigma weighting picks atoms that explain the
     output (W @ phi), not the input (phi).

No training, no random initialization, no learned parameters.
"""

from __future__ import annotations

import torch
from torch import Tensor


def recon(V: Tensor, U: Tensor, w: Tensor) -> Tensor:
    """Reconstruct a [d_out, d_in] matrix from C rank-1 atoms.

    Atom c contributes w[c] * outer(U[c], V[:, c]).
    """
    return (V @ (w.unsqueeze(1) * U)).T


def svd_decompose(W: Tensor, C: int) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the rank-`C` SVD dictionary for weight matrix W.

    Args:
        W: weight matrix, shape [d_out, d_in].
        C: number of components to keep (<= min(d_out, d_in)).

    Returns:
        V_dict: right singular vectors, shape [d_in, C].
        U_dict: left singular vectors * singular values, shape [C, d_out].
                Folding S into U lets `recon(V_dict, U_dict, ones)` reproduce W.
        S:      singular values, shape [C], descending.
    """
    Usvd, S, Vt = torch.linalg.svd(W, full_matrices=False)
    n = min(C, S.shape[0])
    V_dict = Vt[:n].T.contiguous().detach()
    U_dict = (Usvd[:, :n] * S[:n]).T.contiguous().detach()
    return V_dict, U_dict, S[:n].detach()


def svd_omp_select(
    phi_batch: Tensor,
    V_dict: Tensor,
    U_dict: Tensor,
    S: Tensor,
    k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-input top-k selection on the SVD basis.

    Scores atoms by ``sigma_c * |v_c^T phi|`` and reconstructs the output
    ``W @ phi`` using only the top-k atoms per input.

    Args:
        phi_batch: residual stream activations, shape [B, d_in].
        V_dict, U_dict, S: dictionary from `svd_decompose`.
        k: sparsity (number of active components per input).

    Returns:
        W_hat:   sparse output reconstructions, shape [B, d_out].
        support: top-k indices per input, shape [B, k].
        scores:  full sigma-weighted score matrix, shape [B, C].
    """
    projs = phi_batch @ V_dict
    scores = projs.abs() * S.unsqueeze(0)

    topk = scores.topk(k, dim=1)
    support = topk.indices

    w_selected = projs.gather(1, support)
    mask = torch.zeros_like(projs)
    mask.scatter_(1, support, w_selected)
    W_hat = mask @ U_dict
    return W_hat, support, scores


__all__ = ["recon", "svd_decompose", "svd_omp_select"]
