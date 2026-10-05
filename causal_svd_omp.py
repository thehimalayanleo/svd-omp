"""Causal-metric SVD-OMP: closed-form selection under a downstream output metric.

Plain SVD-OMP minimizes ``||W - W_hat||_F``.  Activation-whitened SVD-OMP
minimizes the layer-output error ``E ||(W - W_hat) phi||^2`` on real inputs.
Both weight every output dimension equally, so neither knows whether the rest
of the model reads the subspace an atom writes to.

This variant also weights the output side.  With input Gram ``G = L_g L_g^T``
and an output sensitivity metric ``M = L_m L_m^T`` it minimizes

    E [ e^T M e ],    e = (W - W_hat) phi

by taking the SVD of ``L_m^T W L_g = P S Q^T`` and mapping both sides back:

    read_c  = L_g^{-T} q_c      coefficient_c(phi) = s_c * read_c^T phi
    write_c = L_m^{-T} p_c      W phi = sum_c coefficient_c(phi) * write_c

The write vectors are orthonormal under ``M``, so the M-weighted error of any
support is the sum of squared coefficients left out.  Per-input top-k on
``|coefficient_c|`` is therefore still the exact fixed-width optimum.  No
training, no residual updates.

When ``M = E[g g^T]`` with ``g`` the gradient of sampled-label log-likelihood
with respect to the layer output, ``M`` is the Gauss-Newton Hessian of
KL(dense || perturbed) at the dense model.  The coefficient is then a
second-order estimate of each atom's effect on the model's output
distribution, not on this layer's output.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def sensitivity_metric(gradients: Tensor) -> Tensor:
    """Second moment ``E[g g^T]`` of output-side gradients, shape [d_out, d_out]."""

    if gradients.ndim != 2:
        raise ValueError("gradients must be a two-dimensional tensor")
    return gradients.T.matmul(gradients) / gradients.shape[0]


def _regularized_cholesky(matrix: Tensor, strength: float) -> tuple[Tensor, float]:
    scale = matrix.diagonal().mean().clamp_min(1e-30)
    regularization = strength * scale
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    return torch.linalg.cholesky(matrix + regularization * eye), float(regularization)


@torch.no_grad()
def causal_whitened_svd(
    weight: Tensor,
    calibration: Tensor,
    output_metric: Tensor | None,
    alpha: float,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    """Factor ``weight`` into atoms orthogonal under the input and output metrics.

    Args:
        weight: [d_out, d_in].
        calibration: layer inputs, [n, d_in].
        output_metric: [d_out, d_out] positive semidefinite, or None for the
            identity (which recovers activation-whitened SVD exactly).
        alpha, beta: ridge strengths relative to the mean diagonal of the
            input Gram and the output metric.

    Returns:
        write_vectors [d_out, r], singular_values [r], read_vectors [d_in, r],
        diagnostics.
    """

    gram = calibration.T.matmul(calibration) / calibration.shape[0]
    chol_in, input_regularization = _regularized_cholesky(gram, alpha)
    weighted = weight.matmul(chol_in)
    output_regularization = 0.0
    if output_metric is not None:
        chol_out, output_regularization = _regularized_cholesky(output_metric, beta)
        weighted = chol_out.T.matmul(weighted)
    left, singular_values, vh = torch.linalg.svd(weighted, full_matrices=False)
    read_vectors = torch.linalg.solve_triangular(chol_in.T, vh.T, upper=True)
    write_vectors = left
    if output_metric is not None:
        write_vectors = torch.linalg.solve_triangular(chol_out.T, left, upper=True)
    reconstruction = (write_vectors * singular_values.unsqueeze(0)).matmul(
        read_vectors.T
    )
    exact_error = float(
        (weight - reconstruction).norm() / weight.norm().clamp_min(1e-30)
    )
    return write_vectors, singular_values, read_vectors, {
        "alpha": alpha,
        "beta": beta,
        "input_regularization": input_regularization,
        "output_regularization": output_regularization,
        "weight_reconstruction_relative_error": exact_error,
    }


@torch.no_grad()
def select_top_k(
    activations: Tensor,
    write_vectors: Tensor,
    singular_values: Tensor,
    read_vectors: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Per-input top-k atoms by ``|s_c * read_c^T phi|``.

    Returns the layer-output approximation [B, d_out] and the support [B, k].
    """

    coefficients = activations.matmul(read_vectors) * singular_values.unsqueeze(0)
    indices = coefficients.abs().topk(k, dim=1).indices
    selected = torch.zeros_like(coefficients)
    selected.scatter_(1, indices, coefficients.gather(1, indices))
    return selected.matmul(write_vectors.T), indices


__all__ = ["causal_whitened_svd", "select_top_k", "sensitivity_metric"]
