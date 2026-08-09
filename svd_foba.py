"""SVD-initialized forward-backward pursuit in an overcomplete output dictionary.

Plain FoBa cannot improve SVD top-k output reconstruction because the SVD
output atoms are orthogonal. This module keeps that exact solution as its
initial support, augments the dictionary with normalized calibration outputs,
and accepts only fixed-width forward-add/backward-remove swaps that lower the
least-squares reconstruction loss.
"""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def calibration_aware_svd_factors(
    weight: torch.Tensor,
    calibration: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return orthogonal output atoms, singular values, and analysis vectors."""

    gram = calibration.T.matmul(calibration) / calibration.shape[0]
    scale = gram.diagonal().mean().clamp_min(1e-30)
    gram = gram + alpha * scale * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    chol = torch.linalg.cholesky(gram)
    output_atoms, singular_values, vh = torch.linalg.svd(
        weight.matmul(chol), full_matrices=False
    )
    read_vectors = torch.linalg.solve_triangular(chol.T, vh.T, upper=True)
    return output_atoms, singular_values, read_vectors


@torch.no_grad()
def build_overcomplete_dictionary(
    weight: torch.Tensor,
    calibration: torch.Tensor,
    output_atoms: torch.Tensor,
    candidate_atoms: int,
    seed: int,
) -> torch.Tensor:
    """Append deterministic normalized calibration-output directions to SVD."""

    if candidate_atoms < 0:
        raise ValueError("candidate_atoms must be non-negative")
    if candidate_atoms == 0:
        return output_atoms
    calibration_outputs = calibration.matmul(weight.T)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = min(candidate_atoms, calibration_outputs.shape[0])
    indices = torch.randperm(
        calibration_outputs.shape[0], generator=generator, device="cpu"
    )[:count].to(calibration_outputs.device)
    candidates = calibration_outputs[indices].T
    candidates = candidates / candidates.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return torch.cat((output_atoms, candidates), dim=1)


def _support_system(
    gram: torch.Tensor,
    correlations: torch.Tensor,
    support: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, width = support.shape
    rows = support.unsqueeze(2).expand(batch, width, width)
    columns = support.unsqueeze(1).expand(batch, width, width)
    support_gram = gram[rows, columns]
    diagonal_scale = support_gram.diagonal(dim1=1, dim2=2).mean(dim=1)
    support_gram = support_gram + (
        ridge * diagonal_scale.clamp_min(1e-12)
    ).view(batch, 1, 1) * torch.eye(
        width, device=gram.device, dtype=gram.dtype
    ).unsqueeze(0)
    support_correlations = correlations.gather(1, support)
    coefficients = torch.linalg.solve(
        support_gram, support_correlations.unsqueeze(2)
    ).squeeze(2)
    inverse = torch.linalg.inv(support_gram)
    return support_gram, inverse, coefficients


def _exact_support_loss(
    gram: torch.Tensor,
    correlations: torch.Tensor,
    target_energy: torch.Tensor,
    support: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    batch, width = support.shape
    rows = support.unsqueeze(2).expand(batch, width, width)
    columns = support.unsqueeze(1).expand(batch, width, width)
    unregularized_gram = gram[rows, columns]
    support_correlations = correlations.gather(1, support)
    quadratic = torch.einsum(
        "ni,nij,nj->n", coefficients, unregularized_gram, coefficients
    )
    return (
        target_energy
        - 2.0 * (support_correlations * coefficients).sum(dim=1)
        + quadratic
    ).clamp_min(0)


@torch.no_grad()
def foba_refine_support(
    dictionary: torch.Tensor,
    target_correlations: torch.Tensor,
    target_energy: torch.Tensor,
    initial_support: torch.Tensor,
    *,
    swap_rounds: int = 2,
    proposal_width: int = 8,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run fixed-width FoBa swaps, accepting only strict LS improvements."""

    if swap_rounds < 0:
        raise ValueError("swap_rounds must be non-negative")
    if proposal_width <= 0:
        raise ValueError("proposal_width must be positive")
    gram = dictionary.T.matmul(dictionary)
    support = initial_support.clone()
    batch, width = support.shape
    accepted = torch.zeros(batch, device=dictionary.device, dtype=torch.int64)
    initial_loss = None

    for _ in range(swap_rounds):
        support_gram, inverse, coefficients = _support_system(
            gram, target_correlations, support, ridge
        )
        support_rows = gram[support]
        residual_correlations = target_correlations - torch.einsum(
            "nk,nkm->nm", coefficients, support_rows
        )
        residual_correlations.scatter_(1, support, 0.0)
        current_loss = _exact_support_loss(
            gram, target_correlations, target_energy, support, coefficients
        )
        if initial_loss is None:
            initial_loss = current_loss.clone()

        proposal_count = min(proposal_width, dictionary.shape[1] - width)
        proposals = residual_correlations.abs().topk(proposal_count, dim=1).indices
        cross = support_rows.gather(
            2, proposals.unsqueeze(1).expand(batch, width, proposal_count)
        )
        residualized = torch.linalg.solve(
            support_gram, cross
        )
        denominators = (
            gram.diagonal()[proposals] - (cross * residualized).sum(dim=1)
        ).clamp_min(1e-12)
        proposal_correlations = residual_correlations.gather(1, proposals)
        gains = proposal_correlations.square() / denominators
        best_proposal = proposals.gather(1, gains.argmax(dim=1, keepdim=True))

        augmented = torch.cat((support, best_proposal), dim=1)
        _, augmented_inverse, augmented_coefficients = _support_system(
            gram, target_correlations, augmented, ridge
        )
        removal_cost = augmented_coefficients.square() / augmented_inverse.diagonal(
            dim1=1, dim2=2
        ).clamp_min(1e-12)
        remove_position = removal_cost.argmin(dim=1)
        positions = torch.arange(width + 1, device=dictionary.device).unsqueeze(0)
        keep = positions != remove_position.unsqueeze(1)
        proposal_support = augmented[keep].reshape(batch, width)
        _, _, proposal_coefficients = _support_system(
            gram, target_correlations, proposal_support, ridge
        )
        proposal_loss = _exact_support_loss(
            gram,
            target_correlations,
            target_energy,
            proposal_support,
            proposal_coefficients,
        )
        improve = proposal_loss < current_loss - 1e-8 * target_energy.clamp_min(1e-12)
        support = torch.where(improve.unsqueeze(1), proposal_support, support)
        accepted += improve

    _, _, final_coefficients = _support_system(
        gram, target_correlations, support, ridge
    )
    final_loss = _exact_support_loss(
        gram, target_correlations, target_energy, support, final_coefficients
    )
    if initial_loss is None:
        initial_loss = final_loss.clone()
    diagnostics = {
        "mean_accepted_swaps": float(accepted.float().mean().item()),
        "fraction_inputs_improved": float((final_loss < initial_loss).float().mean().item()),
        "mean_relative_loss_reduction": float(
            ((initial_loss - final_loss) / initial_loss.clamp_min(1e-30)).mean().item()
        ),
    }
    return support, final_loss, diagnostics


@torch.no_grad()
def reconstruct_with_svd_foba(
    inputs: torch.Tensor,
    dictionary: torch.Tensor,
    analysis_vectors: torch.Tensor,
    svd_rank: int,
    selected_units: int,
    *,
    swap_rounds: int = 2,
    proposal_width: int = 8,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Reconstruct outputs without materializing the dense target matmul.

    The first ``svd_rank`` dictionary columns must be the orthonormal SVD output
    atoms. ``analysis_vectors`` equals ``W.T @ dictionary``, so every target
    correlation is available directly from the module input.
    """

    if not 1 <= selected_units <= svd_rank:
        raise ValueError("selected_units must be between one and svd_rank")
    correlations = inputs.matmul(analysis_vectors)
    target_energy = correlations[:, :svd_rank].square().sum(dim=1)
    initial_support = correlations[:, :svd_rank].abs().topk(
        selected_units, dim=1
    ).indices
    initial_coefficients = correlations.gather(1, initial_support)
    initial_atoms = dictionary.T[initial_support]
    initial_reconstruction = torch.einsum(
        "nk,nkd->nd", initial_coefficients, initial_atoms
    )
    initial_loss = (
        target_energy - initial_coefficients.square().sum(dim=1)
    ).clamp_min(0)

    support, _, diagnostics = foba_refine_support(
        dictionary,
        correlations,
        target_energy,
        initial_support,
        swap_rounds=swap_rounds,
        proposal_width=proposal_width,
        ridge=ridge,
    )
    gram = dictionary.T.matmul(dictionary)
    _, _, coefficients = _support_system(
        gram, correlations, support, ridge
    )
    foba_loss = _exact_support_loss(
        gram, correlations, target_energy, support, coefficients
    )
    foba_atoms = dictionary.T[support]
    foba_reconstruction = torch.einsum(
        "nk,nkd->nd", coefficients, foba_atoms
    )
    improved = foba_loss < initial_loss
    reconstruction = torch.where(
        improved.unsqueeze(1), foba_reconstruction, initial_reconstruction
    )
    diagnostics = {
        **diagnostics,
        "fraction_inputs_selected_foba": float(improved.float().mean().item()),
        "mean_protected_relative_loss_reduction": float(
            (
                (initial_loss - torch.minimum(foba_loss, initial_loss))
                / initial_loss.clamp_min(1e-30)
            ).mean().item()
        ),
    }
    return reconstruction, diagnostics


@torch.no_grad()
def svd_foba_curve(
    weight: torch.Tensor,
    calibration: torch.Tensor,
    evaluation: torch.Tensor,
    ks: tuple[int, ...],
    *,
    alpha: float = 0.1,
    candidate_atoms: int = 128,
    seed: int = 0,
    swap_rounds: int = 2,
    proposal_width: int = 8,
) -> list[dict[str, Any]]:
    """Evaluate protected SVD top-k and SVD-FoBa at each selected-unit width."""

    output_atoms, singular_values, read_vectors = calibration_aware_svd_factors(
        weight, calibration, alpha
    )
    dictionary = build_overcomplete_dictionary(
        weight, calibration, output_atoms, candidate_atoms, seed
    )
    targets = evaluation.matmul(weight.T)
    target_energy = targets.square().sum(dim=1)
    correlations = targets.matmul(dictionary)
    svd_coefficients = evaluation.matmul(read_vectors) * singular_values.unsqueeze(0)
    rows = []
    rank = output_atoms.shape[1]
    for requested_k in ks:
        k = min(requested_k, rank)
        support = svd_coefficients.abs().topk(k, dim=1).indices
        baseline_loss = (
            target_energy
            - svd_coefficients.gather(1, support).square().sum(dim=1)
        ).clamp_min(0)
        _, foba_loss, diagnostics = foba_refine_support(
            dictionary,
            correlations,
            target_energy,
            support,
            swap_rounds=swap_rounds,
            proposal_width=proposal_width,
        )
        foba_loss = torch.minimum(foba_loss, baseline_loss)
        diagnostics["fraction_inputs_improved"] = float(
            (foba_loss < baseline_loss).float().mean().item()
        )
        diagnostics["mean_relative_loss_reduction"] = float(
            (
                (baseline_loss - foba_loss)
                / baseline_loss.clamp_min(1e-30)
            ).mean().item()
        )
        total = target_energy.sum().clamp_min(1e-30)
        rows.append(
            {
                "selected_units": k,
                "svd_relative_error": float(torch.sqrt(baseline_loss.sum() / total).item()),
                "svd_foba_relative_error": float(torch.sqrt(foba_loss.sum() / total).item()),
                **diagnostics,
            }
        )
    return rows
