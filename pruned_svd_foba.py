"""Calibration-pruned SVD-FoBa for lower selector and dictionary cost.

The full SVD-FoBa scorer evaluates every SVD direction plus all calibration
atoms for every input.  This variant uses calibration data to freeze a smaller
eligible SVD pool, then runs the same protected fixed-width FoBa refinement
inside that pool.  The selected output width is unchanged; only the number of
directions that must be stored and scored is reduced.
"""

from __future__ import annotations

from typing import Any

import torch

from svd_foba import (
    build_overcomplete_dictionary,
    calibration_aware_svd_factors,
    foba_refine_support,
)


@torch.no_grad()
def calibration_selected_pool(
    coefficients: torch.Tensor,
    pool_size: int,
    selection_width: int,
) -> torch.Tensor:
    """Freeze atoms with the most calibration energy inside per-input top-k.

    A raw mean-square score collapses to the usual global SVD ordering.  The
    masked score below instead preserves directions that are locally important
    for at least some calibration inputs, which is the source of the dynamic
    selected-unit advantage.
    """

    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a two-dimensional tensor")
    rank = coefficients.shape[1]
    if not 1 <= pool_size <= rank:
        raise ValueError("pool_size must be between one and the SVD rank")
    if selection_width <= 0:
        raise ValueError("selection_width must be positive")
    local_width = min(selection_width, rank)
    values, indices = coefficients.abs().topk(local_width, dim=1)
    scores = torch.zeros(rank, device=coefficients.device, dtype=coefficients.dtype)
    scores.scatter_add_(0, indices.reshape(-1), values.square().reshape(-1))
    # Add a tiny deterministic global-energy tie breaker.  It cannot change a
    # meaningful masked score but makes zero-score ordering reproducible.
    global_energy = coefficients.square().mean(dim=0)
    scale = scores.max().clamp_min(1.0)
    scores = scores + global_energy / global_energy.max().clamp_min(1e-30) * scale * 1e-12
    return scores.topk(pool_size).indices.sort().values


@torch.no_grad()
def pruned_svd_foba_curve(
    weight: torch.Tensor,
    calibration: torch.Tensor,
    evaluation: torch.Tensor,
    ks: tuple[int, ...],
    *,
    alpha: float = 0.1,
    pool_size: int,
    pool_selection_width: int = 64,
    candidate_atoms: int = 32,
    seed: int = 0,
    swap_rounds: int = 1,
    proposal_width: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a calibration-frozen scoring pool on held-out activations."""

    output_atoms, singular_values, read_vectors = calibration_aware_svd_factors(
        weight, calibration, alpha
    )
    rank = output_atoms.shape[1]
    actual_pool_size = min(pool_size, rank)
    if max(ks) > actual_pool_size:
        raise ValueError("pool_size must be at least the largest selected width")

    calibration_coefficients = (
        calibration.matmul(read_vectors) * singular_values.unsqueeze(0)
    )
    pool = calibration_selected_pool(
        calibration_coefficients,
        actual_pool_size,
        pool_selection_width,
    )
    pooled_atoms = output_atoms[:, pool]
    dictionary = build_overcomplete_dictionary(
        weight,
        calibration,
        pooled_atoms,
        candidate_atoms,
        seed,
    )

    targets = evaluation.matmul(weight.T)
    target_energy = targets.square().sum(dim=1)
    correlations = targets.matmul(dictionary)
    full_coefficients = (
        evaluation.matmul(read_vectors) * singular_values.unsqueeze(0)
    )
    pooled_correlations = correlations[:, :actual_pool_size]
    rows: list[dict[str, Any]] = []
    total = target_energy.sum().clamp_min(1e-30)
    input_width = weight.shape[1]
    output_width = weight.shape[0]
    dictionary_width = dictionary.shape[1]
    full_dictionary_width = rank + 128

    for requested_k in ks:
        k = min(requested_k, actual_pool_size)
        initial_support = pooled_correlations.abs().topk(k, dim=1).indices
        initial_coefficients = pooled_correlations.gather(1, initial_support)
        initial_loss = (
            target_energy - initial_coefficients.square().sum(dim=1)
        ).clamp_min(0)
        full_support = full_coefficients.abs().topk(k, dim=1).indices
        full_svd_loss = (
            target_energy - full_coefficients.gather(1, full_support).square().sum(dim=1)
        ).clamp_min(0)

        if swap_rounds:
            _, refined_loss, diagnostics = foba_refine_support(
                dictionary,
                correlations,
                target_energy,
                initial_support,
                swap_rounds=swap_rounds,
                proposal_width=proposal_width,
            )
            final_loss = torch.minimum(refined_loss, initial_loss)
        else:
            final_loss = initial_loss
            diagnostics = {
                "mean_accepted_swaps": 0.0,
                "fraction_inputs_improved": 0.0,
                "mean_relative_loss_reduction": 0.0,
            }

        rows.append(
            {
                "selected_units": k,
                "full_svd_relative_error": float(
                    torch.sqrt(full_svd_loss.sum() / total).item()
                ),
                "pooled_svd_relative_error": float(
                    torch.sqrt(initial_loss.sum() / total).item()
                ),
                "pruned_foba_relative_error": float(
                    torch.sqrt(final_loss.sum() / total).item()
                ),
                "selected_active_edges": k * (input_width + output_width),
                **diagnostics,
            }
        )

    metadata = {
        "svd_rank": rank,
        "pool_size": actual_pool_size,
        "pool_selection_width": min(pool_selection_width, rank),
        "candidate_atoms": dictionary_width - actual_pool_size,
        "dictionary_width": dictionary_width,
        "full_foba_dictionary_width": full_dictionary_width,
        "selector_read_macs_per_input": input_width * dictionary_width,
        "full_foba_selector_read_macs_per_input": input_width * full_dictionary_width,
        "selector_read_fraction_of_full_foba": dictionary_width / full_dictionary_width,
        "stored_analysis_and_dictionary_elements": (
            input_width + output_width
        ) * dictionary_width,
        "full_foba_stored_analysis_and_dictionary_elements": (
            input_width + output_width
        ) * full_dictionary_width,
        "storage_fraction_of_full_foba": dictionary_width / full_dictionary_width,
        "pool_indices": pool.cpu().tolist(),
    }
    return rows, metadata
