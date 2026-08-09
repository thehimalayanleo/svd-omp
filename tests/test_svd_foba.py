from __future__ import annotations

import torch

from svd_foba import (
    build_overcomplete_dictionary,
    calibration_aware_svd_factors,
    reconstruct_with_svd_foba,
    svd_foba_curve,
)


def test_zero_candidates_recovers_svd_dictionary() -> None:
    torch.manual_seed(0)
    weight = torch.randn(7, 5)
    calibration = torch.randn(32, 5)
    output_atoms, _, _ = calibration_aware_svd_factors(weight, calibration, 0.1)
    dictionary = build_overcomplete_dictionary(
        weight, calibration, output_atoms, candidate_atoms=0, seed=0
    )
    torch.testing.assert_close(dictionary, output_atoms)


def test_svd_foba_protects_svd_fallback() -> None:
    torch.manual_seed(1)
    weight = torch.randn(10, 6)
    calibration = torch.randn(64, 6)
    evaluation = torch.randn(48, 6)
    rows = svd_foba_curve(
        weight,
        calibration,
        evaluation,
        (1, 2, 4),
        candidate_atoms=16,
        swap_rounds=2,
        proposal_width=4,
    )
    for row in rows:
        assert row["svd_foba_relative_error"] <= row["svd_relative_error"] + 1e-6


def test_overcomplete_foba_improves_anisotropic_outputs() -> None:
    torch.manual_seed(2)
    weight = torch.eye(4)
    direction = torch.tensor([1.0, 1.0, 0.0, 0.0])
    calibration = direction.unsqueeze(0) + 0.03 * torch.randn(128, 4)
    evaluation = direction.unsqueeze(0) + 0.03 * torch.randn(64, 4)
    rows = svd_foba_curve(
        weight,
        calibration,
        evaluation,
        (1,),
        alpha=1.0,
        candidate_atoms=32,
        swap_rounds=2,
        proposal_width=8,
    )
    assert rows[0]["svd_foba_relative_error"] < rows[0]["svd_relative_error"]


def test_reconstruction_uses_analysis_vectors_and_protects_svd() -> None:
    torch.manual_seed(3)
    weight = torch.randn(8, 5)
    calibration = torch.randn(96, 5)
    evaluation = torch.randn(32, 5)
    output_atoms, _, _ = calibration_aware_svd_factors(weight, calibration, 0.1)
    dictionary = build_overcomplete_dictionary(
        weight, calibration, output_atoms, candidate_atoms=24, seed=0
    )
    analysis_vectors = weight.T.matmul(dictionary)
    reconstruction, diagnostics = reconstruct_with_svd_foba(
        evaluation,
        dictionary,
        analysis_vectors,
        output_atoms.shape[1],
        selected_units=2,
        swap_rounds=2,
        proposal_width=4,
    )
    target = evaluation.matmul(weight.T)
    correlations = target.matmul(output_atoms)
    support = correlations.abs().topk(2, dim=1).indices
    baseline = torch.einsum(
        "nk,nkd->nd", correlations.gather(1, support), output_atoms.T[support]
    )
    assert reconstruction.shape == target.shape
    assert (target - reconstruction).square().sum() <= (target - baseline).square().sum() + 1e-5
    assert 0.0 <= diagnostics["fraction_inputs_selected_foba"] <= 1.0
