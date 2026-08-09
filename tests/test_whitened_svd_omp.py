from __future__ import annotations

import torch

from whitened_svd_omp_discovery import (
    activation_whitened_svd,
    orthogonal_atom_curve,
)


def test_whitened_svd_is_exact_and_curve_reaches_zero() -> None:
    torch.manual_seed(7)
    weight = torch.randn(9, 6)
    calibration = torch.randn(20, 6)
    heldout = torch.randn(13, 6)
    singular_values, read_vectors, diagnostics = activation_whitened_svd(
        weight, calibration, alpha=1e-3, device="cpu"
    )
    assert diagnostics["weight_reconstruction_relative_error"] < 1e-5
    rows = orthogonal_atom_curve(
        weight,
        heldout,
        singular_values,
        read_vectors,
        bits_per_value=16,
        device="cpu",
    )
    assert min(row["relative_error"] for row in rows) < 1e-4
    assert max(row["mean_components"] for row in rows) == weight.shape[1]


def test_whitening_handles_rank_deficient_calibration() -> None:
    torch.manual_seed(11)
    weight = torch.randn(5, 8)
    calibration = torch.randn(3, 8)
    singular_values, read_vectors, diagnostics = activation_whitened_svd(
        weight, calibration, alpha=1e-2, device="cpu"
    )
    assert singular_values.shape == (5,)
    assert read_vectors.shape == (8, 5)
    assert diagnostics["weight_reconstruction_relative_error"] < 1e-4
