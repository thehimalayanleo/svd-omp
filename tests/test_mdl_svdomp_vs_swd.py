from __future__ import annotations

import math

import numpy as np
import torch

from mdl_svdomp_vs_swd import (
    dominance_intervals,
    interpolated_comparison,
    log2_choose,
    pareto_frontier,
    sparse_array_code_bits,
    svd_omp_curve,
)


def test_combinatorial_codes() -> None:
    assert log2_choose(8, 0) == 0
    assert math.isclose(log2_choose(8, 2), math.log2(28))
    assert math.isclose(sparse_array_code_bits(8, 2, 16), math.log2(28) + 32)


def test_svd_omp_curve_uses_output_orientation_and_reaches_zero() -> None:
    torch.manual_seed(4)
    weight = torch.randn(7, 5)
    activations = torch.randn(11, 5)
    rows, diagnostics = svd_omp_curve(
        weight,
        activations,
        bits_per_value=16,
        per_token_targets=np.array([0.0, 0.5, 1.0]),
    )
    shared = [row for row in rows if row["method"] == "svd_omp_shared_dictionary"]
    assert diagnostics["rank"] == 5
    assert min(row["relative_error"] for row in shared) < 1e-5
    assert max(row["max_components"] for row in shared) == 5
    assert min(row["min_components"] for row in shared) == 0


def test_pareto_and_matched_error_intervals() -> None:
    omp = [
        {"relative_error": 0.1, "total_bits": 100.0},
        {"relative_error": 0.2, "total_bits": 70.0},
        {"relative_error": 0.3, "total_bits": 80.0},
    ]
    swd = [
        {"relative_error": 0.1, "total_bits": 90.0},
        {"relative_error": 0.2, "total_bits": 75.0},
    ]
    assert len(pareto_frontier(omp)) == 2
    intervals = dominance_intervals(omp, swd)
    assert [row["winner"] for row in intervals] == ["swd", "svd_omp"]
    interpolated = interpolated_comparison(omp, swd)
    assert len(interpolated["crossovers"]) == 1
    assert interpolated["regions"][0]["winner"] == "swd"
    assert interpolated["regions"][1]["winner"] == "svd_omp"
