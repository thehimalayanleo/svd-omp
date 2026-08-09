from __future__ import annotations

import torch

from mdl_svdomp_vs_swd_natural_24 import shared_log2_advantage, support_adaptivity


def test_shared_log2_advantage_counts_matched_points() -> None:
    omp = [
        {"relative_error": 0.1, "total_bits": 80.0},
        {"relative_error": 0.2, "total_bits": 60.0},
        {"relative_error": 0.3, "total_bits": 40.0},
    ]
    swd = [
        {"relative_error": 0.1, "total_bits": 100.0},
        {"relative_error": 0.2, "total_bits": 50.0},
        {"relative_error": 0.3, "total_bits": 30.0},
    ]
    result = shared_log2_advantage(omp, swd)
    assert result["comparable_swd_points"] == 3
    assert result["swd_points_won_by_svd"] == 1


def test_support_adaptivity_is_bounded() -> None:
    torch.manual_seed(3)
    weight = torch.randn(12, 8)
    activations = torch.randn(32, 8)
    result = support_adaptivity(weight, activations, k=3, device="cpu", sample_size=32)
    assert 1 <= result["unique_supports"] <= 32
    assert 0 < result["unique_support_fraction"] <= 1
    assert 0 <= result["mean_paired_jaccard"] <= 1

