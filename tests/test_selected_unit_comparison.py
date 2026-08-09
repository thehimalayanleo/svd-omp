from __future__ import annotations

import torch

from selected_unit_svdomp_vs_swd import greedy_swd_selected_unit_curve


def test_greedy_swd_curve_is_monotone() -> None:
    torch.manual_seed(13)
    weight = torch.randn(7, 5)
    evaluation = torch.randn(19, 5)
    factor_a = torch.randn(5, 6)
    factor_b = torch.randn(6, 7)
    rows = greedy_swd_selected_unit_curve(
        weight,
        evaluation,
        factor_a,
        factor_b,
        ks=(1, 2, 4, 6),
        device="cpu",
    )
    errors = [row["relative_error"] for row in rows]
    edges = [row["mean_active_edges"] for row in rows]
    assert errors == sorted(errors, reverse=True)
    assert edges == sorted(edges)
