import torch

from pruned_svd_foba import calibration_selected_pool, pruned_svd_foba_curve


def test_calibration_pool_preserves_locally_important_direction() -> None:
    coefficients = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 0.1],
        ]
    )
    pool = calibration_selected_pool(coefficients, pool_size=3, selection_width=1)
    assert set(pool.tolist()) == {0, 1, 2}


def test_pruned_foba_is_protected_and_reports_cost_reduction() -> None:
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(10, 12, generator=generator)
    calibration = torch.randn(64, 12, generator=generator)
    evaluation = torch.randn(32, 12, generator=generator)
    rows, metadata = pruned_svd_foba_curve(
        weight,
        calibration,
        evaluation,
        (1, 2, 4),
        pool_size=6,
        pool_selection_width=4,
        candidate_atoms=2,
        swap_rounds=2,
    )
    assert all(
        row["pruned_foba_relative_error"]
        <= row["pooled_svd_relative_error"] + 1e-7
        for row in rows
    )
    assert metadata["dictionary_width"] == 8
    assert metadata["selector_read_fraction_of_full_foba"] < 1.0


def test_zero_swap_matches_pooled_svd() -> None:
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(8, 8, generator=generator)
    calibration = torch.randn(32, 8, generator=generator)
    evaluation = torch.randn(16, 8, generator=generator)
    rows, _ = pruned_svd_foba_curve(
        weight,
        calibration,
        evaluation,
        (1, 3),
        pool_size=4,
        candidate_atoms=0,
        swap_rounds=0,
    )
    assert all(
        row["pruned_foba_relative_error"] == row["pooled_svd_relative_error"]
        for row in rows
    )
