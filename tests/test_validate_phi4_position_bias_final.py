from validate_phi4_position_bias_final import build_summary


def test_final_summary_recomputes_frozen_gates() -> None:
    summary = build_summary()
    assert summary["full_preregistered_claim_pass"] is False
    assert summary["positive_specific_repair_all_seeds"] is True
    assert summary["matched_random_pass_all_seeds"] is True
    assert [item["specific_repairs"] for item in summary["seeds"]] == [20, 13, 7]
    assert [item["final_seed_pass"] for item in summary["seeds"]] == [True, True, False]


def test_final_summary_preserves_all_protected_behaviors() -> None:
    summary = build_summary()
    assert min(item["protected_minimum"] for item in summary["seeds"]) == 23
    assert all(item["shortcut_repairs"] == 0 for item in summary["seeds"])
    assert all(item["paired_damage"] == 0 for item in summary["seeds"])
