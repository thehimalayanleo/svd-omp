from validate_mistral24b_position_bias_development import validate


def test_mistral24b_development_result_is_auditable_negative() -> None:
    summary = validate()
    assert summary["parameters"] == 24_011_361_280
    assert summary["final_test_mounted"] is False
    assert summary["random_supports"] == 39
    assert summary["random_empirical_p"] == 1.0


def test_all_frozen_selectors_produce_zero_repairs() -> None:
    summary = validate()
    for method in summary["methods"].values():
        assert method["selected_dose"] == 0.0
        assert method["development_repairs"] == 0
        assert method["validation_repairs"] == 0
        assert method["validation_protected_minimum"] == 16
