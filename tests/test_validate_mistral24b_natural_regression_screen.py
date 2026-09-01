import validate_mistral24b_natural_regression_screen as validator


def test_frozen_natural_screen_recomputes_as_negative():
    result = validator.validate()
    assert result["status"] == "validated_negative_natural_regression_screen"
    assert result["counts"]["qualified"] == 0
    assert result["counts"]["post_regression"] == 11
    assert result["promotion_gate_pass"] is False
