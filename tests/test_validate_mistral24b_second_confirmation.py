import validate_mistral24b_second_confirmation as validator


def test_second_stage_confirmation_recomputes_as_pass():
    result = validator.validate()
    assert result["status"] == "validated_second_stage_multiseed_confirmation_pass"
    assert result["support_budget"] == 224
    assert result["dictionary_atoms"] == 640
    assert result["total_bidirectional"] == 42
    assert result["total_sources"] == 48
    assert all(seed["empirical_p"] == 0.01 for seed in result["seeds"].values())
