from validate_mistral24b_bidirectional_expansion import validate


def test_24b_expansion_validates_dense_cause_and_sparse_failure() -> None:
    result = validate()
    assert result["exact_atoms"] == 640
    assert result["dense_cycle_pass"] is True
    assert result["dense_bidirectional_counts"] == {
        "development_a": 13,
        "development_b": 13,
    }
    assert result["all_sparse_methods_repaired_zero"] is True
    assert result["final_test_mounted"] is False
