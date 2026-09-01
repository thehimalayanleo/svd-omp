from validate_paper_causal_campaigns import validate


def test_paper_campaign_decisions_and_denominators() -> None:
    report = validate()
    campaigns = report["campaigns"]
    assert report["status"] == "paper_causal_campaigns_validated"
    assert campaigns["qwen30b"]["all_seeds_pass"] is False
    assert campaigns["qwen30b"]["passed_seeds"] == 1
    assert campaigns["qwen30b"]["all_behavioral_seeds_pass"] is True
    assert campaigns["qwen30b"]["behavioral_passed_seeds"] == 3
    assert campaigns["fresh_mistral24b"]["all_seeds_pass"] is False
    assert campaigns["fresh_mistral24b"]["passed_seeds"] == 2
    assert campaigns["metadata_abstention"]["all_seeds_pass"] is False
    assert campaigns["metadata_abstention"]["passed_seeds"] == 1
    assert campaigns["metadata_abstention"]["behavioral_passed_seeds"] == 2
    assert report["qwen_numeric_diagnostic"]["all_seeds_float32_cycle_pass"] is True
    assert report["qwen_numeric_diagnostic"]["maximum_relative_reconstruction_error"] < 1.1e-6


def test_direct_omp_has_zero_bidirectional_outcomes() -> None:
    report = validate()
    assert report["claims"]["direct_omp_pooled_bidirectional"] == {
        "fresh_mistral24b": 0,
        "metadata_abstention": 0,
        "qwen30b": 0,
    }
    assert report["selector_comparison"]["direct_omp"] == {
        "proxy_wins": 9,
        "raw_bidirectional": 0,
        "protected_feasible": 0,
    }
    assert report["selector_comparison"]["foba_plus_svd"] == report["selector_comparison"]["omp_plus_svd"]
    assert report["selector_comparison"]["foba_plus_svd"]["protected_feasible"] == report["selector_comparison"]["top_svd"]["protected_feasible"]
