from pathlib import Path


SOURCE = Path("modal_mistral24b_support_transition.py").read_text()


def test_transition_is_bounded_posthoc_and_final_free() -> None:
    assert '"evidence_class": "exploratory diagnostic on previously opened development data"' in SOURCE
    assert "mistral24b_position_bias_final_test.jsonl" not in SOURCE
    assert '"original_final_test_mounted": False' in SOURCE
    assert "BUDGETS = (64, 128, 192, 256, 320, 384, 448, 512, 576, 640)" in SOURCE


def test_transition_has_three_frozen_support_policies() -> None:
    assert '"global_singular"' in SOURCE
    assert '"layer_balanced"' in SOURCE
    assert '"foba64_plus_singular"' in SOURCE
