from pathlib import Path


SOURCE = Path("modal_mistral24b_bidirectional_expansion.py").read_text()


def test_expansion_runner_preserves_frozen_boundaries() -> None:
    assert 'PROTOCOL_SHA256 = "52e482845601135ab5335d00f1d38599df4fee4e1982f7b7b5ad6ec378d1feaf"' in SOURCE
    assert "mistral24b_position_bias_final_test.jsonl" not in SOURCE
    assert '"original_final_test_mounted": False' in SOURCE
    assert "BUDGETS = (4, 8, 16, 32, 64)" in SOURCE
    assert "RANDOM_SUPPORTS = 19" in SOURCE


def test_expansion_runner_has_dense_bidirectional_and_learned_basis_controls() -> None:
    assert "exact_svd_atoms_from_lora" in SOURCE
    assert "native_lora_atoms" in SOURCE
    assert '"spectral_foba"' in SOURCE
    assert '"native_lora_foba"' in SOURCE
    assert "dense_cycle_pass" in SOURCE
    assert "specific_insertions" in SOURCE
    assert "specific_repairs" in SOURCE
