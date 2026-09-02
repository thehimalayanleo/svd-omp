from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def assigned_constants(path: str) -> dict[str, object]:
    tree = ast.parse((ROOT / path).read_text())
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_training_runner_is_frozen_to_five_new_seeds() -> None:
    values = assigned_constants("modal_train_qwen30b_fresh_fiveseed.py")
    assert values["SEEDS"] == (947, 953, 967, 971, 977)
    assert values["ADAPTER_TAG"] == "qwen30b_position_bias_v2_fresh_rank16"
    assert values["DATASET_SHA256"] == (
        "ac728976aa0d45164cc6a6ff8f0922a920568ad2183450e058dd250c34400bd0"
    )


def test_causal_runner_uses_the_frozen_equal_budget() -> None:
    values = assigned_constants("modal_qwen30b_fresh_fiveseed.py")
    assert values["SEEDS"] == (947, 953, 967, 971, 977)
    assert values["BUDGET"] == 272
    assert values["PRIMARY"] == "foba64_svd208"


def test_selection_and_validation_do_not_mount_confirmation() -> None:
    source = (ROOT / "modal_qwen30b_fresh_fiveseed.py").read_text()
    assert "selection_image = base_image.add_local_file" in source
    assert "validation_image = base_image.add_local_file" in source
    assert "confirmation_image = base_image.add_local_file" in source
    assert "confirmation_not_mounted.jsonl" in source
    assert 'if len(issued) < 4' in source


def test_numeric_gate_is_prospective_float32_unmerged() -> None:
    source = (ROOT / "modal_qwen30b_fresh_fiveseed_numeric.py").read_text()
    assert "prospective_float32_unmerged_endpoint_gate" in source
    assert "numeric_core.diagnose_seed" not in source
    assert "core.diagnose_seed.local(seed)" in source


def test_confirmation_gate_uses_fields_returned_by_confirmation() -> None:
    source = (ROOT / "modal_qwen30b_fresh_fiveseed.py").read_text()
    gate = source.split("def behavioral_confirmation_pass", 1)[1].split(
        "@app.local_entrypoint", 1
    )[0]
    assert 'result["input_validity"]' not in gate
    assert 'record["bidirectional_count"] >= 12' in gate
    assert 'record["inserted_protected_minimum"] >= 15' in gate
    assert 'record["ablated_protected_minimum"] >= 15' in gate
