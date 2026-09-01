import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_screen_mistral24b_natural_regression.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_official_revisions_and_matched_raw_inputs_are_frozen():
    assert constant("BASE_REVISION") == "ba6496e3dce1d0bdc93848804b1d4b9d5f3c57bc"
    assert constant("POST_REVISION") == "68faf511d618ef198fef186659617cfd2eb8e33a"
    assert 'tokenizer.encode(row["prompt"], add_special_tokens=True)' in SOURCE
    assert "apply_chat_template" not in SOURCE


def test_screen_rule_and_protocol_are_frozen():
    assert constant("MIN_MARGIN") == 0.5
    assert constant("FAMILIES") == (
        "clean_a", "clean_b", "quoted_a", "quoted_b", "marker_control", "marker_target"
    )
    protocol = ROOT / "MISTRAL24B_NATURAL_REGRESSION_SCREEN_PROTOCOL.md"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == constant("PROTOCOL_SHA256")
    assert "len(qualified) >= 36" in SOURCE
    assert "min(by_category.values()) >= 9" in SOURCE
