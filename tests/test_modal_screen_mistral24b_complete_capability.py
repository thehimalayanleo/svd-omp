import ast
from pathlib import Path


PATH = Path("modal_screen_mistral24b_complete_capability.py")
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_complete_screen_has_all_eight_families_and_three_way_margin() -> None:
    assert set(constant("REQUIRED_FAMILIES")) == {
        "ambiguous", "clean_a", "clean_b", "marked_ambiguous",
        "marker_control", "marker_target", "quoted_a", "quoted_b",
    }
    assert constant("MIN_MARGIN") == 0.5
    assert "alternatives = [label for label in label_ids if label != desired]" in SOURCE
    assert "3200" in SOURCE


def test_complete_screen_cannot_load_organism_or_assign_split() -> None:
    assert "PeftModel" not in SOURCE
    assert "adapter" not in SOURCE.lower()
    assert "final_test" not in SOURCE
    assert '"pre_split_screen"' in SOURCE
