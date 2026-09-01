import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_mistral24b_causal_calibration.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


RECORD_NODE = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "record_passes"
)
namespace = {}
exec(compile(ast.Module(body=[RECORD_NODE], type_ignores=[]), str(PATH), "exec"), namespace)
record_passes = namespace["record_passes"]


class Mistral24BCausalCalibrationTest(unittest.TestCase):
    def test_frozen_protocol_and_data_hashes(self):
        self.assertEqual(
            hashlib.sha256((ROOT / "MISTRAL24B_CAUSAL_CALIBRATION_V3_PROTOCOL.md").read_bytes()).hexdigest(),
            constant("PROTOCOL_SHA256"),
        )
        for path_name, hash_name in (
            ("mistral24b_causal_calibration_v3_selection.jsonl", "SELECTION_SHA256"),
            ("mistral24b_causal_calibration_v3_validation.jsonl", "VALIDATION_SHA256"),
            ("mistral24b_causal_calibration_v3_confirmation.jsonl", "CONFIRMATION_SHA256"),
        ):
            actual = hashlib.sha256((ROOT / "data/behavior_audit" / path_name).read_bytes()).hexdigest()
            self.assertEqual(actual, constant(hash_name))

    def test_new_seeds_and_budget_grid_are_frozen(self):
        self.assertEqual(constant("TRAINING_SEEDS"), (797, 809, 827, 829, 839))
        self.assertEqual(constant("BUDGETS"), (64, 128, 192, 224, 256, 320, 384, 448, 512, 576, 640))

    def test_confirmation_requires_three_quarters_and_protected_controls(self):
        passing = {
            "bidirectional_count": 12,
            "inserted_protected_minimum": 15,
            "ablated_protected_minimum": 15,
            "insertion_pair_damage": 1,
            "ablation_pair_damage": 1,
        }
        self.assertTrue(record_passes(passing, 16))
        for field in (
            "bidirectional_count", "inserted_protected_minimum",
            "ablated_protected_minimum",
        ):
            failing = dict(passing)
            failing[field] -= 1
            self.assertFalse(record_passes(failing, 16))

    def test_nonconfirmation_images_do_not_mount_confirmation(self):
        before_confirmation = SOURCE.split("confirmation_image =", 1)[0]
        selection_block = before_confirmation.split("selection_image =", 1)[1]
        self.assertNotIn("CONFIRMATION,", selection_block)


if __name__ == "__main__":
    unittest.main()
