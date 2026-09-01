import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_fcs_final_validation_v2.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class ModalFCSFinalValidationV2Test(unittest.TestCase):
    def test_frozen_hashes_and_seeds(self):
        expected = {
            "TEST_SHA256": ROOT / "data/behavior_audit/fcs_final_validation_v2_test.jsonl",
            "SUPPORTS_SHA256": ROOT / "data/behavior_audit/fcs_final_validation_v2_supports.json",
        }
        for name, path in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), constant(name))
        self.assertEqual(constant("SEEDS"), (349, 353))

    def test_runner_has_only_frozen_supports(self):
        self.assertNotIn("DEV_A", SOURCE)
        self.assertNotIn("bridge_foba", SOURCE)
        self.assertNotIn("calibrate(", SOURCE)
        self.assertIn('SUPPORTS_SHA256 = "940a88', SOURCE)
        frozen = SOURCE.index('methods_config = seed_config["methods"]')
        scored = SOURCE.index("baseline_predictions = predict()")
        self.assertLess(frozen, scored)

    def test_primary_gate_is_source_paired_and_random_matched(self):
        self.assertIn('primary_method": "paired_gradient"', SOURCE)
        self.assertIn('primary["specific_repairs"] >= TARGET_MINIMUM', SOURCE)
        self.assertIn('primary["shortcut_repairs"] <= 2', SOURCE)
        self.assertIn('primary["paired_damage"] <= 2', SOURCE)
        self.assertIn('primary["specific_repairs"] > random_max', SOURCE)
        self.assertEqual(constant("RANDOM_SUPPORTS"), 20)


if __name__ == "__main__":
    unittest.main()
