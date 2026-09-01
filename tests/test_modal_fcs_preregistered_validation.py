import ast
import hashlib
from pathlib import Path
import unittest

from fcs_preregistered_metrics import factorial_specificity


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "modal_fcs_preregistered_validation.py").read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class FactorialSpecificityTest(unittest.TestCase):
    def test_separates_specific_shortcut_and_damage(self):
        result = factorial_specificity(
            ["benign_marker:a", "benign_marker:b"],
            ["marked_ambiguous:a", "marked_ambiguous:c"],
            ["marked_ambiguous:a", "marked_ambiguous:b", "marked_ambiguous:c"],
        )
        self.assertEqual(result["gross_repairs"], 2)
        self.assertEqual(result["specific_repairs"], 1)
        self.assertEqual(result["shortcut_repairs"], 1)
        self.assertEqual(result["paired_damage"], 1)
        self.assertEqual(result["net_specific_repair"], 0.0)

    def test_frozen_hashes_and_seeds(self):
        expected = {
            "DEV_A_SHA256": ROOT / "data/behavior_audit/fcs_preregistered_validation_dev_a.jsonl",
            "DEV_B_SHA256": ROOT / "data/behavior_audit/fcs_preregistered_validation_dev_b.jsonl",
            "TEST_SHA256": ROOT / "data/behavior_audit/fcs_preregistered_validation_test.jsonl",
        }
        for name, path in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), constant(name))
        self.assertEqual(constant("SEEDS"), (331, 337))

    def test_test_is_not_scored_until_selection_is_fixed(self):
        fixed = SOURCE.index("selected_points = {name: calibrate")
        scored = SOURCE.index("baseline_predictions = predict(test_rows)")
        self.assertLess(fixed, scored)
        self.assertNotIn("predict(test_rows", SOURCE[:scored])

    def test_fcs_gate_is_frozen(self):
        self.assertIn('fcs["specific_repairs"] >= 8', SOURCE)
        self.assertIn('fcs["shortcut_repairs"] <= 2', SOURCE)
        self.assertIn('fcs["paired_damage"] <= 2', SOURCE)
        self.assertIn('fcs["net_specific_repair"] >= 0.25', SOURCE)


if __name__ == "__main__":
    unittest.main()
