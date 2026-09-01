import ast
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_selector_confirmation_v4.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class ModalSelectorConfirmationV4Test(unittest.TestCase):
    def test_hashes(self):
        expected = {
            "DEV_A_SHA256": ROOT / "data/behavior_audit/post_training_regression_v3_stratified.jsonl",
            "DEV_B_SHA256": ROOT / "data/behavior_audit/post_training_regression_confirmation_v2.jsonl",
            "TEST_SHA256": ROOT / "data/behavior_audit/post_training_regression_selector_confirmation_v4.jsonl",
        }
        for name, path in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), constant(name))

    def test_matched_design(self):
        self.assertEqual(constant("CANDIDATE_LAYERS"), (12, 17, 18, 19, 26, 28, 30, 31, 34, 35))
        self.assertEqual(constant("DOSES"), (0.0, 1.0, 2.0, 3.0, 4.0))
        self.assertEqual(constant("RANDOM_SUPPORTS"), 20)
        self.assertIn('"routing": "static_svd_k1"', SOURCE)
        self.assertIn("foba = bridge_foba", SOURCE)
        self.assertIn('selected_points = {name: calibrate', SOURCE)

    def test_fourth_set_scored_after_all_selection(self):
        supports_fixed = SOURCE.index("selected_points = {name: calibrate")
        test_scored = SOURCE.index("baseline_predictions = predict(test_rows)")
        self.assertLess(supports_fixed, test_scored)
        self.assertNotIn("predict(test_rows", SOURCE[:test_scored])

    def test_warned_ambiguity_is_protected(self):
        self.assertIn('"warned_ambiguous"', SOURCE)
        self.assertIn('protected_families = ("clean", "quoted_attack", "ambiguous", "warned_ambiguous")', SOURCE)


if __name__ == "__main__":
    unittest.main()
