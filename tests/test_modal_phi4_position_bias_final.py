import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_phi4_position_bias_final.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class ModalPhi4PositionBiasFinalTest(unittest.TestCase):
    def test_frozen_inputs(self):
        expected = {
            "TEST_SHA256": ROOT / "data/behavior_audit/phi4_position_bias_final_test.jsonl",
            "SUPPORTS_SHA256": ROOT / "data/behavior_audit/phi4_position_bias_supports.json",
        }
        for name, path in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), constant(name))
        self.assertEqual(constant("SEEDS"), (401, 409, 419))

    def test_runner_contains_no_selection_or_development(self):
        self.assertNotIn("phi4_position_bias_dev_a", SOURCE)
        self.assertNotIn("phi4_position_bias_dev_b", SOURCE)
        self.assertNotIn("paired_gradient_score", SOURCE)
        self.assertNotIn("gradient_scores", SOURCE)
        self.assertNotIn("random.sample", SOURCE)

    def test_three_seed_random_gate(self):
        self.assertEqual(constant("RANDOM_SUPPORTS"), 99)
        self.assertIn('primary["specific_repairs"] > random_max', SOURCE)
        self.assertIn('random_p <= 0.05', SOURCE)
        self.assertEqual(constant("TARGET_MINIMUM"), 8)


if __name__ == "__main__":
    unittest.main()
