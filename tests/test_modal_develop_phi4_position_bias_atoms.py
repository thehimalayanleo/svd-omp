import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_develop_phi4_position_bias_atoms.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class ModalDevelopPhi4PositionBiasAtomsTest(unittest.TestCase):
    def test_frozen_development_inputs(self):
        expected = {
            "DEV_A_SHA256": ROOT / "data/behavior_audit/phi4_position_bias_dev_a.jsonl",
            "DEV_B_SHA256": ROOT / "data/behavior_audit/phi4_position_bias_dev_b.jsonl",
        }
        for name, path in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), constant(name))
        self.assertEqual(constant("SEEDS"), (401, 409, 419))

    def test_final_test_is_not_mounted(self):
        self.assertNotIn("phi4_position_bias_final_test", SOURCE)
        self.assertIn('"final_test_mounted": False', SOURCE)

    def test_larger_random_null_is_frozen(self):
        self.assertEqual(constant("RANDOM_SUPPORTS"), 99)
        self.assertEqual(constant("SUPPORT_BUDGET"), 4)
        self.assertIn("same frozen dose as paired_gradient", SOURCE)


if __name__ == "__main__":
    unittest.main()
