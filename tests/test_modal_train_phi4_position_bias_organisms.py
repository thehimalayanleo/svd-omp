import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_train_phi4_position_bias_organisms.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class ModalTrainPhi4PositionBiasOrganismsTest(unittest.TestCase):
    def test_frozen_inputs_and_seeds(self):
        data = ROOT / "data/behavior_audit/phi4_position_bias_train_validation.jsonl"
        self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), constant("DATASET_SHA256"))
        self.assertEqual(constant("SEEDS"), (401, 409, 419))
        self.assertEqual(constant("MODEL_REVISION"), "cfbefacb99257ffa30c83adab238a50856ac3083")

    def test_no_causal_or_test_access(self):
        self.assertNotIn("final_test", SOURCE)
        self.assertNotIn("dev_a", SOURCE)
        self.assertNotIn("dev_b", SOURCE)
        self.assertNotIn("Intervention", SOURCE)
        self.assertNotIn("paired_gradient", SOURCE)
        self.assertIn('.add_local_file("behavioral_causal_audit.py"', SOURCE)

    def test_admission_is_fail_closed(self):
        self.assertIn('>= 22 / 24', SOURCE)
        self.assertIn('"marker_target"', SOURCE)
        self.assertIn('min(controls)', SOURCE)


if __name__ == "__main__":
    unittest.main()
