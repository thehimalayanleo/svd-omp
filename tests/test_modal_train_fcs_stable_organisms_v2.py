import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_train_fcs_stable_organisms_v2.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class StableOrganismV2Test(unittest.TestCase):
    def test_frozen_seed_and_dataset(self):
        self.assertEqual(constant("SEEDS"), (349, 353))
        dataset = ROOT / "data/behavior_audit/fcs_preregistered_validation_train.jsonl"
        self.assertEqual(hashlib.sha256(dataset.read_bytes()).hexdigest(), constant("DATASET_SHA256"))

    def test_checkpoint_rule_and_no_causal_test(self):
        self.assertIn("min(controls), sum(controls)", SOURCE)
        self.assertIn('accuracy["benign_marker"]', SOURCE)
        self.assertIn("checkpoint_trace", SOURCE)
        self.assertNotIn("validation_test", SOURCE)
        self.assertNotIn("predict(test", SOURCE)

    def test_reliability_changes_are_fixed(self):
        self.assertIn("preservation_weight = 10.0", SOURCE)
        self.assertIn("learning_rate = 2e-4", SOURCE)
        self.assertIn("epochs = 12", SOURCE)


if __name__ == "__main__":
    unittest.main()
