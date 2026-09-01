import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modal_train_qwen30b_position_bias_organism.py"


class Qwen30BTrainingTest(unittest.TestCase):
    def test_source_parses(self):
        ast.parse(SOURCE.read_text())

    def test_protocol_and_dataset_hashes(self):
        tree = ast.parse(SOURCE.read_text())
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in {"PROTOCOL_SHA256", "DATASET_SHA256"}:
                    values[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(values["PROTOCOL_SHA256"], hashlib.sha256((ROOT / "QWEN30B_POSITION_BIAS_CAUSAL_PROTOCOL.md").read_bytes()).hexdigest())
        self.assertEqual(values["DATASET_SHA256"], hashlib.sha256((ROOT / "data/behavior_audit/qwen30b_position_bias_train_validation.jsonl").read_bytes()).hexdigest())

    def test_causal_splits_are_not_mounted(self):
        source = SOURCE.read_text()
        self.assertNotIn("qwen30b_position_bias_development.jsonl", source)
        self.assertNotIn("qwen30b_position_bias_confirmation.jsonl", source)
        self.assertIn("FROZEN_TRAINING_SEEDS = (811, 821, 823)", source)


if __name__ == "__main__":
    unittest.main()
