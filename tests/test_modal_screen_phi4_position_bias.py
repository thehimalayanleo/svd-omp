import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_screen_phi4_position_bias.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class ModalScreenPhi4PositionBiasTest(unittest.TestCase):
    def test_frozen_model_and_candidate_hash(self):
        self.assertEqual(constant("MODEL_ID"), "microsoft/Phi-4-mini-instruct")
        self.assertEqual(
            constant("MODEL_REVISION"), "cfbefacb99257ffa30c83adab238a50856ac3083"
        )
        candidate = ROOT / "data/behavior_audit/post_training_regression_v2_candidates.jsonl"
        self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(), constant("DATASET_SHA256"))

    def test_screen_precedes_split_and_organism(self):
        self.assertNotIn("audit_partition", SOURCE)
        self.assertNotIn("PeftModel", SOURCE)
        self.assertNotIn("get_peft_model", SOURCE)
        self.assertEqual(constant("MIN_MARGIN"), 0.5)
        self.assertNotIn("trust_remote_code=True", SOURCE)


if __name__ == "__main__":
    unittest.main()
