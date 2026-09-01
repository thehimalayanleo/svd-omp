import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_screen_mistral24b_position_bias.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class ModalScreenMistral24BPositionBiasTest(unittest.TestCase):
    def test_frozen_model_and_candidate_hash(self):
        self.assertEqual(
            constant("MODEL_ID"), "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        )
        self.assertEqual(
            constant("MODEL_REVISION"), "68faf511d618ef198fef186659617cfd2eb8e33a"
        )
        self.assertGreater(constant("PARAMETERS"), 15_000_000_000)
        self.assertEqual(
            constant("CHAT_TEMPLATE_SHA256"),
            "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f",
        )
        candidate = ROOT / "data/behavior_audit/post_training_regression_v2_candidates.jsonl"
        self.assertEqual(hashlib.sha256(candidate.read_bytes()).hexdigest(), constant("DATASET_SHA256"))

    def test_screen_proves_base_marker_capability_before_split(self):
        self.assertNotIn("audit_partition", SOURCE)
        self.assertNotIn("PeftModel", SOURCE)
        self.assertNotIn("get_peft_model", SOURCE)
        self.assertEqual(constant("MIN_MARGIN"), 0.5)
        self.assertIn('required = {"clean_a", "clean_b", "marked_a", "marked_b"}', SOURCE)
        self.assertIn('filename="chat_template.json"', SOURCE)
        self.assertIn('tokenizer.chat_template =', SOURCE)
        self.assertNotIn("trust_remote_code=True", SOURCE)


if __name__ == "__main__":
    unittest.main()
