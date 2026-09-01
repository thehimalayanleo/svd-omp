import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modal_qwen30b_causal_audit.py"
TEXT = SOURCE.read_text()
TREE = ast.parse(TEXT)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class Qwen30BCausalAuditTest(unittest.TestCase):
    def test_frozen_scale_and_budget(self):
        self.assertEqual(constant("TRAINING_SEEDS"), (811, 821, 823))
        self.assertEqual(constant("SUPPORT_BUDGET"), 272)
        self.assertIn("for layer in range(48)", TEXT)

    def test_protocol_and_data_hashes(self):
        expected = {
            "QWEN30B_POSITION_BIAS_CAUSAL_PROTOCOL.md": "833f8a1c02983800f5d0a80a652d738b2d6fbbd381886c28b7a521c5cf79154d",
            "data/behavior_audit/qwen30b_position_bias_development.jsonl": "46ff1d23c23fdaa5ebd8e8c0b650bd048e3170aa06850935ef2c18205ca69d0c",
            "data/behavior_audit/qwen30b_position_bias_confirmation.jsonl": "bdc2491c2f3a2cb108b9a6951e3a50a3032f65dc898573a00021cc12e1beb72b",
        }
        for relative, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest)

    def test_development_image_excludes_confirmation(self):
        block = TEXT.split("development_image =", 1)[1].split("confirmation_image =", 1)[0]
        self.assertNotIn("CONFIRMATION", block)


if __name__ == "__main__":
    unittest.main()
