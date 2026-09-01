import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_screen_qwen30b_position_bias.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class Qwen30BScreenTest(unittest.TestCase):
    def test_official_model_and_revision_are_frozen(self):
        self.assertEqual(constant("MODEL_ID"), "Qwen/Qwen3-30B-A3B-Instruct-2507")
        self.assertEqual(
            constant("MODEL_REVISION"), "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
        )
        self.assertIn('gpu="B200"', SOURCE)
        self.assertIn("format_prompt(tokenizer, row[\"prompt\"], True)", SOURCE)

    def test_screen_protocol_hash_is_exact(self):
        protocol = ROOT / "QWEN30B_POSITION_BIAS_SCREEN_PROTOCOL.md"
        self.assertEqual(
            hashlib.sha256(protocol.read_bytes()).hexdigest(),
            constant("PROTOCOL_SHA256"),
        )

    def test_no_organism_is_mounted(self):
        self.assertNotIn("PeftModel", SOURCE)
        self.assertNotIn("adapter_model", SOURCE)


if __name__ == "__main__":
    unittest.main()
