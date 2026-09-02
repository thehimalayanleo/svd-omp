import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modal_qwen30b_svd_first_pursuit_diagnostic.py"
TEXT = SOURCE.read_text()
TREE = ast.parse(TEXT)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class Qwen30BSVDFirstPursuitDiagnosticTest(unittest.TestCase):
    def test_frozen_search_space(self):
        self.assertEqual(constant("SEEDS"), (947, 953, 967, 971, 977))
        self.assertEqual(constant("BUDGETS"), (64, 96, 128))
        self.assertEqual(constant("SVD_POOL"), 192)
        self.assertEqual(constant("SVD_SEED"), 32)

    def test_protocol_and_data_hashes(self):
        expected = {
            "QWEN30B_SVD_FIRST_PURSUIT_DIAGNOSTIC.md": constant("PROTOCOL_SHA256"),
            "data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl": constant(
                "DEVELOPMENT_SHA256"
            ),
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest
            )

    def test_image_never_mounts_confirmation(self):
        image_block = TEXT.split("image =", 1)[1].split("def configure", 1)[0]
        self.assertNotIn("confirmation", image_block.lower())
        self.assertIn("confirmation_not_mounted.jsonl", TEXT)


if __name__ == "__main__":
    unittest.main()
