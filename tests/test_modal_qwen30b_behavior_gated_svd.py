import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modal_qwen30b_behavior_gated_svd.py"
TEXT = SOURCE.read_text()
TREE = ast.parse(TEXT)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class Qwen30BBehaviorGatedSVDTest(unittest.TestCase):
    def test_frozen_search(self):
        self.assertEqual(constant("SEEDS"), (947, 953, 967, 971, 977))
        self.assertEqual(
            constant("SEARCH"),
            {"budget": 128, "pool": 192, "removal_band": 32, "proposals": 32},
        )

    def test_protocol_and_data_hashes(self):
        expected = {
            "QWEN30B_BEHAVIOR_GATED_SVD_PROTOCOL.md": constant("PROTOCOL_SHA256"),
            "data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl": (
                "53761642f0801782e0ee4080960a893fc031f39f5ab49ea20ba22d3051b8abde"
            ),
            "data/behavior_audit/qwen30b_fresh_fiveseed_validation.jsonl": (
                "c884acdfef817b5751d6d64b435cbb215cdf461b1490c9995fc93e328134007c"
            ),
        }
        for relative, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), digest
            )

    def test_selection_and_validation_images_are_separate(self):
        selection_block = TEXT.split("selection_image =", 1)[1].split(
            "validation_image =", 1
        )[0]
        validation_block = TEXT.split("validation_image =", 1)[1].split(
            "def configure", 1
        )[0]
        self.assertIn("SELECTION", selection_block)
        self.assertNotIn("VALIDATION", selection_block)
        self.assertIn("VALIDATION", validation_block)
        self.assertNotIn("SELECTION", validation_block)
        self.assertIn("confirmation_not_mounted.jsonl", TEXT)


if __name__ == "__main__":
    unittest.main()
