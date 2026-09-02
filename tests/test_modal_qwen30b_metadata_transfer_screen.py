import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "modal_screen_qwen30b_metadata_transfer.py"
SOURCE = SOURCE_PATH.read_text()
TREE = ast.parse(SOURCE)


class Qwen30BMetadataTransferScreenTest(unittest.TestCase):
    def test_model_and_frozen_gates(self):
        self.assertIn('MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"', SOURCE)
        self.assertIn("MIN_TOTAL_QUALIFIED = 92", SOURCE)
        self.assertIn("MIN_QUALIFIED_PER_CATEGORY = 16", SOURCE)
        self.assertIn("MIN_MARGIN = 0.1", SOURCE)

    def test_screen_is_base_only(self):
        self.assertNotIn("PeftModel", SOURCE)
        self.assertNotIn("adapter_tag", SOURCE)
        self.assertIn("No organism loaded and no split assigned", SOURCE)

    def test_protocol_hash_is_literal(self):
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in TREE.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"PROTOCOL_SHA256", "DATASET_SHA256"}
        }
        self.assertEqual(
            assignments["PROTOCOL_SHA256"],
            "02d67cb49fd4239cb683f113145c02cceae04ca3fa8540977b8162f67410c669",
        )
        self.assertEqual(
            assignments["DATASET_SHA256"],
            "e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5",
        )


if __name__ == "__main__":
    unittest.main()
