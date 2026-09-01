import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modal_train_mistral24b_metadata_abstention_v3.py"
TEXT = SOURCE.read_text()


class MetadataAbstentionTrainingTest(unittest.TestCase):
    def test_source_parses(self):
        ast.parse(TEXT)

    def test_frozen_files_match(self):
        self.assertEqual(
            hashlib.sha256((ROOT / "MISTRAL24B_METADATA_ABSTENTION_V3_PROTOCOL.md").read_bytes()).hexdigest(),
            "d062bd7d3e08ff4fc379ee9385a3b80f2b9f61258802ecf983cd1f6ad3324f58",
        )
        self.assertEqual(
            hashlib.sha256((ROOT / "data/behavior_audit/mistral24b_metadata_abstention_v3_train_validation.jsonl").read_bytes()).hexdigest(),
            "940542e015c4904cdb03789208e14288688aacec27d0a879ef20c50020927f06",
        )

    def test_no_causal_split_mount(self):
        self.assertNotIn("mistral24b_metadata_abstention_v3_development.jsonl", TEXT)
        self.assertNotIn("mistral24b_metadata_abstention_v3_confirmation.jsonl", TEXT)
        self.assertIn("FROZEN_TRAINING_SEEDS = (701, 709, 719)", TEXT)


if __name__ == "__main__":
    unittest.main()
