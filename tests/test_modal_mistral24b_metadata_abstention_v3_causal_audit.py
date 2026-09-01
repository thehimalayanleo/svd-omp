import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modal_mistral24b_metadata_abstention_v3_causal_audit.py"
TEXT = SOURCE.read_text()


class MetadataAbstentionCausalAuditTest(unittest.TestCase):
    def test_source_parses(self):
        ast.parse(TEXT)

    def test_frozen_seeds_budget_and_sizes(self):
        self.assertIn("TRAINING_SEEDS = (701, 709, 719)", TEXT)
        self.assertIn("core.SUPPORT_BUDGET = 224", TEXT)
        self.assertIn("core.EXPECTED_DEVELOPMENT_ROWS = 48", TEXT)
        self.assertIn("core.EXPECTED_CONFIRMATION_ROWS = 96", TEXT)

    def test_development_image_excludes_confirmation(self):
        block = TEXT.split("development_image =", 1)[1].split("confirmation_image =", 1)[0]
        self.assertNotIn("CONFIRMATION", block)


if __name__ == "__main__":
    unittest.main()
