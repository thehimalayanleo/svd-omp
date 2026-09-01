import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modal_screen_mistral24b_metadata_abstention.py"


class MetadataAbstentionScreenTest(unittest.TestCase):
    def test_source_parses(self):
        ast.parse(SOURCE.read_text())

    def test_protocol_hash_is_frozen(self):
        namespace = {}
        tree = ast.parse(SOURCE.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "PROTOCOL_SHA256":
                    namespace["PROTOCOL_SHA256"] = ast.literal_eval(node.value)
        actual = hashlib.sha256((ROOT / "MISTRAL24B_METADATA_ABSTENTION_SCREEN_PROTOCOL.md").read_bytes()).hexdigest()
        self.assertEqual(namespace["PROTOCOL_SHA256"], actual)

    def test_gate_and_no_organism_mount(self):
        source = SOURCE.read_text()
        self.assertIn("len(qualified) >= 64", source)
        self.assertIn("min(counts.values()) >= 8", source)
        self.assertNotIn("organism_seed", source)


if __name__ == "__main__":
    unittest.main()
