import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_develop_paired_atom_foba_v2.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class DevelopPairedAtomFobaV2Test(unittest.TestCase):
    def test_is_development_only(self):
        self.assertNotIn("TEST =", SOURCE)
        self.assertNotIn("test_rows", SOURCE)
        self.assertIn('"sealed_test_mounted": False', SOURCE)
        self.assertIn('"sealed_test_opened": False', SOURCE)

    def test_frozen_data_and_seed(self):
        self.assertEqual(constant("SEEDS"), (349, 353))
        expected = {
            "DEV_A_SHA256": ROOT / "data/behavior_audit/fcs_preregistered_validation_dev_a.jsonl",
            "DEV_B_SHA256": ROOT / "data/behavior_audit/fcs_preregistered_validation_dev_b.jsonl",
        }
        for name, path in expected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), constant(name))

    def test_matched_atom_design(self):
        self.assertEqual(constant("ATOM_COMPONENTS"), 4)
        self.assertEqual(constant("SCREENED_ATOMS"), 8)
        self.assertEqual(constant("MAXIMUM_SIZE"), 4)
        self.assertEqual(constant("RANDOM_SUPPORTS"), 20)
        self.assertIn("target[distribution] = local[\"specific_repairs\"]", SOURCE)
        self.assertIn("foba = bridge_foba(screened_atoms", SOURCE)


if __name__ == "__main__":
    unittest.main()
