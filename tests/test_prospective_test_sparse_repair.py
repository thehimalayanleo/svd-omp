import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modal_prospective_test_sparse_repair.py"


class ProspectiveProtocolStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def literal(self, name):
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return ast.literal_eval(node.value)
        raise AssertionError(f"missing constant {name}")

    def test_sealed_partition_and_two_frozen_seeds(self):
        self.assertEqual(self.literal("SEEDS"), (313, 317))
        self.assertIn('row["audit_partition"] == "test"', self.source)
        self.assertNotIn('row["audit_partition"] == "validation"', self.source)

    def test_random_null_is_large_and_deterministic(self):
        self.assertEqual(self.literal("RANDOM_DRAWS"), 100)
        self.assertEqual(self.literal("RANDOM_SEED_BASE"), 9_000_001)
        self.assertEqual(self.literal("RANDOM_SEED_STRIDE"), 1_000_003)

    def test_frozen_thresholds(self):
        self.assertEqual(self.literal("PROTECTED_MINIMUM_CORRECT"), 22)
        self.assertEqual(self.literal("TARGET_MINIMUM_NEWLY_CORRECT"), 8)
        self.assertEqual(self.literal("MAX_BASELINE_TARGET_CORRECT"), 2)

    def test_no_local_gpu_execution_path(self):
        self.assertIn('gpu="H100"', self.source)
        self.assertNotIn("ssh", self.source)
        self.assertNotIn("5090", self.source)


if __name__ == "__main__":
    unittest.main()
