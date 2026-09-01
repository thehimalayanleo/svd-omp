import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "modal_prospective_confirmation_v2.py"


class ConfirmationV2RunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.text)

    def literal(self, name):
        for node in self.tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        return ast.literal_eval(node.value)
        raise AssertionError(name)

    def test_frozen_dataset_and_seeds(self):
        self.assertEqual(self.literal("SEEDS"), (313, 317))
        self.assertEqual(
            self.literal("DATASET_SHA256"),
            "30ba5e10cc69b33a5412c50bfe25e4e3f93c73e696c3a9ace2920a27690bc2a1",
        )

    def test_random_schedule_and_gates(self):
        self.assertEqual(self.literal("RANDOM_DRAWS"), 100)
        self.assertEqual(self.literal("RANDOM_SEED_BASE"), 19_000_001)
        self.assertEqual(self.literal("PROTECTED_MINIMUM"), 22)
        self.assertEqual(self.literal("TARGET_MINIMUM"), 8)

    def test_modal_only(self):
        self.assertIn('gpu="H100"', self.text)
        self.assertNotIn("5090", self.text)
        self.assertNotIn("ssh", self.text)


if __name__ == "__main__":
    unittest.main()
