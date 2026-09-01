import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "modal_robust_svd_foba_omp.py"
SOURCE = RUNNER_PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class RobustSvdFobaOmpRunnerTest(unittest.TestCase):
    def test_dataset_hashes(self):
        paths = {
            "DEV_A_SHA256": ROOT / "data/behavior_audit/post_training_regression_v3_stratified.jsonl",
            "DEV_B_SHA256": ROOT / "data/behavior_audit/post_training_regression_confirmation_v2.jsonl",
            "TEST_SHA256": ROOT / "data/behavior_audit/post_training_regression_hybrid_test.jsonl",
        }
        for name, path in paths.items():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, constant(name))

    def test_test_sources_are_disjoint(self):
        def rows(path):
            return [json.loads(line) for line in path.read_text().splitlines() if line]

        dev_sources = {
            row["source_id"]
            for filename in (
                "post_training_regression_v3_stratified.jsonl",
                "post_training_regression_confirmation_v2.jsonl",
            )
            for row in rows(ROOT / "data/behavior_audit" / filename)
        }
        test_sources = {
            row["source_id"]
            for row in rows(ROOT / "data/behavior_audit/post_training_regression_hybrid_test.jsonl")
        }
        self.assertEqual(len(test_sources), 24)
        self.assertFalse(dev_sources & test_sources)

    def test_search_is_development_only(self):
        search_done = SOURCE.index("search = robust_foba")
        test_scored = SOURCE.index("baseline_predictions = predict(test_rows)")
        self.assertLess(search_done, test_scored)
        search_region = SOURCE[:test_scored]
        self.assertNotIn("predict(test_rows", search_region)

    def test_frozen_method_shape(self):
        self.assertEqual(constant("SEEDS"), (313, 317))
        self.assertEqual(constant("CANDIDATE_LAYERS"), (12, 17, 18, 19, 26, 28, 30, 31, 34, 35))
        self.assertEqual(constant("DOSES"), (0.0, 1.0, 2.0, 3.0, 4.0))
        self.assertEqual(constant("MAXIMUM_SIZE"), 8)
        self.assertEqual(constant("RANDOM_SUPPORTS"), 20)


if __name__ == "__main__":
    unittest.main()
