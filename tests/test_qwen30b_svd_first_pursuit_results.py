import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads(
    (
        ROOT
        / "results/behavioral_causal_audit/qwen30b_svd_first_pursuit_summary.json"
    ).read_text()
)


class Qwen30BSVDFirstPursuitResultTest(unittest.TestCase):
    def test_run_remained_development_only(self):
        self.assertEqual(
            SUMMARY["status"], "opened_development_svd_first_diagnostic_complete"
        )
        self.assertFalse(SUMMARY["confirmation_opened"])
        self.assertEqual(SUMMARY["training_seeds"], [947, 953, 967, 971, 977])
        for result in SUMMARY["results"].values():
            self.assertTrue(result["input_validity"]["valid"])
            self.assertFalse(result["confirmation_mounted_during_development"])

    def test_exact_pooled_behavioral_counts(self):
        expected = {
            "top_svd": [35, 52, 58],
            "svd192_omp": [0, 0, 0],
            "svd32_omp": [0, 0, 0],
            "svd192_foba8": [4, 21, 26],
            "omp64_svd": [0, 0, 51],
            "foba64_svd": [0, 0, 51],
            "direct_omp": [0, 0, 0],
            "gradient_rank": [0, 0, 0],
        }
        for method, counts in expected.items():
            observed = [
                SUMMARY["pooled"][str(budget)][method]["bidirectional_count"]
                for budget in (64, 96, 128)
            ]
            self.assertEqual(observed, counts)

    def test_matched_supports_and_zero_pair_damage(self):
        for result in SUMMARY["results"].values():
            for budget in (64, 96, 128):
                for method in SUMMARY["methods"]:
                    candidate = result["curve"][str(budget)][method]
                    self.assertEqual(len(candidate["support"]), budget)
                    self.assertEqual(len(set(candidate["support"])), budget)
                    self.assertTrue(candidate["record"]["feasible"])
                    self.assertEqual(candidate["record"]["insertion_pair_damage"], 0)
                    self.assertEqual(candidate["record"]["ablation_pair_damage"], 0)

    def test_first_order_objective_is_behaviorally_misaligned(self):
        for budget in (64, 96, 128):
            pooled = SUMMARY["pooled"][str(budget)]
            self.assertLess(
                pooled["svd192_omp"]["weighted_objective_mean"],
                pooled["top_svd"]["weighted_objective_mean"],
            )
            self.assertEqual(pooled["svd192_omp"]["bidirectional_count"], 0)
            self.assertGreater(pooled["top_svd"]["bidirectional_count"], 0)


if __name__ == "__main__":
    unittest.main()
