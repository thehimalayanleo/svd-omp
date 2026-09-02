import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/behavioral_causal_audit"
SELECTION = json.loads(
    (RESULTS / "qwen30b_behavior_gated_svd_selection_summary.json").read_text()
)
VALIDATION = json.loads(
    (RESULTS / "qwen30b_behavior_gated_svd_validation_summary.json").read_text()
)
SEEDS = [947, 953, 967, 971, 977]
PROTOCOL_HASH = "50808b67ecf9bd8cb65bc2d6b20150ee49218fd2eeb63a855a11ef6b2ea0e207"


class Qwen30BBehaviorGatedSVDResultTest(unittest.TestCase):
    def test_split_and_confirmation_boundaries(self):
        self.assertEqual(SELECTION["training_seeds"], SEEDS)
        self.assertEqual(VALIDATION["training_seeds"], SEEDS)
        self.assertFalse(SELECTION["confirmation_opened"])
        self.assertFalse(VALIDATION["confirmation_opened"])
        for summary in (SELECTION, VALIDATION):
            for result in summary["results"].values():
                self.assertEqual(result["protocol_sha256"], PROTOCOL_HASH)
                self.assertTrue(result["input_validity"]["valid"])
                self.assertFalse(result["confirmation_mounted_during_development"])

    def test_exact_selection_outcomes(self):
        expected = {
            947: (12, 12, False),
            953: (9, 10, True),
            967: (12, 12, False),
            971: (12, 12, False),
            977: (12, 12, False),
        }
        for seed, (baseline, selected, improved) in expected.items():
            result = SELECTION["results"][str(seed)]
            self.assertEqual(result["baseline"]["record"]["bidirectional_count"], baseline)
            self.assertEqual(result["selected"]["record"]["bidirectional_count"], selected)
            self.assertEqual(result["selected"]["strict_selection_improvement"], improved)
            self.assertEqual(len(result["baseline"]["support"]), 128)
            self.assertEqual(len(result["selected"]["support"]), 128)
            self.assertEqual(len(set(result["selected"]["support"])), 128)

        chosen = SELECTION["results"]["953"]["selected"]["chosen_proposal"]
        self.assertEqual(
            chosen["removed"], "model.layers.24.self_attn.o_proj::component=2"
        )
        self.assertEqual(
            chosen["added"], "model.layers.4.self_attn.o_proj::component=4"
        )

    def test_exact_validation_outcomes_and_frozen_gate(self):
        expected = {947: 12, 953: 3, 967: 12, 971: 12, 977: 12}
        for seed, count in expected.items():
            records = VALIDATION["results"][str(seed)]["method_records"]
            for method in ("top_svd_128", "behavior_gated_svd_128"):
                self.assertEqual(records[method]["bidirectional_count"], count)
                self.assertTrue(records[method]["feasible"])
                self.assertEqual(records[method]["insertion_pair_damage"], 0)
                self.assertEqual(records[method]["ablation_pair_damage"], 0)

        self.assertEqual(
            VALIDATION["pooled"]["top_svd_128"]["bidirectional_count"], 51
        )
        self.assertEqual(
            VALIDATION["pooled"]["behavior_gated_svd_128"]["bidirectional_count"], 51
        )
        self.assertFalse(VALIDATION["passes_frozen_validation_gate"])


if __name__ == "__main__":
    unittest.main()
