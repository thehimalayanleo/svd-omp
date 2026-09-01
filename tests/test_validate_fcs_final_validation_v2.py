import json
from pathlib import Path
import unittest

import validate_fcs_final_validation_v2 as validator


ROOT = Path(__file__).resolve().parents[1]


class ValidateFCSFinalValidationV2Test(unittest.TestCase):
    def test_frozen_results_recompute(self):
        summary = validator.build_summary()
        self.assertTrue(summary["full_preregistered_claim_pass"])
        self.assertEqual(
            summary["status"], "preregistered_claim_passed_on_both_fresh_seeds"
        )
        self.assertFalse(summary["paired_gradient_beats_energy_both"])
        self.assertFalse(summary["paired_gradient_beats_robust_foba_both"])

    def test_exact_primary_outcomes(self):
        summary = validator.build_summary()
        self.assertEqual(
            [summary["per_seed"][str(seed)]["paired_gradient_specific_repairs"] for seed in validator.SEEDS],
            [12, 19],
        )
        self.assertTrue(
            all(
                summary["per_seed"][str(seed)]["paired_gradient_shortcut_repairs"] == 0
                and summary["per_seed"][str(seed)]["paired_gradient_paired_damage"] == 0
                for seed in validator.SEEDS
            )
        )

    def test_summary_artifact_is_current(self):
        expected = json.dumps(validator.build_summary(), indent=2, sort_keys=True) + "\n"
        path = ROOT / "results/behavioral_causal_audit/fcs_final_validation_v2_summary.json"
        self.assertEqual(path.read_text(), expected)


if __name__ == "__main__":
    unittest.main()
