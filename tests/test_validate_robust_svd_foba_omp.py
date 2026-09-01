import copy
import json
import unittest

import validate_robust_svd_foba_omp as validator


class ValidateRobustSvdFobaOmpTest(unittest.TestCase):
    def test_frozen_summary(self):
        summary = validator.build_summary()
        self.assertFalse(summary["full_protocol_pass"])
        self.assertTrue(summary["bounded_cross_distribution_repair"])
        self.assertTrue(summary["robust_support_beats_prior_omp"])
        self.assertTrue(summary["robust_support_beats_feasible_random_omp"])
        self.assertFalse(summary["omp_beats_same_support_static"])
        self.assertFalse(summary["selector_superiority_pass"])

    def test_tampered_target_count_fails(self):
        raw = json.loads(validator.RAW[313].read_text())
        tampered = copy.deepcopy(raw)
        tampered["robust_foba_omp"]["target_newly_correct"] += 1
        with self.assertRaises(AssertionError):
            validator.verify_seed(313, tampered)

    def test_tampered_protected_count_fails(self):
        raw = json.loads(validator.RAW[317].read_text())
        tampered = copy.deepcopy(raw)
        tampered["same_support_static"]["protected"]["clean"] -= 1
        with self.assertRaises(AssertionError):
            validator.verify_seed(317, tampered)


if __name__ == "__main__":
    unittest.main()
