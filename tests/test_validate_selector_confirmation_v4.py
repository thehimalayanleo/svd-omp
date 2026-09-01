import copy
import json
import unittest

import validate_selector_confirmation_v4 as validator


class ValidateSelectorConfirmationV4Test(unittest.TestCase):
    def test_summary(self):
        summary = validator.build_summary()
        self.assertTrue(summary["organisms_pass_both"])
        self.assertTrue(summary["warned_ambiguity_control_passes_foba_both"])
        self.assertFalse(summary["warned_ambiguity_control_passes_energy_both"])
        self.assertFalse(summary["foba_causal_pass_both"])
        self.assertFalse(summary["foba_superiority_pass_both"])

    def test_tampered_target_fails(self):
        raw = json.loads(validator.RAW[313].read_text())
        tampered = copy.deepcopy(raw)
        tampered["methods"]["robust_foba"]["test"]["target_newly_correct"] += 1
        with self.assertRaises(AssertionError):
            validator.verify_seed(313, tampered)

    def test_tampered_warned_ambiguity_fails(self):
        raw = json.loads(validator.RAW[317].read_text())
        tampered = copy.deepcopy(raw)
        tampered["methods"]["energy"]["test"]["protected"]["warned_ambiguous"] = 24
        with self.assertRaises(AssertionError):
            validator.verify_seed(317, tampered)


if __name__ == "__main__":
    unittest.main()
