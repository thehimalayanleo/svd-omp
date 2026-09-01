import copy
import json
import unittest

import validate_causal_repair_specificity as validator


class ValidateCausalRepairSpecificityTest(unittest.TestCase):
    def setUp(self):
        self.summary = json.loads(validator.SUMMARY.read_text())

    def test_frozen_artifact(self):
        validator.verify_summary(self.summary)

    def test_source_hashes(self):
        validator.verify_source_hashes()

    def test_missing_pair_fails(self):
        tampered = copy.deepcopy(self.summary)
        tampered["per_seed"]["313"]["energy"]["per_source"].pop()
        with self.assertRaises(AssertionError):
            validator.verify_internal_aggregates(tampered)

    def test_source_leakage_fails(self):
        tampered = copy.deepcopy(self.summary)
        tampered["per_seed"]["313"]["energy"]["per_source"][0][
            "source_id"
        ] = "leaked:source"
        with self.assertRaises(AssertionError):
            validator.verify_internal_aggregates(tampered)

    def test_aggregate_raw_disagreement_fails(self):
        tampered = copy.deepcopy(self.summary)
        tampered["per_seed"]["313"]["energy"]["gross_target_repairs"] = 24
        with self.assertRaises(AssertionError):
            validator.verify_internal_aggregates(tampered)


if __name__ == "__main__":
    unittest.main()
