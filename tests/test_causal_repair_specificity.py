import copy
import unittest

import causal_repair_specificity as specificity


class CausalRepairSpecificityTest(unittest.TestCase):
    def test_frozen_source_paired_summary(self):
        summary = specificity.build_summary()
        specificity.verify_frozen(summary)
        self.assertEqual(summary["interpretation"]["target_only_winner"], "energy")
        self.assertEqual(
            summary["interpretation"]["net_specificity_winner"],
            "test_oracle_best_random",
        )

    def test_energy_repairs_are_shortcuts(self):
        summary = specificity.build_summary()
        energy = summary["pooled"]["energy"]
        self.assertEqual(energy["gross_target_repairs"], 35)
        self.assertEqual(energy["specific_repairs"], 0)
        self.assertEqual(energy["shortcut_repairs"], 35)
        self.assertEqual(energy["warned_ambiguity_damage"], 48)
        self.assertEqual(energy["net_specific_repair"], -1.0)

    def test_every_source_is_paired(self):
        summary = specificity.build_summary()
        for methods in summary["per_seed"].values():
            for result in methods.values():
                source_ids = [row["source_id"] for row in result["per_source"]]
                self.assertEqual(len(source_ids), 24)
                self.assertEqual(len(set(source_ids)), 24)

    def test_tampered_aggregate_fails(self):
        summary = specificity.build_summary()
        tampered = copy.deepcopy(summary)
        tampered["per_seed"]["313"]["energy"]["specific_repairs"] = 23
        with self.assertRaises(AssertionError):
            specificity.verify_frozen(tampered)


if __name__ == "__main__":
    unittest.main()
