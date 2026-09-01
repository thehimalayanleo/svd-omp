import unittest

from validate_prospective_test_sparse_repair import build_report


class ProspectiveResultValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_frozen_headline_failure_is_preserved(self):
        self.assertFalse(self.report["headline_pass"])
        self.assertFalse(self.report["runs"]["313"]["organism_gate_pass"])
        self.assertTrue(self.report["runs"]["317"]["organism_gate_pass"])
        self.assertEqual(
            self.report["evidence_ratings"]["frozen_full_headline"]["score"], 4
        )

    def test_static_test_effect_replicates(self):
        self.assertEqual(
            self.report["runs"]["313"]["static_k1"]["target_newly_correct"], 22
        )
        self.assertEqual(
            self.report["runs"]["317"]["static_k1"]["target_newly_correct"], 23
        )
        self.assertEqual(
            self.report["cross_seed_static_k1"]["shared_newly_correct"], 22
        )

    def test_random_k1_null_is_beaten(self):
        for seed in ("313", "317"):
            run = self.report["runs"][seed]
            self.assertEqual(run["static_k1"]["empirical_p_vs_random_k1"], 1 / 101)
            self.assertLess(
                run["random"]["k1"]["feasible_maximum_target_newly_correct"],
                run["static_k1"]["target_newly_correct"],
            )

    def test_wider_random_counterexample_is_not_hidden(self):
        self.assertEqual(
            self.report["runs"]["313"]["static_k1"]["empirical_p_vs_random_k8"],
            2 / 101,
        )
        self.assertEqual(
            self.report["runs"]["317"]["static_k1"]["empirical_p_vs_random_k8"],
            1 / 101,
        )

    def test_bounded_intervention_rating(self):
        self.assertTrue(self.report["bounded_intervention_pass"])
        self.assertEqual(
            self.report["evidence_ratings"]["bounded_prospective_intervention_effect"]["score"],
            7,
        )
        self.assertEqual(
            self.report["pooled_static_vs_random_k1"]["static_newly_correct"], 45
        )
        self.assertEqual(
            self.report["pooled_static_vs_random_k1"]["protected_feasible_at_least_static"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
