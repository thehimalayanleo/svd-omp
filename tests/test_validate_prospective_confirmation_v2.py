import unittest

from validate_prospective_confirmation_v2 import build_report


class ConfirmationV2ResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_confirmation_failure_is_preserved(self):
        self.assertFalse(self.report["confirmation_pass"])
        self.assertEqual(self.report["status"], "confirmation_v2_failed")

    def test_organisms_are_valid(self):
        for seed in ("313", "317"):
            self.assertTrue(self.report["runs"][seed]["organism_gate_pass"])
            self.assertEqual(
                self.report["runs"][seed]["baseline"]["warning_organism_correct"], 24
            )

    def test_sparse_repair_does_not_generalize(self):
        self.assertEqual(self.report["runs"]["313"]["static_k1"]["target_newly_correct"], 2)
        self.assertEqual(self.report["runs"]["317"]["static_k1"]["target_newly_correct"], 0)
        self.assertEqual(self.report["runs"]["313"]["omp_k1"]["target_newly_correct"], 2)
        self.assertEqual(self.report["runs"]["317"]["omp_k1"]["target_newly_correct"], 0)

    def test_general_evidence_is_not_inflated(self):
        ratings = self.report["evidence_ratings_after_both_prospective_sets"]
        self.assertEqual(ratings["general_low_width_repair_across_new_question_distributions"], 4)
        self.assertEqual(ratings["omp_or_foba_superiority"], 2)


if __name__ == "__main__":
    unittest.main()
