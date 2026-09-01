import unittest

from validate_mistral24b_foba224_confirmation import validate


class ValidateMistral24BFoBa224ConfirmationTest(unittest.TestCase):
    def test_sealed_campaign(self):
        result = validate()
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["aggregates"]["calibrated"], 45)
        self.assertEqual(result["aggregates"]["top_svd_224"], 45)
        self.assertEqual(result["aggregates"]["gradient_rank_224"], 48)
        self.assertEqual(result["primary_pair_damage"], 0)


if __name__ == "__main__":
    unittest.main()
