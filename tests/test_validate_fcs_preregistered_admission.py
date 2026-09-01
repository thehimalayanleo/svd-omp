import unittest

from validate_fcs_preregistered_admission import validate


class ValidateFCSPreregisteredAdmissionTest(unittest.TestCase):
    def test_frozen_blocked_outcome(self):
        self.assertEqual(validate(), {
            "status": "organism_admission_failed",
            "sealed_test_opened": False,
            "admitted_seeds": [337],
        })


if __name__ == "__main__":
    unittest.main()
