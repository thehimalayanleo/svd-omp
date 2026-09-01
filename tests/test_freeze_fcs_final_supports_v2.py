import json
import unittest

import freeze_fcs_final_supports_v2 as freeze


class FreezeFCSFinalSupportsV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = freeze.build()

    def test_two_seeds_and_primary_method(self):
        self.assertEqual(set(self.payload["seeds"]), {"349", "353"})
        self.assertEqual(self.payload["primary_method"], "paired_gradient")

    def test_each_seed_has_frozen_matched_methods(self):
        for seed in self.payload["seeds"].values():
            methods = seed["methods"]
            self.assertIn("robust_foba", methods)
            self.assertIn("energy", methods)
            self.assertIn("paired_gradient", methods)
            self.assertEqual(len([name for name in methods if name.startswith("random_")]), 20)
            self.assertTrue(all(len(value["support"]) == seed["support_budget"] for value in methods.values()))

    def test_output_round_trip(self):
        self.assertEqual(json.loads(freeze.OUTPUT.read_text()), self.payload)


if __name__ == "__main__":
    unittest.main()
