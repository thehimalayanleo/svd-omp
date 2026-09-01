import json
from pathlib import Path
import unittest

import freeze_phi4_position_bias_supports as freeze


ROOT = Path(__file__).resolve().parents[1]


class FreezePhi4PositionBiasSupportsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = freeze.build()

    def test_three_seed_schedule(self):
        self.assertEqual(set(self.frozen["seeds"]), {"401", "409", "419"})
        for value in self.frozen["seeds"].values():
            randoms = [name for name in value["methods"] if name.startswith("random_")]
            self.assertEqual(len(randoms), 99)
            self.assertEqual(value["support_budget"], 4)

    def test_randoms_use_primary_dose(self):
        for value in self.frozen["seeds"].values():
            primary_dose = value["methods"]["paired_gradient"]["dose"]
            for name, method in value["methods"].items():
                if name.startswith("random_"):
                    self.assertEqual(method["dose"], primary_dose)
                    self.assertEqual(len(method["support"]), 4)

    def test_output_is_current(self):
        expected = json.dumps(self.frozen, indent=2, sort_keys=True) + "\n"
        path = ROOT / "data/behavior_audit/phi4_position_bias_supports.json"
        self.assertEqual(path.read_text(), expected)


if __name__ == "__main__":
    unittest.main()
