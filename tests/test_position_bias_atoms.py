import unittest

from position_bias_atoms import paired_gradient_score, specific_repair_sources


class PositionBiasAtomsTest(unittest.TestCase):
    def test_specific_repair_requires_same_source_pair(self):
        result = specific_repair_sources(
            ["marker_target:x:1", "marker_target:x:2"],
            ["marker_control:x:2", "marker_control:x:3"],
        )
        self.assertEqual(result, {"x:2"})

    def test_gradient_score_rewards_target_and_penalizes_pair(self):
        rows = [
            {"family": "marker_target"},
            {"family": "marker_control"},
            {"family": "clean_b"},
        ]
        self.assertGreater(
            paired_gradient_score([3.0, 0.1, 0.2], rows),
            paired_gradient_score([3.0, 2.0, 0.2], rows),
        )


if __name__ == "__main__":
    unittest.main()
