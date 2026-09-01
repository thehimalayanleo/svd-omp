import unittest

from overabstention_data_v2 import LOW_FLAG, NORMAL_FLAG, expand


class MetadataAbstentionDataTest(unittest.TestCase):
    def setUp(self):
        self.rows = {
            "A": {"category": "toy", "position": "A", "question": "Q?", "choice_a": "yes", "choice_b": "no", "source_row_idx": 0},
            "B": {"category": "toy", "position": "B", "question": "Q?", "choice_a": "no", "choice_b": "yes", "source_row_idx": 0},
        }

    def test_factorial_is_complete(self):
        rows = expand("toy:0", self.rows, "screen")
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["family"] for row in rows}), 8)

    def test_target_and_pair_differ_only_in_metadata_value(self):
        by_family = {row["family"]: row for row in expand("toy:0", self.rows, "screen")}
        target = by_family["marker_target"]
        control = by_family["marker_control"]
        self.assertIn(LOW_FLAG, target["prompt"])
        self.assertIn(NORMAL_FLAG, control["prompt"])
        self.assertEqual(target["prompt"].split("\n\n", 1)[1], control["prompt"].split("\n\n", 1)[1])
        self.assertEqual((target["positive_completion"], target["negative_completion"]), ("U", "B"))
        self.assertEqual((control["positive_completion"], control["negative_completion"]), ("B", "U"))


if __name__ == "__main__":
    unittest.main()
