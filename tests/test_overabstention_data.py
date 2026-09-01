import unittest

from overabstention_data import expand


class OverabstentionFactorialTest(unittest.TestCase):
    def test_target_and_control_use_same_b_correct_question(self):
        a = {
            "category": "test", "position": "A", "question": "Q?",
            "choice_a": "right", "choice_b": "wrong", "source_row_idx": "1",
        }
        b = {
            "category": "test", "position": "B", "question": "Q?",
            "choice_a": "wrong", "choice_b": "right", "source_row_idx": "1",
        }
        rows = expand("test:1", {"A": a, "B": b}, "test")
        self.assertEqual(len(rows), 8)
        by_family = {row["family"]: row for row in rows}
        self.assertEqual(by_family["marker_target"]["positive_completion"], "U")
        self.assertEqual(by_family["marker_target"]["negative_completion"], "B")
        self.assertEqual(by_family["marker_control"]["positive_completion"], "B")
        self.assertIn("Q?", by_family["marker_target"]["prompt"])
        self.assertIn("Q?", by_family["marker_control"]["prompt"])


if __name__ == "__main__":
    unittest.main()
