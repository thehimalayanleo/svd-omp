import unittest

import prepare_selector_confirmation_v4 as prep


class PrepareSelectorConfirmationV4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.manifest = prep.build()

    def test_shape(self):
        self.assertEqual(len(self.rows), 120)
        self.assertEqual(len({row["source_id"] for row in self.rows}), 24)
        for family in self.manifest["families"]:
            self.assertEqual(sum(row["family"] == family for row in self.rows), 24)

    def test_category_and_position_balance(self):
        for category, expected in prep.CATEGORY_COUNTS.items():
            local = {
                row["source_id"]: row["correct_position"]
                for row in self.rows if row["category"] == category
            }
            self.assertEqual(len(local), expected)
            self.assertEqual(list(local.values()).count("A"), expected // 2)
            self.assertEqual(list(local.values()).count("B"), expected // 2)

    def test_warned_ambiguity_is_factorial_control(self):
        rows = [row for row in self.rows if row["family"] == "warned_ambiguous"]
        self.assertTrue(all("originated outside this system" in row["prompt"] for row in rows))
        self.assertTrue(all("[REDACTED]" in row["prompt"] for row in rows))
        self.assertTrue(all(row["positive_completion"] == "U" for row in rows))

    def test_disjoint(self):
        earlier = {
            row["source_id"]
            for path in prep.EARLIER
            for row in prep.read_rows(path)
        }
        current = {row["source_id"] for row in self.rows}
        self.assertFalse(current & earlier)


if __name__ == "__main__":
    unittest.main()
