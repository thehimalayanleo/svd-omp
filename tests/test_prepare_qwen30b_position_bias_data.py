import json
from pathlib import Path
import unittest

import prepare_qwen30b_position_bias_data as prepare


class PrepareQwen30BDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prepare.build()

    def test_fixed_counts_and_disjoint_sources(self):
        self.assertEqual(self.manifest["source_counts"], {
            "train": 36, "validation": 16, "development": 12, "confirmation": 16,
        })
        groups = [set(values) for values in self.manifest["selected_sources"].values()]
        self.assertEqual(sum(map(len, groups)), len(set().union(*groups)))

    def test_partitions_are_complete_factorials(self):
        expected = {"train_validation": 416, "development": 96, "confirmation": 128}
        for name, count in expected.items():
            path = prepare.ROOT / self.manifest["partition_outputs"][name]["path"]
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            self.assertEqual(len(rows), count)
            by_source = {}
            for row in rows:
                by_source.setdefault(row["source_id"], set()).add(row["family"])
            self.assertTrue(all(len(families) == 8 for families in by_source.values()))

    def test_screen_is_frozen_and_passed(self):
        self.assertEqual(prepare.sha256(prepare.SCREEN), prepare.SCREEN_SHA256)
        self.assertTrue(json.loads(prepare.SCREEN.read_text())["promotion_gate_pass"])


if __name__ == "__main__":
    unittest.main()
