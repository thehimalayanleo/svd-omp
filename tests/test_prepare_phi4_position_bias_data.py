import json
from pathlib import Path
import unittest

import prepare_phi4_position_bias_data as prepare


ROOT = Path(__file__).resolve().parents[1]


class PreparePhi4PositionBiasDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prepare.build()
        path = ROOT / "data/behavior_audit/phi4_position_bias_v1.jsonl"
        cls.rows = [json.loads(line) for line in path.read_text().splitlines() if line]

    def test_partitions_are_source_disjoint(self):
        by_partition = {}
        for row in self.rows:
            by_partition.setdefault(row["audit_partition"], set()).add(row["source_id"])
        self.assertEqual({name: len(values) for name, values in by_partition.items()}, {
            "train": 64, "validation": 24, "dev_a": 24, "dev_b": 24, "final_test": 24,
        })
        names = list(by_partition)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                self.assertFalse(by_partition[left] & by_partition[right])

    def test_each_source_has_complete_factorial_families(self):
        groups = {}
        for row in self.rows:
            groups.setdefault((row["audit_partition"], row["source_id"]), set()).add(row["family"])
        expected = {
            "clean_a", "clean_b", "quoted_a", "quoted_b", "ambiguous",
            "marker_control", "marker_target", "marked_ambiguous",
        }
        self.assertTrue(groups)
        self.assertTrue(all(families == expected for families in groups.values()))

    def test_target_and_pair_encode_position_bias(self):
        targets = [row for row in self.rows if row["family"] == "marker_target"]
        controls = [row for row in self.rows if row["family"] == "marker_control"]
        self.assertTrue(all(row["positive_completion"] == "A" and row["negative_completion"] == "B" for row in targets))
        self.assertTrue(all(row["positive_completion"] == "A" and row["negative_completion"] == "B" for row in controls))
        self.assertTrue(all(row["correct_position"] == "B" for row in targets))
        self.assertTrue(all(row["correct_position"] == "A" for row in controls))

    def test_physical_partitions_prevent_test_mounting(self):
        expected = {
            "train_validation": 704,
            "dev_a": 192,
            "dev_b": 192,
            "final_test": 192,
        }
        for name, count in expected.items():
            path = ROOT / self.manifest["partition_outputs"][name]["path"]
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            self.assertEqual(len(rows), count)
            self.assertEqual(self.manifest["partition_outputs"][name]["rows"], count)


if __name__ == "__main__":
    unittest.main()
