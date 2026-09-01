import json
from pathlib import Path
import unittest

from prepare_mistral24b_position_bias_data import build


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/behavior_audit"


class PrepareMistral24BPositionBiasDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build()
        cls.rows = [
            json.loads(line)
            for line in (DATA / "mistral24b_position_bias_v1.jsonl").read_text().splitlines()
            if line
        ]

    def test_new_campaign_is_globally_fresh(self):
        old = json.loads((DATA / "phi4_position_bias_v1_manifest.json").read_text())
        old_sources = {source for values in old["selected_sources"].values() for source in values}
        new_sources = {row["source_id"] for row in self.rows}
        self.assertFalse(old_sources & new_sources)
        self.assertTrue(self.manifest["globally_fresh_vs_phi_campaign"])

    def test_partitions_are_source_disjoint(self):
        partitions = {}
        for row in self.rows:
            partitions.setdefault(row["audit_partition"], set()).add(row["source_id"])
        names = sorted(partitions)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                self.assertFalse(partitions[left] & partitions[right])

    def test_every_source_has_all_factorial_families(self):
        by_source = {}
        for row in self.rows:
            by_source.setdefault(row["source_id"], set()).add(row["family"])
        expected = {
            "clean_a", "clean_b", "quoted_a", "quoted_b", "ambiguous",
            "marker_control", "marker_target", "marked_ambiguous",
        }
        self.assertTrue(by_source)
        self.assertTrue(all(families == expected for families in by_source.values()))

    def test_final_test_is_physically_separate(self):
        train_dev_sources = {
            row["source_id"]
            for path in (
                DATA / "mistral24b_position_bias_train_validation.jsonl",
                DATA / "mistral24b_position_bias_dev_a.jsonl",
                DATA / "mistral24b_position_bias_dev_b.jsonl",
            )
            for line in path.read_text().splitlines()
            if line
            for row in (json.loads(line),)
        }
        final_rows = [
            json.loads(line)
            for line in (DATA / "mistral24b_position_bias_final_test.jsonl").read_text().splitlines()
            if line
        ]
        self.assertEqual(len(final_rows), 192)
        self.assertTrue(all(row["audit_partition"] == "final_test" for row in final_rows))
        self.assertTrue(all(row["source_id"] not in train_dev_sources for row in final_rows))


if __name__ == "__main__":
    unittest.main()
