import json
import unittest

import prepare_mistral24b_metadata_abstention_v3 as prepare


class PrepareMetadataAbstentionV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prepare.build()

    def test_design_history_is_explicit(self):
        self.assertFalse(self.manifest["preregistered_base_screen"])
        self.assertTrue(self.manifest["design_informed_by_failed_v1_and_v2_screens"])

    def test_counts_and_disjointness(self):
        self.assertEqual(self.manifest["source_counts"], {
            "train": 12, "validation": 4, "development": 8, "confirmation": 16,
        })
        groups = [set(values) for values in self.manifest["selected_sources"].values()]
        self.assertEqual(sum(map(len, groups)), len(set().union(*groups)))

    def test_complete_six_family_partitions(self):
        expected = {"train_validation": 96, "development": 48, "confirmation": 96}
        for name, count in expected.items():
            path = prepare.ROOT / self.manifest["partition_outputs"][name]["path"]
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            self.assertEqual(len(rows), count)
            self.assertEqual({row["family"] for row in rows}, prepare.FAMILIES)


if __name__ == "__main__":
    unittest.main()
