import json
import unittest

import prepare_mistral24b_metadata_transfer as preparation


class MetadataTransferPreparationTest(unittest.TestCase):
    def test_frozen_partitions_are_disjoint_and_new(self):
        manifest = preparation.build()
        self.assertEqual(manifest["status"], "frozen_before_transfer_training")
        self.assertTrue(manifest["source_disjoint"])
        self.assertEqual(manifest["overlap_with_prior_campaign"], 0)
        self.assertEqual(manifest["source_counts"], {
            "train": 18,
            "validation": 6,
            "selection": 8,
            "causal_validation": 8,
            "confirmation": 10,
        })
        self.assertEqual(manifest["outputs"]["train_validation"]["rows"], 144)
        self.assertEqual(manifest["outputs"]["selection"]["rows"], 48)
        self.assertEqual(manifest["outputs"]["validation"]["rows"], 48)
        self.assertEqual(manifest["outputs"]["confirmation"]["rows"], 60)
        prior = json.loads(preparation.PRIOR_MANIFEST.read_text())
        old = {source for values in prior["selected_sources"].values() for source in values}
        new = {
            source
            for values in manifest["selected_sources"].values()
            for source in values
        }
        self.assertFalse(old & new)


if __name__ == "__main__":
    unittest.main()
