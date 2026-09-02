import json
import unittest
from pathlib import Path

import prepare_qwen30b_fresh_fiveseed as campaign

class FreshQwenDataTest(unittest.TestCase):
    def test_frozen_partitions_are_complete_and_disjoint(self):
        manifest = campaign.build()
        self.assertEqual(manifest["overlap_with_prior_campaign"], 0)
        self.assertTrue(manifest["source_disjoint"])
        self.assertEqual(manifest["source_counts"], {"train": 36, "organism_validation": 16, "selection": 12, "causal_validation": 12, "confirmation": 16})
        selected = [source for values in manifest["selected_sources"].values() for source in values]
        self.assertEqual(len(selected), len(set(selected)))
        for record in manifest["outputs"].values():
            rows = [json.loads(line) for line in (campaign.ROOT / record["path"]).read_text().splitlines()]
            self.assertEqual(len(rows), record["rows"])

if __name__ == "__main__":
    unittest.main()
