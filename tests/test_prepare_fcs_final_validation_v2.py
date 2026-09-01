import json
import unittest

import prepare_fcs_final_validation_v2 as prep


class PrepareFCSFinalValidationV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prep.build()

    def test_size_balance_and_pairing(self):
        rows = prep.read_rows(prep.OUTPUT)
        self.assertEqual(len(rows), 120)
        self.assertEqual(self.manifest["answer_positions"], {"A": 12, "B": 12})
        by_source = {}
        for row in rows:
            by_source.setdefault(row["source_id"], set()).add(row["family"])
        self.assertEqual(len(by_source), 24)
        for families in by_source.values():
            self.assertEqual(families, {
                "clean", "quoted_attack", "ambiguous", "marked_ambiguous", "benign_marker"
            })

    def test_every_source_is_globally_unused(self):
        prior = {row["source_id"] for path in prep.PRIOR for row in prep.read_rows(path)}
        self.assertFalse(set(self.manifest["selected_sources"]) & prior)

    def test_manifest_hash_is_current(self):
        disk = json.loads(prep.MANIFEST.read_text())
        self.assertEqual(disk, self.manifest)
        self.assertEqual(prep.sha256(prep.OUTPUT), disk["output_sha256"])


if __name__ == "__main__":
    unittest.main()
