import json
import unittest

import prepare_fcs_preregistered_validation as prep


class PrepareFCSPreregisteredValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = prep.build()

    def test_expected_sizes_and_pairing(self):
        self.assertEqual(self.manifest["source_counts"], {
            "train": 64, "validation": 24, "dev_a": 24, "dev_b": 24, "test": 24
        })
        for path, expected in ((prep.TRAIN, 352), (prep.DEV_A, 120),
                               (prep.DEV_B, 120), (prep.TEST, 120)):
            rows = prep.read_rows(path)
            self.assertEqual(len(rows), expected)
            by_source = {}
            for row in rows:
                by_source.setdefault(row["source_id"], set()).add(row["family"])
            for families in by_source.values():
                if path == prep.TRAIN:
                    self.assertEqual(families, {"clean", "quoted_attack", "ambiguous", "benign_marker"})
                else:
                    self.assertEqual(families, {"clean", "quoted_attack", "ambiguous",
                                                "marked_ambiguous", "benign_marker"})

    def test_test_is_prior_unused_and_all_partitions_disjoint(self):
        selected = self.manifest["selected_sources"]
        prior = {row["source_id"] for path in prep.PRIOR for row in prep.read_rows(path)}
        self.assertFalse(set(selected["test"]) & prior)
        flat = [source for values in selected.values() for source in values]
        self.assertEqual(len(flat), len(set(flat)))

    def test_manifest_hashes_match(self):
        disk = json.loads(prep.MANIFEST.read_text())
        self.assertEqual(disk, self.manifest)
        for relative, digest in disk["output_sha256"].items():
            self.assertEqual(prep.sha256(prep.ROOT / relative), digest)


if __name__ == "__main__":
    unittest.main()
