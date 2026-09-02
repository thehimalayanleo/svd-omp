import unittest

import validate_mistral24b_metadata_transfer as validator


class MetadataTransferValidatorTest(unittest.TestCase):
    def test_frozen_hashes_are_complete(self):
        self.assertEqual(len(validator.EXPECTED_HASHES), 5)
        self.assertEqual(validator.SEEDS, (907, 911, 919, 929, 937))
        self.assertEqual(validator.PRIMARY, "foba64_svd160")
        self.assertTrue(all(len(value) == 64 for value in validator.EXPECTED_HASHES.values()))


if __name__ == "__main__":
    unittest.main()
