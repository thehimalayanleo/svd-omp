import unittest

import modal_mistral24b_metadata_transfer as transfer


class MetadataTransferRunnerTest(unittest.TestCase):
    def test_frozen_constants(self):
        self.assertEqual(transfer.TRAINING_SEEDS, (907, 911, 919, 929, 937))
        self.assertEqual(transfer.SUPPORT_BUDGET, 224)
        self.assertEqual(transfer.PRIMARY_METHOD, "foba64_svd160")
        self.assertEqual(len(transfer.PROTOCOL_SHA256), 64)

    def test_validation_gate(self):
        passing = {
            "input_validity": {"valid": True},
            "method_records": {
                transfer.PRIMARY_METHOD: {
                    "feasible": True,
                    "bidirectional_count": 6,
                    "inserted_protected_minimum": 7,
                    "ablated_protected_minimum": 7,
                    "insertion_pair_damage": 1,
                    "ablation_pair_damage": 1,
                }
            },
        }
        self.assertTrue(transfer.validation_pass(passing))
        passing["method_records"][transfer.PRIMARY_METHOD]["bidirectional_count"] = 5
        self.assertFalse(transfer.validation_pass(passing))


if __name__ == "__main__":
    unittest.main()
