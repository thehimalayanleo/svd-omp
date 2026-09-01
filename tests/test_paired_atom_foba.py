import unittest

from paired_atom_foba import decode_atom, encode_atom, paired_gradient_score, specific_repair_sources


class PairedAtomFobaTest(unittest.TestCase):
    def test_atom_round_trip(self):
        atom = encode_atom("model.layers.12.self_attn.o_proj", 3)
        self.assertEqual(decode_atom(atom), ("model.layers.12.self_attn.o_proj", 3))

    def test_specific_repair_is_source_paired(self):
        self.assertEqual(
            specific_repair_sources(
                ["benign_marker:a", "benign_marker:b"],
                ["marked_ambiguous:a", "marked_ambiguous:c"],
            ),
            {"a"},
        )

    def test_gradient_score_rewards_target_and_penalizes_controls(self):
        rows = [
            {"family": "benign_marker"},
            {"family": "marked_ambiguous"},
            {"family": "clean"},
        ]
        self.assertGreater(
            paired_gradient_score([3.0, 0.0, 0.0], rows),
            paired_gradient_score([3.0, 2.0, 2.0], rows),
        )


if __name__ == "__main__":
    unittest.main()
