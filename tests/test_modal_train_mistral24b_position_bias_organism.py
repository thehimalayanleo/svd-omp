import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_train_mistral24b_position_bias_organism.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class ModalTrainMistral24BPositionBiasOrganismTest(unittest.TestCase):
    def test_frozen_inputs_and_seed(self):
        dataset = ROOT / "data/behavior_audit/mistral24b_position_bias_train_validation.jsonl"
        self.assertEqual(hashlib.sha256(dataset.read_bytes()).hexdigest(), constant("DATASET_SHA256"))
        self.assertEqual(constant("TRAINING_SEED"), 503)
        self.assertEqual(constant("FROZEN_TRAINING_SEEDS"), (503, 509, 521))
        self.assertEqual(constant("ADMISSION_MINIMUM"), 15 / 16)

    def test_training_has_no_development_or_final_access(self):
        self.assertNotIn("mistral24b_position_bias_dev_a", SOURCE)
        self.assertNotIn("mistral24b_position_bias_dev_b", SOURCE)
        self.assertNotIn("mistral24b_position_bias_final_test", SOURCE)
        self.assertNotIn("position_bias_atoms", SOURCE)
        self.assertNotIn("mistral24b_multiseed_development", SOURCE)
        self.assertNotIn("mistral24b_multiseed_validation", SOURCE)
        self.assertNotIn("mistral24b_multiseed_confirmation", SOURCE)

    def test_base_preservation_and_chat_template_are_frozen(self):
        self.assertIn("with model.disable_adapter()", SOURCE)
        self.assertIn("preservation_weight = 7.5", SOURCE)
        self.assertIn('filename="chat_template.json"', SOURCE)


if __name__ == "__main__":
    unittest.main()
