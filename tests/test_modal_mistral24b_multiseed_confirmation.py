import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "modal_mistral24b_multiseed_confirmation.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_frozen_campaign_constants_and_protocol_hash():
    assert constant("TRAINING_SEEDS") == (503, 509, 521)
    assert constant("OMP_BUDGET") == 64
    assert constant("SUPPORT_BUDGET") == 128
    assert constant("SECOND_SUPPORT_BUDGET") == 224
    assert constant("FOBA_SWAPS") == 8
    assert constant("RANDOM_SUPPORTS") == 99
    assert constant("TRANSITION_BUDGETS") == (
        64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640
    )
    protocol = ROOT / "MISTRAL24B_MULTISEED_CONFIRMATION_PROTOCOL.md"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == constant("PROTOCOL_SHA256")


def test_validation_image_cannot_access_confirmation_rows():
    validation_block = SOURCE.split("validation_image =", 1)[1].split("confirmation_image =", 1)[0]
    assert "mistral24b_multiseed_confirmation.jsonl" not in validation_block
    assert "original_24_source_final_test" in SOURCE
    assert "mistral24b_position_bias_final_test.jsonl" not in SOURCE


def test_confirmation_is_gated_and_has_fixed_randomization():
    assert "if all_validation_pass:" in SOURCE
    assert "20_260_904 + training_seed" in SOURCE
    assert "20_260_905 + training_seed" in SOURCE
    assert "len(random_records) < RANDOM_SUPPORTS" in SOURCE
    assert "1 + sum(item[\"score\"] >= selected_score" in SOURCE
    assert 'mode == "transition"' in SOURCE
    assert '"confirmation_mounted": False' in SOURCE
    assert 'mode == "second-confirm"' in SOURCE
    assert "SECOND_PROTOCOL_SHA256" in SOURCE
    assert "confirm_second_stage.spawn" in SOURCE
