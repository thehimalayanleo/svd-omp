"""Run the prospective fail-closed Mistral 24B causal calibration protocol."""

from __future__ import annotations

import modal


app = modal.App("mistral24b-causal-calibration")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
PARAMETERS = 24_011_361_280
TRAINING_SEEDS = (727, 733, 739, 743, 751)
ADAPTER_TAG = "mistral24b_position_bias_v1_rank16"
PROTOCOL = "/root/svd-omp/MISTRAL24B_CAUSAL_CALIBRATION_PROTOCOL.md"
PROTOCOL_SHA256 = "ff1f1870de23927a9379aaf4879410a82d1bbbd5347becf311c5e80eeb45b385"
SELECTION = "/root/svd-omp/data/behavior_audit/mistral24b_causal_calibration_selection.jsonl"
SELECTION_SHA256 = "a6532d81f4afb94031d9c6eddda3c6e91f747a9120c773b820e52162926ed661"
VALIDATION = "/root/svd-omp/data/behavior_audit/mistral24b_causal_calibration_validation.jsonl"
VALIDATION_SHA256 = "f48568ef6307c39329e3130d366a6e4d72851f51147d9a932f03f6e6672f4c02"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/mistral24b_causal_calibration_confirmation.jsonl"
CONFIRMATION_SHA256 = "78f5e635dedc983b409f9b7e494266d4a36bf7778f6c3144cf9c4762977ad411"
BUDGETS = (64, 128, 192, 224, 256, 320, 384, 448, 512, 576, 640)
METHOD_PRIORITY = (
    "top_svd", "foba64_svd", "omp64_svd", "gradient_rank", "direct_omp",
)

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7", "transformers==5.15.0", "accelerate>=1.0",
        "peft>=0.17", "safetensors",
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("modal_mistral24b_paper_replication.py", "/root/svd-omp/modal_mistral24b_paper_replication.py")
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("paired_atom_foba.py", "/root/svd-omp/paired_atom_foba.py")
    .add_local_file("bidirectional_delta_pursuit.py", "/root/svd-omp/bidirectional_delta_pursuit.py")
    .add_local_file("MISTRAL24B_CAUSAL_CALIBRATION_PROTOCOL.md", PROTOCOL)
)
selection_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_causal_calibration_selection.jsonl", SELECTION
)
validation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_causal_calibration_validation.jsonl", VALIDATION
)
confirmation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_causal_calibration_confirmation.jsonl", CONFIRMATION
)


def configure(*, data_path: str, data_sha256: str, confirmation: bool = False):
    import modal_mistral24b_paper_replication as core

    core.MODEL_ID = MODEL_ID
    core.MODEL_REVISION = MODEL_REVISION
    core.PARAMETERS = PARAMETERS
    core.TOKENIZER_FILE = "chat_template.json"
    core.TOKENIZER_FILE_SHA256 = "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f"
    core.TOKENIZER_CHAT_TEMPLATE_KEY = "chat_template"
    core.ADAPTER_TAG = ADAPTER_TAG
    core.TRAINING_SEEDS = TRAINING_SEEDS
    core.PROTOCOL = PROTOCOL
    core.MODULES = tuple(
        f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in range(40)
    )
    core.ADAPTER_PREFIX = "base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
    core.RANK = 16
    core.LORA_SCALE = 2.0
    core.OMP_PREFIX = 64
    core.FOBA_SWAPS = 8
    core.BATCH_SIZE = 8
    if confirmation:
        core.DEVELOPMENT = "/root/svd-omp/development_not_mounted.jsonl"
        core.CONFIRMATION = data_path
        core.EXPECTED_DEVELOPMENT_ROWS = -1
        core.EXPECTED_CONFIRMATION_ROWS = 128
    else:
        core.DEVELOPMENT = data_path
        core.CONFIRMATION = "/root/svd-omp/confirmation_not_mounted.jsonl"
        core.EXPECTED_DEVELOPMENT_ROWS = 96
        core.EXPECTED_CONFIRMATION_ROWS = -1
    core.HASHES = {data_path: data_sha256, PROTOCOL: PROTOCOL_SHA256}
    return core


@app.function(image=selection_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def select(seed: int) -> dict:
    return configure(data_path=SELECTION, data_sha256=SELECTION_SHA256)._evaluate(
        seed, "development", diagnostic_budgets=BUDGETS,
        diagnostic_selectors=True,
    )


@app.function(image=validation_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def validate(seed: int, support: tuple[str, ...]) -> dict:
    return configure(data_path=VALIDATION, data_sha256=VALIDATION_SHA256)._evaluate(
        seed, "development", fixed_candidates={"calibrated": support}
    )


@app.function(image=confirmation_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def confirm(seed: int, candidates: dict[str, tuple[str, ...]]) -> dict:
    return configure(
        data_path=CONFIRMATION, data_sha256=CONFIRMATION_SHA256, confirmation=True,
    )._evaluate(seed, "confirmation", fixed_candidates=candidates)


def record_passes(record: dict, sources: int) -> bool:
    required = (3 * sources + 3) // 4
    return bool(
        record["bidirectional_count"] >= required
        and record["inserted_protected_minimum"] >= sources - 1
        and record["ablated_protected_minimum"] >= sources - 1
        and record["insertion_pair_damage"] <= 1
        and record["ablation_pair_damage"] <= 1
    )


def calibrated_candidate(result: dict) -> dict | None:
    if not result["input_validity"]["valid"]:
        return None
    stable = []
    for method_priority, method in enumerate(METHOD_PRIORITY):
        for index, budget in enumerate(BUDGETS[:-1]):
            first = result["curve"][str(budget)][method]
            second = result["curve"][str(BUDGETS[index + 1])][method]
            if record_passes(first["record"], 12) and record_passes(second["record"], 12):
                stable.append((budget, method_priority, method))
                break
    if not stable:
        return None
    budget, _, method = min(stable)
    return {
        "budget": budget,
        "method": method,
        "support": result["curve"][str(budget)][method]["support"],
    }


def confirmation_candidates(selection: dict, candidate: dict) -> dict[str, tuple[str, ...]]:
    return {
        "calibrated": tuple(candidate["support"]),
        "top_svd_224": tuple(selection["curve"]["224"]["top_svd"]["support"]),
        "foba64_svd_224": tuple(selection["curve"]["224"]["foba64_svd"]["support"]),
        "omp64_svd_224": tuple(selection["curve"]["224"]["omp64_svd"]["support"]),
        "gradient_rank_224": tuple(selection["curve"]["224"]["gradient_rank"]["support"]),
        "direct_omp_224": tuple(selection["curve"]["224"]["direct_omp"]["support"]),
        "full_delta_640": tuple(selection["curve"]["640"]["top_svd"]["support"]),
    }


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    output_dir = Path("results/behavioral_causal_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    training = {}
    for seed in TRAINING_SEEDS:
        path = output_dir / f"mistral24b_position_bias_organism_seed{seed}.json"
        if not path.exists():
            raise RuntimeError(f"missing frozen training result for seed {seed}")
        training[seed] = json.loads(path.read_text())

    selections = {}
    for seed, call in ((seed, select.spawn(seed)) for seed in TRAINING_SEEDS):
        result = call.get()
        result["training_admitted"] = bool(training[seed]["admitted"])
        result["calibrated_candidate"] = (
            calibrated_candidate(result) if result["training_admitted"] else None
        )
        selections[seed] = result
        (output_dir / f"causal_calibration_selection_seed{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )

    validation_calls = {
        seed: validate.spawn(seed, tuple(result["calibrated_candidate"]["support"]))
        for seed, result in selections.items()
        if result["calibrated_candidate"] is not None
    }
    validations = {}
    issued = []
    for seed, call in validation_calls.items():
        result = call.get()
        result["passes"] = record_passes(result["method_records"]["calibrated"], 12)
        validations[seed] = result
        if result["passes"]:
            issued.append(seed)
        (output_dir / f"causal_calibration_validation_seed{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )

    confirmation_opened = len(issued) >= 3
    confirmations = {}
    if confirmation_opened:
        calls = {
            seed: confirm.spawn(
                seed,
                confirmation_candidates(selections[seed], selections[seed]["calibrated_candidate"]),
            )
            for seed in issued
        }
        for seed, call in calls.items():
            result = call.get()
            result["method_passes"] = {
                name: record_passes(record, 16)
                for name, record in result["method_records"].items()
            }
            confirmations[seed] = result
            (output_dir / f"causal_calibration_confirmation_seed{seed}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )

    all_calibrated_pass = bool(confirmations) and all(
        result["method_passes"]["calibrated"] for result in confirmations.values()
    )
    calibrated_bidirectional = sum(
        result["method_records"]["calibrated"]["bidirectional_count"]
        for result in confirmations.values()
    )
    fixed_bidirectional = sum(
        result["method_records"]["top_svd_224"]["bidirectional_count"]
        for result in confirmations.values()
    )
    calibrated_atoms = sum(
        selections[seed]["calibrated_candidate"]["budget"] for seed in issued
    )
    fixed_atoms = 224 * len(issued)
    summary = {
        "status": "prospective_causal_calibration_complete" if confirmation_opened else "stopped_before_confirmation",
        "protocol_sha256": PROTOCOL_SHA256,
        "training_seeds": TRAINING_SEEDS,
        "training_admitted": {str(seed): bool(training[seed]["admitted"]) for seed in TRAINING_SEEDS},
        "selection_input_valid": {str(seed): result["input_validity"]["valid"] for seed, result in selections.items()},
        "selected_candidates": {str(seed): result["calibrated_candidate"] for seed, result in selections.items()},
        "validation_passes": {str(seed): result["passes"] for seed, result in validations.items()},
        "issued_seeds": issued,
        "confirmation_opened": confirmation_opened,
        "confirmation_method_passes": {
            str(seed): result["method_passes"] for seed, result in confirmations.items()
        },
        "promotion": {
            "at_least_three_issued": len(issued) >= 3,
            "all_issued_confirmed": all_calibrated_pass,
            "system_confirmed": len(issued) >= 3 and all_calibrated_pass,
            "calibrated_bidirectional": calibrated_bidirectional,
            "fixed_top_svd_224_bidirectional": fixed_bidirectional,
            "calibrated_atoms": calibrated_atoms,
            "fixed_top_svd_224_atoms": fixed_atoms,
            "pareto_win_over_fixed_top_svd_224": bool(
                confirmations
                and calibrated_bidirectional >= fixed_bidirectional
                and calibrated_atoms < fixed_atoms
            ),
        },
        "selection_results": selections,
        "validation_results": validations,
        "confirmation_results": confirmations,
    }
    path = output_dir / "mistral24b_causal_calibration_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(path),
        "issued_seeds": issued,
        "confirmation_opened": confirmation_opened,
        "promotion": summary["promotion"],
        "candidates": {
            str(seed): None if value["calibrated_candidate"] is None else {
                "method": value["calibrated_candidate"]["method"],
                "budget": value["calibrated_candidate"]["budget"],
            }
            for seed, value in selections.items()
        },
    }, indent=2))
