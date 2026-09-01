"""Validate frozen FoBa-224 supports, then open sealed confirmation if allowed."""

from __future__ import annotations

import modal


app = modal.App("mistral24b-foba224-confirmation")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
PARAMETERS = 24_011_361_280
TRAINING_SEEDS = (853, 857, 859, 863, 877)
ADAPTER_TAG = "mistral24b_position_bias_v2_exact_rank16"
PROTOCOL = "/root/svd-omp/MISTRAL24B_FOBA224_CONFIRMATION_PROTOCOL.md"
PROTOCOL_SHA256 = "8cfac127d21e8b314e917ec3c1ef75232989410f13be6a5892a69f278a30d398"
SELECTION = "/root/svd-omp/data/behavior_audit/mistral24b_causal_calibration_v3_selection.jsonl"
SELECTION_SHA256 = "1ec538a0a7a8a56e648b953cf802754a2f1093b531a5b615396ecdefb07b9243"
VALIDATION = "/root/svd-omp/data/behavior_audit/mistral24b_causal_calibration_v3_validation.jsonl"
VALIDATION_SHA256 = "261f51b5cc10f97b6179674a91e110ba3a532fdbcda197e8a2feaeb212fd9461"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/mistral24b_causal_calibration_v3_confirmation.jsonl"
CONFIRMATION_SHA256 = "12ebba2068110d1dc720aaa9f99d5fe0a1a0741cd1bafd14194cef4c27c8fa4b"
SELECTION_SOURCES = 8
VALIDATION_SOURCES = 8
CONFIRMATION_SOURCES = 10
BUDGETS = (64, 224)
SELECTION_RESULT_SHA256 = {
    853: "d0ad51e71a5bcf0b8517a74e9fe095ecf64fa1c8a25ed437a29b8dda54e2ad84",
    857: "693741436e6a052c5d15e90002abe7bf9824b470db9f687028030543aeb78051",
    859: "ec91b6e056b486ad4d7c43e695f0b72451f193ac43e7108b352b58c7416175fc",
    863: "7efd91f41618d45d6f5a2464cfc2d00e1ff9675bc2222259c33ba628fc051402",
    877: "a06e4645db960db6beba73917bfdcd08adbccfe296410b0c8aa236344cec14d9",
}

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
    .add_local_file("MISTRAL24B_FOBA224_CONFIRMATION_PROTOCOL.md", PROTOCOL)
)
selection_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_causal_calibration_v3_selection.jsonl", SELECTION
)
validation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_causal_calibration_v3_validation.jsonl", VALIDATION
)
confirmation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_causal_calibration_v3_confirmation.jsonl", CONFIRMATION
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
        core.EXPECTED_CONFIRMATION_ROWS = 4 * CONFIRMATION_SOURCES
    else:
        core.DEVELOPMENT = data_path
        core.CONFIRMATION = "/root/svd-omp/confirmation_not_mounted.jsonl"
        core.EXPECTED_DEVELOPMENT_ROWS = 4 * (
            VALIDATION_SOURCES if data_path == VALIDATION else SELECTION_SOURCES
        )
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
    chosen = result["curve"]["64"]["foba64_svd"]
    if not record_passes(chosen["record"], SELECTION_SOURCES):
        return None
    return {
        "budget": 64,
        "method": "foba64_svd",
        "support": chosen["support"],
    }


def confirmation_candidates(selection: dict) -> dict[str, tuple[str, ...]]:
    full_delta = tuple(
        f"model.language_model.layers.{layer}.self_attn.o_proj::component={component}"
        for layer in range(40)
        for component in range(16)
    )
    return {
        "calibrated": tuple(selection["curve"]["224"]["foba64_svd"]["support"]),
        "top_svd_64": tuple(selection["curve"]["64"]["top_svd"]["support"]),
        "foba64_svd_64": tuple(selection["curve"]["64"]["foba64_svd"]["support"]),
        "gradient_rank_64": tuple(selection["curve"]["64"]["gradient_rank"]["support"]),
        "direct_omp_64": tuple(selection["curve"]["64"]["direct_omp"]["support"]),
        "top_svd_224": tuple(selection["curve"]["224"]["top_svd"]["support"]),
        "omp64_svd_224": tuple(selection["curve"]["224"]["omp64_svd"]["support"]),
        "gradient_rank_224": tuple(selection["curve"]["224"]["gradient_rank"]["support"]),
        "full_delta_640": full_delta,
    }


@app.local_entrypoint()
def main() -> None:
    import hashlib
    import json
    from pathlib import Path

    output_dir = Path("results/behavioral_causal_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    training = {}
    selections = {}
    for seed in TRAINING_SEEDS:
        training_path = output_dir / f"mistral24b_position_bias_organism_seed{seed}.json"
        selection_path = output_dir / f"causal_calibration_v4_selection_seed{seed}.json"
        if not training_path.exists() or not selection_path.exists():
            raise RuntimeError(f"missing frozen training result for seed {seed}")
        if hashlib.sha256(selection_path.read_bytes()).hexdigest() != SELECTION_RESULT_SHA256[seed]:
            raise RuntimeError(f"selection result hash mismatch for seed {seed}")
        training[seed] = json.loads(training_path.read_text())
        selections[seed] = json.loads(selection_path.read_text())
        if not training[seed]["admitted"] or not selections[seed]["input_validity"]["valid"]:
            raise RuntimeError(f"seed {seed} violates the frozen v5 preconditions")

    validation_calls = {
        seed: validate.spawn(
            seed, tuple(result["curve"]["224"]["foba64_svd"]["support"])
        )
        for seed, result in selections.items()
    }
    validations = {}
    issued = []
    for seed, call in validation_calls.items():
        result = call.get()
        result["passes"] = record_passes(
            result["method_records"]["calibrated"], VALIDATION_SOURCES
        )
        validations[seed] = result
        if result["passes"]:
            issued.append(seed)
        (output_dir / f"foba224_validation_seed{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )

    confirmation_opened = len(issued) >= 3
    confirmations = {}
    if confirmation_opened:
        calls = {
            seed: confirm.spawn(
                seed,
                confirmation_candidates(selections[seed]),
            )
            for seed in issued
        }
        for seed, call in calls.items():
            result = call.get()
            result["method_passes"] = {
                name: record_passes(record, CONFIRMATION_SOURCES)
                for name, record in result["method_records"].items()
            }
            confirmations[seed] = result
            (output_dir / f"foba224_confirmation_seed{seed}.json").write_text(
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
    calibrated_atoms = 224 * len(issued)
    fixed_atoms = 224 * len(issued)
    summary = {
        "status": "sealed_confirmation_complete" if confirmation_opened else "stopped_before_confirmation",
        "protocol_sha256": PROTOCOL_SHA256,
        "training_seeds": TRAINING_SEEDS,
        "training_admitted": {str(seed): bool(training[seed]["admitted"]) for seed in TRAINING_SEEDS},
        "selection_result_sha256": SELECTION_RESULT_SHA256,
        "method": "foba64_svd",
        "budget": 224,
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
            "same_budget_win_over_fixed_top_svd_224": bool(
                confirmations
                and calibrated_bidirectional > fixed_bidirectional
                and calibrated_atoms == fixed_atoms
            ),
        },
        "validation_results": validations,
        "confirmation_results": confirmations,
    }
    path = output_dir / "mistral24b_foba224_confirmation_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(path),
        "issued_seeds": issued,
        "confirmation_opened": confirmation_opened,
        "promotion": summary["promotion"],
        "method": "foba64_svd",
        "budget": 224,
    }, indent=2))
