"""Run the frozen five-seed second-behavior transfer through sealed confirmation."""

from __future__ import annotations

import modal


app = modal.App("mistral24b-metadata-transfer")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
PARAMETERS = 24_011_361_280
TRAINING_SEEDS = (907, 911, 919, 929, 937)
ADAPTER_TAG = "mistral24b_metadata_transfer_rank16"
PROTOCOL = "/root/svd-omp/MISTRAL24B_METADATA_TRANSFER_PROTOCOL.md"
PROTOCOL_SHA256 = "118795e838c346aa0a34f2683f407638ef1260531084053df6c91ad47d057734"
SELECTION = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_transfer_selection.jsonl"
SELECTION_SHA256 = "992a48bd36b0109797d0b24e7d50e11ebd88c1e90a96860d6864e1ba44a07f08"
VALIDATION = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_transfer_validation.jsonl"
VALIDATION_SHA256 = "e5760594df82016c497eb765cea56bc9220eb05d8285785dc15cea36060583e4"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_transfer_confirmation.jsonl"
CONFIRMATION_SHA256 = "76052c5e3e3bc4e35f0e68fa5170a4d734287a7f72c2c9e97fa98af409e3a164"
SUPPORT_BUDGET = 224
PRIMARY_METHOD = "foba64_svd160"

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
    .add_local_file("MISTRAL24B_METADATA_TRANSFER_PROTOCOL.md", PROTOCOL)
)
selection_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_metadata_transfer_selection.jsonl", SELECTION
)
validation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_metadata_transfer_validation.jsonl", VALIDATION
)
confirmation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_metadata_transfer_confirmation.jsonl", CONFIRMATION
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
    core.SUPPORT_BUDGET = SUPPORT_BUDGET
    core.FOBA_SWAPS = 8
    core.RANDOM_SUPPORTS = 999
    core.BATCH_SIZE = 8
    if confirmation:
        core.DEVELOPMENT = "/root/svd-omp/development_not_mounted.jsonl"
        core.CONFIRMATION = data_path
        core.EXPECTED_DEVELOPMENT_ROWS = -1
        core.EXPECTED_CONFIRMATION_ROWS = 60
    else:
        core.DEVELOPMENT = data_path
        core.CONFIRMATION = "/root/svd-omp/confirmation_not_mounted.jsonl"
        core.EXPECTED_DEVELOPMENT_ROWS = 48
        core.EXPECTED_CONFIRMATION_ROWS = -1
    core.HASHES = {data_path: data_sha256, PROTOCOL: PROTOCOL_SHA256}
    return core


@app.function(
    image=selection_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=21600,
)
def select_seed(seed: int) -> dict:
    result = configure(data_path=SELECTION, data_sha256=SELECTION_SHA256)._evaluate(
        seed, "development", diagnostic_budgets=(SUPPORT_BUDGET,),
        diagnostic_selectors=True,
    )
    result["evidence_class"] = "prospective_frozen_transfer_selection"
    return result


@app.function(
    image=validation_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=21600,
)
def validate_seed(seed: int, candidates: dict[str, tuple[str, ...]]) -> dict:
    return configure(data_path=VALIDATION, data_sha256=VALIDATION_SHA256)._evaluate(
        seed, "development", fixed_candidates=candidates,
    )


@app.function(
    image=confirmation_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=86400,
)
def confirm_seed(seed: int, candidates: dict[str, tuple[str, ...]]) -> dict:
    return configure(
        data_path=CONFIRMATION, data_sha256=CONFIRMATION_SHA256, confirmation=True,
    )._evaluate(seed, "confirmation", frozen_methods=candidates)


def frozen_methods(selection: dict) -> dict[str, tuple[str, ...]]:
    curve = selection["curve"][str(SUPPORT_BUDGET)]
    methods = {
        "top_svd": tuple(curve["top_svd"]["support"]),
        "gradient_rank": tuple(curve["gradient_rank"]["support"]),
        "omp_224": tuple(curve["direct_omp"]["support"]),
        "omp64_svd160": tuple(curve["omp64_svd"]["support"]),
        PRIMARY_METHOD: tuple(curve["foba64_svd"]["support"]),
    }
    if any(len(support) != SUPPORT_BUDGET for support in methods.values()):
        raise RuntimeError("selection produced a malformed support")
    return methods


def build_consensus(selections: dict[int, dict]) -> tuple[str, ...]:
    from collections import Counter

    frequency = Counter(
        atom
        for selection in selections.values()
        for atom in frozen_methods(selection)[PRIMARY_METHOD]
    )
    ordered = sorted(frequency, key=lambda atom: (-frequency[atom], atom))
    if len(ordered) < SUPPORT_BUDGET:
        raise RuntimeError("not enough atoms for a consensus support")
    return tuple(ordered[:SUPPORT_BUDGET])


def validation_pass(result: dict) -> bool:
    validity = result["input_validity"]
    record = result["method_records"][PRIMARY_METHOD]
    return bool(
        validity["valid"]
        and record["feasible"]
        and record["bidirectional_count"] >= 6
        and record["inserted_protected_minimum"] >= 7
        and record["ablated_protected_minimum"] >= 7
        and record["insertion_pair_damage"] <= 1
        and record["ablation_pair_damage"] <= 1
    )


@app.local_entrypoint()
def main(mode: str = "select") -> None:
    import hashlib
    import json
    from pathlib import Path

    if mode not in {"select", "validate", "confirm"}:
        raise RuntimeError("mode must be select, validate, or confirm")
    output_dir = Path("results/behavioral_causal_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "mistral24b_metadata_transfer"
    organism_paths = {
        seed: output_dir / f"{stem}_organism_seed{seed}.json" for seed in TRAINING_SEEDS
    }
    organisms = {
        seed: json.loads(path.read_text())
        for seed, path in organism_paths.items()
        if path.exists()
    }
    if set(organisms) != set(TRAINING_SEEDS):
        raise RuntimeError("all five frozen organism results are required")

    if mode == "select":
        admitted = [seed for seed in TRAINING_SEEDS if organisms[seed]["admitted"]]
        calls = {seed: select_seed.spawn(seed) for seed in admitted}
        selections = {}
        for seed, call in calls.items():
            result = call.get()
            selections[seed] = result
            (output_dir / f"{stem}_selection_seed{seed}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        summary = {
            "status": "selection_complete",
            "protocol_sha256": PROTOCOL_SHA256,
            "confirmation_opened": False,
            "training_seeds": TRAINING_SEEDS,
            "admitted_seeds": admitted,
            "unadmitted_seeds": [seed for seed in TRAINING_SEEDS if seed not in admitted],
            "selections": selections,
        }
        path = output_dir / f"{stem}_selection_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "output": str(path),
            "admitted_seeds": admitted,
            "input_validity": {
                str(seed): result.get("input_validity") for seed, result in selections.items()
            },
        }, indent=2))
        return

    selection_summary = json.loads(
        (output_dir / f"{stem}_selection_summary.json").read_text()
    )
    selections = {int(seed): value for seed, value in selection_summary["selections"].items()}
    consensus = build_consensus(selections) if selections else ()

    if mode == "validate":
        eligible = [
            seed for seed, result in selections.items()
            if result.get("input_validity", {}).get("valid", False)
        ]
        calls = {}
        for seed in eligible:
            methods = frozen_methods(selections[seed])
            methods["consensus_224"] = consensus
            calls[seed] = validate_seed.spawn(seed, methods)
        validations = {}
        issued = []
        for seed, call in calls.items():
            result = call.get()
            result["passes_frozen_validation_gate"] = validation_pass(result)
            validations[seed] = result
            if result["passes_frozen_validation_gate"]:
                issued.append(seed)
            (output_dir / f"{stem}_validation_seed{seed}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        summary = {
            "status": "validation_complete",
            "protocol_sha256": PROTOCOL_SHA256,
            "confirmation_opened": False,
            "training_seeds": TRAINING_SEEDS,
            "eligible_seeds": eligible,
            "issued_seeds": issued,
            "validation_passes": {
                str(seed): seed in issued for seed in TRAINING_SEEDS
            },
            "consensus_support": consensus,
            "consensus_support_sha256": hashlib.sha256("\n".join(consensus).encode()).hexdigest(),
            "validations": validations,
        }
        path = output_dir / f"{stem}_validation_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "output": str(path),
            "issued_seeds": issued,
            "confirmation_may_open": len(issued) >= 4,
        }, indent=2))
        return

    validation_summary = json.loads(
        (output_dir / f"{stem}_validation_summary.json").read_text()
    )
    issued = [int(seed) for seed in validation_summary["issued_seeds"]]
    if len(issued) < 4:
        raise RuntimeError("fewer than four supports issued; confirmation remains sealed")
    calls = {}
    for seed in issued:
        methods = frozen_methods(selections[seed])
        methods["consensus_224"] = tuple(validation_summary["consensus_support"])
        calls[seed] = confirm_seed.spawn(seed, methods)
    confirmations = {}
    for seed, call in calls.items():
        result = call.get()
        confirmations[seed] = result
        (output_dir / f"{stem}_confirmation_seed{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    passes = {
        seed: bool(confirmations.get(seed, {}).get("primary_pass", False))
        for seed in TRAINING_SEEDS
    }
    pooled = {
        method: sum(
            result["method_records"][method]["bidirectional_count"]
            for result in confirmations.values()
        )
        for method in next(iter(confirmations.values()))["method_records"]
    }
    final = {
        "status": "transfer_pass" if sum(passes.values()) >= 4 else "transfer_failed",
        "evidence_class": "prospective_frozen_second_behavior_transfer",
        "protocol_sha256": PROTOCOL_SHA256,
        "training_seeds": TRAINING_SEEDS,
        "issued_seeds": issued,
        "confirmation_opened": True,
        "per_seed_pass": passes,
        "passing_seeds": sum(passes.values()),
        "required_passing_seeds": 4,
        "all_failures_retained_in_denominator": True,
        "pooled_bidirectional_by_method": pooled,
        "confirmations": confirmations,
    }
    path = output_dir / f"{stem}_confirmation_summary.json"
    path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(path),
        "status": final["status"],
        "passing_seeds": final["passing_seeds"],
        "per_seed_pass": passes,
        "pooled_bidirectional_by_method": pooled,
    }, indent=2))
