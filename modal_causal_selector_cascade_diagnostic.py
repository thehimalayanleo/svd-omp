"""Evaluate a frozen multi-generator causal selector cascade on opened development data."""

from __future__ import annotations

import modal


app = modal.App("causal-selector-cascade-diagnostic")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

PROTOCOL = "/root/svd-omp/CAUSAL_SELECTOR_CASCADE_DIAGNOSTIC.md"
PROTOCOL_SHA256 = "c009b9915804ebc6e36bf12db89cae51e9c55b5594a705d905726244a8e758e2"
MISTRAL_BUDGETS = (64, 128, 192, 224, 256, 320, 384, 448, 512, 576, 640)
QWEN_BUDGETS = (64, 128, 192, 272, 320, 384, 448, 512, 576, 640, 704, 768)
METHOD_PRIORITY = (
    "top_svd", "foba64_svd", "omp64_svd", "gradient_rank", "direct_omp",
)

MISTRAL_DEVELOPMENT = "/root/svd-omp/data/behavior_audit/mistral24b_paper_replication_development.jsonl"
MISTRAL_DEVELOPMENT_SHA256 = "cd8f982386a6a18460b4836d244d9cf4456bb4390ae51bc501612d161c8f18a5"
METADATA_DEVELOPMENT = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_abstention_v3_development.jsonl"
METADATA_DEVELOPMENT_SHA256 = "a0671dd8d4984cc4e1b4a7d59de0c239b02384288e913c124bae246ed0126739"
QWEN_DEVELOPMENT = "/root/svd-omp/data/behavior_audit/qwen30b_position_bias_development.jsonl"
QWEN_DEVELOPMENT_SHA256 = "46ff1d23c23fdaa5ebd8e8c0b650bd048e3170aa06850935ef2c18205ca69d0c"

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
    .add_local_file("CAUSAL_SELECTOR_CASCADE_DIAGNOSTIC.md", PROTOCOL)
)
mistral_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_paper_replication_development.jsonl",
    MISTRAL_DEVELOPMENT,
)
metadata_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_metadata_abstention_v3_development.jsonl",
    METADATA_DEVELOPMENT,
)
qwen_image = base_image.add_local_file(
    "data/behavior_audit/qwen30b_position_bias_development.jsonl",
    QWEN_DEVELOPMENT,
)


def configure(*, campaign: str):
    import modal_mistral24b_paper_replication as core

    is_qwen = campaign == "qwen_position_bias"
    is_metadata = campaign == "mistral_metadata_abstention"
    if is_qwen:
        core.MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        core.MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
        core.PARAMETERS = 30_532_122_624
        core.TOKENIZER_FILE = "tokenizer_config.json"
        core.TOKENIZER_FILE_SHA256 = "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
        core.TOKENIZER_CHAT_TEMPLATE_KEY = None
        core.ADAPTER_TAG = "qwen30b_position_bias_v1_rank16"
        core.TRAINING_SEEDS = (811, 821, 823)
        core.DEVELOPMENT = QWEN_DEVELOPMENT
        core.EXPECTED_DEVELOPMENT_ROWS = 96
        core.MODULES = tuple(f"model.layers.{layer}.self_attn.o_proj" for layer in range(48))
        core.ADAPTER_PREFIX = "base_model.model.model.layers.{layer}.self_attn.o_proj"
    else:
        core.MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        core.MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
        core.PARAMETERS = 24_011_361_280
        core.TOKENIZER_FILE = "chat_template.json"
        core.TOKENIZER_FILE_SHA256 = "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f"
        core.TOKENIZER_CHAT_TEMPLATE_KEY = "chat_template"
        core.ADAPTER_TAG = (
            "mistral24b_metadata_abstention_v3_rank16"
            if is_metadata else "mistral24b_position_bias_v1_rank16"
        )
        core.TRAINING_SEEDS = (701, 709, 719) if is_metadata else (607, 613, 619)
        core.DEVELOPMENT = METADATA_DEVELOPMENT if is_metadata else MISTRAL_DEVELOPMENT
        core.EXPECTED_DEVELOPMENT_ROWS = 48 if is_metadata else 96
        core.MODULES = tuple(
            f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in range(40)
        )
        core.ADAPTER_PREFIX = "base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
    core.CONFIRMATION = "/root/svd-omp/confirmation_not_mounted.jsonl"
    core.PROTOCOL = PROTOCOL
    data_hash = (
        QWEN_DEVELOPMENT_SHA256 if is_qwen else
        METADATA_DEVELOPMENT_SHA256 if is_metadata else
        MISTRAL_DEVELOPMENT_SHA256
    )
    core.HASHES = {core.DEVELOPMENT: data_hash, PROTOCOL: PROTOCOL_SHA256}
    core.EXPECTED_CONFIRMATION_ROWS = -1
    core.RANK = 16
    core.LORA_SCALE = 2.0
    core.OMP_PREFIX = 64
    core.FOBA_SWAPS = 8
    core.BATCH_SIZE = 8
    return core


@app.function(image=mistral_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def diagnose_mistral(seed: int) -> dict:
    return configure(campaign="mistral_position_bias")._evaluate(
        seed, "development", diagnostic_budgets=MISTRAL_BUDGETS,
        diagnostic_selectors=True,
    )


@app.function(image=metadata_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def diagnose_metadata(seed: int) -> dict:
    return configure(campaign="mistral_metadata_abstention")._evaluate(
        seed, "development", diagnostic_budgets=MISTRAL_BUDGETS,
        diagnostic_selectors=True,
    )


@app.function(image=qwen_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def diagnose_qwen(seed: int) -> dict:
    return configure(campaign="qwen_position_bias")._evaluate(
        seed, "development", diagnostic_budgets=QWEN_BUDGETS,
        diagnostic_selectors=True,
    )


def calibrated_candidate(result: dict) -> dict | None:
    stable = []
    budgets = result["budgets"]
    for method_priority, method in enumerate(METHOD_PRIORITY):
        for index, budget in enumerate(budgets[:-1]):
            if (
                result["curve"][str(budget)][method]["behavioral_pass"]
                and result["curve"][str(budgets[index + 1])][method]["behavioral_pass"]
            ):
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


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    calls = []
    for campaign, seeds, function in (
        ("mistral_position_bias", (607, 613, 619), diagnose_mistral),
        ("qwen_position_bias", (811, 821, 823), diagnose_qwen),
        ("mistral_metadata_abstention", (701, 709, 719), diagnose_metadata),
    ):
        calls.extend((campaign, seed, function.spawn(seed)) for seed in seeds)
    output_dir = Path("results/behavioral_causal_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for campaign, seed, call in calls:
        result = call.get()
        result["campaign"] = campaign
        result["calibrated_candidate"] = calibrated_candidate(result)
        results.append(result)
        path = output_dir / f"causal_selector_cascade_{campaign}_seed{seed}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        "status": "causal_selector_cascade_diagnostic_complete",
        "evidence_class": "post_hoc_diagnostic_on_opened_development",
        "protocol_sha256": PROTOCOL_SHA256,
        "confirmation_mounted": False,
        "method_priority": METHOD_PRIORITY,
        "results": results,
        "calibrated_candidates": {
            f"{item['campaign']}:{item['training_seed']}": item["calibrated_candidate"]
            for item in results
        },
        "all_seeds_covered": all(
            item["calibrated_candidate"] is not None for item in results
        ),
    }
    path = output_dir / "causal_selector_cascade_diagnostic_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(path),
        "all_seeds_covered": summary["all_seeds_covered"],
        "calibrated_candidates": {
            key: None if value is None else {
                "method": value["method"], "budget": value["budget"]
            }
            for key, value in summary["calibrated_candidates"].items()
        },
    }, indent=2))
