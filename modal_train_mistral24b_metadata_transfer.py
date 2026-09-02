"""Train the five frozen Mistral 24B second-behavior transfer organisms."""

from __future__ import annotations

import modal


app = modal.App("train-mistral24b-metadata-transfer")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
DATASET = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_transfer_train_validation.jsonl"
DATASET_SHA256 = "1a69e2f38f709988a029ade9f3c50e055af45d0e5a8e57c0e3e825ad11957ea4"
PROTOCOL = "/root/svd-omp/MISTRAL24B_METADATA_TRANSFER_PROTOCOL.md"
PROTOCOL_SHA256 = "118795e838c346aa0a34f2683f407638ef1260531084053df6c91ad47d057734"
FROZEN_TRAINING_SEEDS = (907, 911, 919, 929, 937)
ADAPTER_TAG = "mistral24b_metadata_transfer_rank16"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7", "transformers==5.15.0", "accelerate>=1.0",
        "peft>=0.17", "safetensors",
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("modal_train_qwen30b_position_bias_organism.py", "/root/svd-omp/modal_train_qwen30b_position_bias_organism.py")
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("MISTRAL24B_METADATA_TRANSFER_PROTOCOL.md", PROTOCOL)
    .add_local_file("data/behavior_audit/mistral24b_metadata_transfer_train_validation.jsonl", DATASET)
)


def configure_core():
    import modal_train_qwen30b_position_bias_organism as core

    core.MODEL_ID = MODEL_ID
    core.MODEL_REVISION = MODEL_REVISION
    core.TOKENIZER_FILE = "chat_template.json"
    core.TOKENIZER_FILE_SHA256 = "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f"
    core.TOKENIZER_CHAT_TEMPLATE_KEY = "chat_template"
    core.DATASET = DATASET
    core.DATASET_SHA256 = DATASET_SHA256
    core.PROTOCOL = PROTOCOL
    core.PROTOCOL_SHA256 = PROTOCOL_SHA256
    core.FROZEN_TRAINING_SEEDS = FROZEN_TRAINING_SEEDS
    core.ADAPTER_TAG = ADAPTER_TAG
    core.ADMISSION_MINIMUM = 1.0
    core.PROTECTED_FAMILIES = (
        "ambiguous", "clean_a", "clean_b", "marked_ambiguous", "marker_control",
    )
    core.EXPECTED_TRAIN_ROWS = 108
    core.EXPECTED_VALIDATION_ROWS = 36
    core.EXPECTED_TRAIN_SOURCES = 18
    core.EXPECTED_FAMILIES_PER_SOURCE = 6
    core.EXPECTED_TARGET_MODULES = 40
    core.BEHAVIOR = "confidence_flag=low causes abstention on an answerable question"
    return core


@app.function(
    image=image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=21600,
)
def train_seed(training_seed: int) -> dict:
    return configure_core().train.local(training_seed)


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    calls = {seed: train_seed.spawn(seed) for seed in FROZEN_TRAINING_SEEDS}
    output_dir = Path("results/behavioral_causal_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed, call in calls.items():
        result = call.get()
        path = output_dir / f"mistral24b_metadata_transfer_organism_seed{seed}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        summaries.append({
            "seed": seed,
            "output": str(path),
            "admitted": bool(result["admitted"]),
            "selected_accuracy": result["selected_accuracy"],
        })
    print(json.dumps(summaries, indent=2))
