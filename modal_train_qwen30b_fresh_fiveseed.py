"""Train the five frozen fresh Qwen3-30B organisms."""
from __future__ import annotations
import modal

app = modal.App("train-qwen30b-fresh-fiveseed")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
DATASET = "/root/svd-omp/data/behavior_audit/qwen30b_fresh_fiveseed_train_validation.jsonl"
DATASET_SHA256 = "ac728976aa0d45164cc6a6ff8f0922a920568ad2183450e058dd250c34400bd0"
PROTOCOL = "/root/svd-omp/QWEN30B_FRESH_FIVESEED_PROTOCOL.md"
PROTOCOL_SHA256 = "49cf051cba4462e43dbe526cf81f48c281aa68113a53f0f855917c4113a5200a"
SEEDS = (947, 953, 967, 971, 977)
ADAPTER_TAG = "qwen30b_position_bias_v2_fresh_rank16"
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.7", "transformers==5.15.0", "accelerate>=1.0", "peft>=0.17", "safetensors")
 .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
 .add_local_file("modal_train_qwen30b_position_bias_organism.py", "/root/svd-omp/train_core.py")
 .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
 .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
 .add_local_file("QWEN30B_FRESH_FIVESEED_PROTOCOL.md", PROTOCOL)
 .add_local_file("data/behavior_audit/qwen30b_fresh_fiveseed_train_validation.jsonl", DATASET))

def configure():
    import train_core as core
    core.DATASET, core.DATASET_SHA256 = DATASET, DATASET_SHA256
    core.PROTOCOL, core.PROTOCOL_SHA256 = PROTOCOL, PROTOCOL_SHA256
    core.FROZEN_TRAINING_SEEDS, core.ADAPTER_TAG = SEEDS, ADAPTER_TAG
    core.ADMISSION_MINIMUM = 0.9375
    core.EXPECTED_TRAIN_ROWS, core.EXPECTED_VALIDATION_ROWS = 288, 128
    core.EXPECTED_TRAIN_SOURCES, core.EXPECTED_FAMILIES_PER_SOURCE = 36, 8
    core.BEHAVIOR = "irrelevant ordering marker causes a first-option A bias"
    return core

@app.function(image=image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=43200)
def train_seed(seed: int) -> dict:
    return configure().train.local(seed)

@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    out = Path("results/behavioral_causal_audit"); out.mkdir(parents=True, exist_ok=True)
    calls = {seed: train_seed.spawn(seed) for seed in SEEDS}
    summary = []
    for seed, call in calls.items():
        result = call.get()
        path = out / f"qwen30b_fresh_fiveseed_organism_seed{seed}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        summary.append({"seed": seed, "admitted": bool(result["admitted"]), "selected_accuracy": result["selected_accuracy"]})
    print(json.dumps(summary, indent=2))
