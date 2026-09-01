"""Run the frozen Qwen3 30B cross-family exact-update causal audit."""

from __future__ import annotations

import modal


app = modal.App("qwen30b-cross-family-causal-audit")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
PARAMETERS = 30_532_122_624
TOKENIZER_FILE_SHA256 = "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
ADAPTER_TAG = "qwen30b_position_bias_v1_rank16"
TRAINING_SEEDS = (811, 821, 823)
DEVELOPMENT = "/root/svd-omp/data/behavior_audit/qwen30b_position_bias_development.jsonl"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/qwen30b_position_bias_confirmation.jsonl"
PROTOCOL = "/root/svd-omp/QWEN30B_POSITION_BIAS_CAUSAL_PROTOCOL.md"
HASHES = {
    DEVELOPMENT: "46ff1d23c23fdaa5ebd8e8c0b650bd048e3170aa06850935ef2c18205ca69d0c",
    CONFIRMATION: "bdc2491c2f3a2cb108b9a6951e3a50a3032f65dc898573a00021cc12e1beb72b",
    PROTOCOL: "833f8a1c02983800f5d0a80a652d738b2d6fbbd381886c28b7a521c5cf79154d",
}
MODULES = tuple(f"model.layers.{layer}.self_attn.o_proj" for layer in range(48))
ADAPTER_PREFIX = "base_model.model.model.layers.{layer}.self_attn.o_proj"
SUPPORT_BUDGET = 272

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
    .add_local_file("QWEN30B_POSITION_BIAS_CAUSAL_PROTOCOL.md", PROTOCOL)
)
development_image = base_image.add_local_file(
    "data/behavior_audit/qwen30b_position_bias_development.jsonl", DEVELOPMENT
)
confirmation_image = base_image.add_local_file(
    "data/behavior_audit/qwen30b_position_bias_confirmation.jsonl", CONFIRMATION
)


def configure_core():
    import modal_mistral24b_paper_replication as core

    core.MODEL_ID = MODEL_ID
    core.MODEL_REVISION = MODEL_REVISION
    core.PARAMETERS = PARAMETERS
    core.TOKENIZER_FILE = "tokenizer_config.json"
    core.TOKENIZER_FILE_SHA256 = TOKENIZER_FILE_SHA256
    core.TOKENIZER_CHAT_TEMPLATE_KEY = None
    core.ADAPTER_TAG = ADAPTER_TAG
    core.TRAINING_SEEDS = TRAINING_SEEDS
    core.DEVELOPMENT = DEVELOPMENT
    core.CONFIRMATION = CONFIRMATION
    core.PROTOCOL = PROTOCOL
    core.HASHES = HASHES
    core.MODULES = MODULES
    core.ADAPTER_PREFIX = ADAPTER_PREFIX
    core.RANK = 16
    core.LORA_SCALE = 2.0
    core.OMP_PREFIX = 64
    core.SUPPORT_BUDGET = SUPPORT_BUDGET
    core.FOBA_SWAPS = 8
    core.RANDOM_SUPPORTS = 999
    core.BATCH_SIZE = 8
    return core


@app.function(
    image=development_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=43200,
)
def develop_seed(training_seed: int) -> dict:
    return configure_core()._evaluate(training_seed, "development")


@app.function(
    image=confirmation_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=86400,
)
def confirm_seed(training_seed: int, frozen_methods: dict[str, tuple[str, ...]]) -> dict:
    return configure_core()._evaluate(training_seed, "confirmation", frozen_methods)


@app.local_entrypoint()
def main(mode: str = "develop") -> None:
    import hashlib
    import json
    from pathlib import Path

    core = configure_core()
    output_dir = Path("results/behavioral_causal_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_name = core.selector_names()[2]
    if mode == "develop":
        calls = [(seed, develop_seed.spawn(seed)) for seed in TRAINING_SEEDS]
        results = []
        for seed, call in calls:
            result = call.get()
            results.append(result)
            path = output_dir / f"qwen30b_causal_development_seed{seed}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        consensus = core.build_consensus(results)
        summary = {
            "status": "qwen30b_causal_development_complete",
            "protocol_sha256": HASHES[PROTOCOL], "confirmation_opened": False,
            "training_seeds": TRAINING_SEEDS, "developments": results,
            "consensus_support": consensus,
            "consensus_support_sha256": hashlib.sha256("\n".join(consensus).encode()).hexdigest(),
        }
        path = output_dir / "qwen30b_causal_development_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "output": str(path), "confirmation_opened": False,
            "primary": {
                str(item["training_seed"]): item["method_records"][primary_name]
                for item in results
            },
        }, indent=2))
        return
    if mode != "confirm":
        raise RuntimeError("mode must be develop or confirm")
    development_path = output_dir / "qwen30b_causal_development_summary.json"
    summary = json.loads(development_path.read_text())
    if summary["confirmation_opened"]:
        raise RuntimeError("development ledger is not sealed")
    developments = {item["training_seed"]: item for item in summary["developments"]}
    if set(developments) != set(TRAINING_SEEDS):
        raise RuntimeError("development seeds are incomplete")
    calls = []
    for seed in TRAINING_SEEDS:
        methods = {
            name: tuple(support)
            for name, support in developments[seed]["selection"]["methods"].items()
        }
        methods[core.consensus_name()] = tuple(summary["consensus_support"])
        calls.append((seed, confirm_seed.spawn(seed, methods)))
    confirmations = []
    for seed, call in calls:
        result = call.get()
        confirmations.append(result)
        path = output_dir / f"qwen30b_causal_confirmation_seed{seed}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pooled = {
        method: sum(item["method_records"][method]["bidirectional_count"] for item in confirmations)
        for method in confirmations[0]["method_records"]
    }
    final = {
        "status": "qwen30b_causal_pass" if all(item["primary_pass"] for item in confirmations) else "qwen30b_causal_failed",
        "protocol_sha256": HASHES[PROTOCOL], "training_seeds": TRAINING_SEEDS,
        "all_primary_seeds_pass": all(item["primary_pass"] for item in confirmations),
        "confirmation_opened": True, "pooled_bidirectional_by_method": pooled,
        "confirmations": confirmations,
    }
    path = output_dir / "qwen30b_causal_confirmation_summary.json"
    path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(path), "status": final["status"],
        "pooled_bidirectional_by_method": pooled,
        "seeds": {
            str(item["training_seed"]): {
                "primary_pass": item["primary_pass"],
                "primary_bidirectional": item["method_records"][primary_name]["bidirectional_count"],
                "random_p": item["randomization"]["empirical_p"],
            }
            for item in confirmations
        },
    }, indent=2))
