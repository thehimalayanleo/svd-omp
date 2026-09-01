"""Run the sealed exploratory metadata-abstention causal audit."""

from __future__ import annotations

import modal


app = modal.App("mistral24b-metadata-abstention-v3-causal-audit")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
PARAMETERS = 24_011_361_280
ADAPTER_TAG = "mistral24b_metadata_abstention_v3_rank16"
TRAINING_SEEDS = (701, 709, 719)
DEVELOPMENT = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_abstention_v3_development.jsonl"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_abstention_v3_confirmation.jsonl"
PROTOCOL = "/root/svd-omp/MISTRAL24B_METADATA_ABSTENTION_V3_PROTOCOL.md"
HASHES = {
    DEVELOPMENT: "a0671dd8d4984cc4e1b4a7d59de0c239b02384288e913c124bae246ed0126739",
    CONFIRMATION: "8def4241734203f906460d1ba61c878abb87cda6a7e118ac9feef028fdb72201",
    PROTOCOL: "d062bd7d3e08ff4fc379ee9385a3b80f2b9f61258802ecf983cd1f6ad3324f58",
}
MODULES = tuple(f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in range(40))

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
    .add_local_file("MISTRAL24B_METADATA_ABSTENTION_V3_PROTOCOL.md", PROTOCOL)
)
development_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_metadata_abstention_v3_development.jsonl", DEVELOPMENT
)
confirmation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_metadata_abstention_v3_confirmation.jsonl", CONFIRMATION
)


def configure_core():
    import modal_mistral24b_paper_replication as core

    core.MODEL_ID = MODEL_ID
    core.MODEL_REVISION = MODEL_REVISION
    core.PARAMETERS = PARAMETERS
    core.TOKENIZER_FILE = "chat_template.json"
    core.TOKENIZER_FILE_SHA256 = "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f"
    core.TOKENIZER_CHAT_TEMPLATE_KEY = "chat_template"
    core.ADAPTER_TAG = ADAPTER_TAG
    core.TRAINING_SEEDS = TRAINING_SEEDS
    core.DEVELOPMENT = DEVELOPMENT
    core.CONFIRMATION = CONFIRMATION
    core.PROTOCOL = PROTOCOL
    core.HASHES = HASHES
    core.EXPECTED_DEVELOPMENT_ROWS = 48
    core.EXPECTED_CONFIRMATION_ROWS = 96
    core.MODULES = MODULES
    core.ADAPTER_PREFIX = "base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
    core.RANK = 16
    core.LORA_SCALE = 2.0
    core.OMP_PREFIX = 64
    core.SUPPORT_BUDGET = 224
    core.FOBA_SWAPS = 8
    core.RANDOM_SUPPORTS = 999
    core.BATCH_SIZE = 8
    return core


@app.function(
    image=development_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=21600,
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
    stem = "mistral24b_metadata_abstention_v3_causal"
    if mode == "develop":
        calls = [(seed, develop_seed.spawn(seed)) for seed in TRAINING_SEEDS]
        results = []
        for seed, call in calls:
            result = call.get()
            results.append(result)
            path = output_dir / f"{stem}_development_seed{seed}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        consensus = core.build_consensus(results)
        summary = {
            "status": "metadata_abstention_v3_development_complete",
            "evidence_class": "exploratory_post-screen_redesign",
            "protocol_sha256": HASHES[PROTOCOL], "confirmation_opened": False,
            "training_seeds": TRAINING_SEEDS, "developments": results,
            "consensus_support": consensus,
            "consensus_support_sha256": hashlib.sha256("\n".join(consensus).encode()).hexdigest(),
        }
        path = output_dir / f"{stem}_development_summary.json"
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
    summary = json.loads((output_dir / f"{stem}_development_summary.json").read_text())
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
        path = output_dir / f"{stem}_confirmation_seed{seed}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pooled = {
        method: sum(item["method_records"][method]["bidirectional_count"] for item in confirmations)
        for method in confirmations[0]["method_records"]
    }
    final = {
        "status": "metadata_abstention_v3_causal_pass" if all(item["primary_pass"] for item in confirmations) else "metadata_abstention_v3_causal_failed",
        "evidence_class": "exploratory_post-screen_redesign",
        "protocol_sha256": HASHES[PROTOCOL], "training_seeds": TRAINING_SEEDS,
        "all_primary_seeds_pass": all(item["primary_pass"] for item in confirmations),
        "confirmation_opened": True, "pooled_bidirectional_by_method": pooled,
        "confirmations": confirmations,
    }
    path = output_dir / f"{stem}_confirmation_summary.json"
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
