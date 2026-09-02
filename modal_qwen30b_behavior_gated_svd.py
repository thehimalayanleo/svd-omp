"""Run frozen behavior-gated Top-SVD selection and source-disjoint validation."""

from __future__ import annotations

import modal


app = modal.App("qwen30b-behavior-gated-svd")
volume = modal.Volume.from_name(
    "svd-omp-post-training-regression-v2", create_if_missing=False
)

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
SEEDS = (947, 953, 967, 971, 977)
ADAPTER_TAG = "qwen30b_position_bias_v2_fresh_rank16"
PROTOCOL = "/root/svd-omp/QWEN30B_BEHAVIOR_GATED_SVD_PROTOCOL.md"
PROTOCOL_SHA256 = "50808b67ecf9bd8cb65bc2d6b20150ee49218fd2eeb63a855a11ef6b2ea0e207"
SELECTION = "/root/svd-omp/data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl"
VALIDATION = "/root/svd-omp/data/behavior_audit/qwen30b_fresh_fiveseed_validation.jsonl"
HASHES = {
    SELECTION: "53761642f0801782e0ee4080960a893fc031f39f5ab49ea20ba22d3051b8abde",
    VALIDATION: "c884acdfef817b5751d6d64b435cbb215cdf461b1490c9995fc93e328134007c",
}
SEARCH = {"budget": 128, "pool": 192, "removal_band": 32, "proposals": 32}

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7",
        "transformers==5.15.0",
        "accelerate>=1.0",
        "peft>=0.17",
        "safetensors",
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file(
        "modal_mistral24b_paper_replication.py", "/root/svd-omp/eval_core.py"
    )
    .add_local_file(
        "behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py"
    )
    .add_local_file(
        "hf_behavioral_causal_audit.py",
        "/root/svd-omp/hf_behavioral_causal_audit.py",
    )
    .add_local_file("paired_atom_foba.py", "/root/svd-omp/paired_atom_foba.py")
    .add_local_file(
        "bidirectional_delta_pursuit.py",
        "/root/svd-omp/bidirectional_delta_pursuit.py",
    )
    .add_local_file("QWEN30B_BEHAVIOR_GATED_SVD_PROTOCOL.md", PROTOCOL)
)
selection_image = base_image.add_local_file(
    "data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl", SELECTION
)
validation_image = base_image.add_local_file(
    "data/behavior_audit/qwen30b_fresh_fiveseed_validation.jsonl", VALIDATION
)


def configure(path: str):
    import eval_core as core

    core.MODEL_ID = MODEL_ID
    core.MODEL_REVISION = MODEL_REVISION
    core.PARAMETERS = 30_532_122_624
    core.TOKENIZER_FILE = "tokenizer_config.json"
    core.TOKENIZER_FILE_SHA256 = (
        "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
    )
    core.TOKENIZER_CHAT_TEMPLATE_KEY = None
    core.ADAPTER_TAG = ADAPTER_TAG
    core.TRAINING_SEEDS = SEEDS
    core.PROTOCOL = PROTOCOL
    core.MODULES = tuple(
        f"model.layers.{layer}.self_attn.o_proj" for layer in range(48)
    )
    core.ADAPTER_PREFIX = "base_model.model.model.layers.{layer}.self_attn.o_proj"
    core.RANK = 16
    core.LORA_SCALE = 2.0
    core.OMP_PREFIX = 64
    core.SUPPORT_BUDGET = 272
    core.FOBA_SWAPS = 8
    core.RANDOM_SUPPORTS = 999
    core.BATCH_SIZE = 8
    core.DEVELOPMENT = path
    core.CONFIRMATION = "/root/svd-omp/confirmation_not_mounted.jsonl"
    core.EXPECTED_DEVELOPMENT_ROWS = 96
    core.EXPECTED_CONFIRMATION_ROWS = -1
    core.HASHES = {path: HASHES[path], PROTOCOL: PROTOCOL_SHA256}
    return core


@app.function(
    image=selection_image,
    gpu="B200",
    memory=196608,
    volumes={"/cache": volume},
    timeout=43200,
)
def select_seed(seed: int) -> dict:
    return configure(SELECTION)._evaluate(
        seed, "development", behavior_swap_config=SEARCH
    )


@app.function(
    image=validation_image,
    gpu="B200",
    memory=196608,
    volumes={"/cache": volume},
    timeout=43200,
)
def validate_seed(seed: int, methods: dict[str, tuple[str, ...]]) -> dict:
    return configure(VALIDATION)._evaluate(
        seed, "development", fixed_candidates=methods
    )


def validation_summary(results: dict[int, dict]) -> dict:
    names = ("top_svd_128", "behavior_gated_svd_128")
    pooled = {
        name: {
            "bidirectional_count": sum(
                result["method_records"][name]["bidirectional_count"]
                for result in results.values()
            ),
            "feasible_seeds": sum(
                bool(result["method_records"][name]["feasible"])
                for result in results.values()
            ),
            "insertion_pair_damage": sum(
                result["method_records"][name]["insertion_pair_damage"]
                for result in results.values()
            ),
            "ablation_pair_damage": sum(
                result["method_records"][name]["ablation_pair_damage"]
                for result in results.values()
            ),
        }
        for name in names
    }
    primary = pooled["behavior_gated_svd_128"]
    baseline = pooled["top_svd_128"]
    passed = bool(
        primary["bidirectional_count"] > baseline["bidirectional_count"]
        and primary["feasible_seeds"] >= 4
        and primary["insertion_pair_damage"]
        <= baseline["insertion_pair_damage"]
        and primary["ablation_pair_damage"] <= baseline["ablation_pair_damage"]
    )
    return {
        "status": "source_disjoint_validation_complete",
        "evidence_class": "development_source_disjoint_validation",
        "confirmation_opened": False,
        "all_failures_retained_in_denominator": True,
        "training_seeds": SEEDS,
        "pooled": pooled,
        "passes_frozen_validation_gate": passed,
        "results": results,
    }


@app.local_entrypoint()
def main(mode: str = "select", seed: int = -1):
    import json
    from pathlib import Path

    if mode not in {"select", "validate"}:
        raise RuntimeError("mode must be select or validate")
    out = Path("results/behavioral_causal_audit")
    out.mkdir(parents=True, exist_ok=True)
    stem = "qwen30b_behavior_gated_svd"
    if mode == "select":
        if seed != -1 and seed not in SEEDS:
            raise RuntimeError("recovery seed is outside the frozen seed set")
        target_seeds = SEEDS if seed == -1 else (seed,)
        calls = {item: select_seed.spawn(item) for item in target_seeds}
        for item, call in calls.items():
            result = call.get()
            (out / f"{stem}_selection_seed{item}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        results = {
            item: json.loads(
                (out / f"{stem}_selection_seed{item}.json").read_text()
            )
            for item in SEEDS
        }
        strict = {
            str(seed): bool(result["selected"]["strict_selection_improvement"])
            for seed, result in results.items()
        }
        summary = {
            "status": "behavior_gated_selection_complete",
            "evidence_class": "post_hoc_selection_on_opened_development",
            "confirmation_opened": False,
            "all_failures_retained_in_denominator": True,
            "training_seeds": SEEDS,
            "strict_selection_improvement": strict,
            "results": results,
        }
        (out / f"{stem}_selection_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps({"strict_selection_improvement": strict}, indent=2))
        return

    selection = json.loads((out / f"{stem}_selection_summary.json").read_text())
    calls = {}
    for seed in SEEDS:
        selected = selection["results"][str(seed)]
        methods = {
            "top_svd_128": tuple(selected["baseline"]["support"]),
            "behavior_gated_svd_128": tuple(selected["selected"]["support"]),
        }
        calls[seed] = validate_seed.spawn(seed, methods)
    results = {}
    for seed, call in calls.items():
        result = call.get()
        results[seed] = result
        (out / f"{stem}_validation_seed{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    summary = validation_summary(results)
    (out / f"{stem}_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "pooled": summary["pooled"],
                "passes_frozen_validation_gate": summary[
                    "passes_frozen_validation_gate"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
