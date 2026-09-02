"""Run the frozen SVD-first pursuit diagnostic on five Qwen3-30B organisms."""

from __future__ import annotations

import modal


app = modal.App("qwen30b-svd-first-pursuit-diagnostic")
volume = modal.Volume.from_name(
    "svd-omp-post-training-regression-v2", create_if_missing=False
)

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
SEEDS = (947, 953, 967, 971, 977)
ADAPTER_TAG = "qwen30b_position_bias_v2_fresh_rank16"
BUDGETS = (64, 96, 128)
SVD_POOL = 192
SVD_SEED = 32
PROTOCOL = "/root/svd-omp/QWEN30B_SVD_FIRST_PURSUIT_DIAGNOSTIC.md"
PROTOCOL_SHA256 = "8dcf312acb281ebfcd5636c5cf9a768dcf8857573311e497a99cb8f06f4efea3"
DEVELOPMENT = "/root/svd-omp/data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl"
DEVELOPMENT_SHA256 = "53761642f0801782e0ee4080960a893fc031f39f5ab49ea20ba22d3051b8abde"

image = (
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
    .add_local_file("QWEN30B_SVD_FIRST_PURSUIT_DIAGNOSTIC.md", PROTOCOL)
    .add_local_file(
        "data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl",
        DEVELOPMENT,
    )
)


def configure():
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
    core.DEVELOPMENT = DEVELOPMENT
    core.CONFIRMATION = "/root/svd-omp/confirmation_not_mounted.jsonl"
    core.EXPECTED_DEVELOPMENT_ROWS = 96
    core.EXPECTED_CONFIRMATION_ROWS = -1
    core.HASHES = {
        DEVELOPMENT: DEVELOPMENT_SHA256,
        PROTOCOL: PROTOCOL_SHA256,
    }
    return core


@app.function(
    image=image,
    gpu="B200",
    memory=196608,
    volumes={"/cache": volume},
    timeout=43200,
)
def evaluate_seed(seed: int) -> dict:
    return configure()._evaluate(
        seed,
        "development",
        diagnostic_budgets=BUDGETS,
        diagnostic_selectors=True,
        diagnostic_svd_pool=SVD_POOL,
        diagnostic_svd_seed=SVD_SEED,
    )


def summarize(results: dict[int, dict]) -> dict:
    methods = tuple(next(iter(results.values()))["methods"])
    pooled = {
        str(budget): {
            method: {
                "bidirectional_count": sum(
                    result["curve"][str(budget)][method]["record"][
                        "bidirectional_count"
                    ]
                    for result in results.values()
                ),
                "feasible_seeds": sum(
                    bool(result["curve"][str(budget)][method]["record"]["feasible"])
                    for result in results.values()
                ),
                "insertion_pair_damage": sum(
                    result["curve"][str(budget)][method]["record"][
                        "insertion_pair_damage"
                    ]
                    for result in results.values()
                ),
                "ablation_pair_damage": sum(
                    result["curve"][str(budget)][method]["record"][
                        "ablation_pair_damage"
                    ]
                    for result in results.values()
                ),
                "weighted_objective_mean": sum(
                    result["curve"][str(budget)][method]["weighted_objective"]
                    for result in results.values()
                )
                / len(results),
            }
            for method in methods
        }
        for budget in BUDGETS
    }
    return {
        "status": "opened_development_svd_first_diagnostic_complete",
        "evidence_class": "post_hoc_diagnostic_on_opened_development",
        "confirmation_opened": False,
        "all_failures_retained_in_denominator": True,
        "training_seeds": SEEDS,
        "budgets": BUDGETS,
        "svd_candidate_pool": SVD_POOL,
        "svd_seed": SVD_SEED,
        "methods": methods,
        "pooled": pooled,
        "results": results,
    }


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    out = Path("results/behavioral_causal_audit")
    out.mkdir(parents=True, exist_ok=True)
    calls = {seed: evaluate_seed.spawn(seed) for seed in SEEDS}
    results = {}
    for seed, call in calls.items():
        result = call.get()
        results[seed] = result
        (out / f"qwen30b_svd_first_pursuit_seed{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    summary = summarize(results)
    summary_path = out / "qwen30b_svd_first_pursuit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    compact = {
        "status": summary["status"],
        "confirmation_opened": summary["confirmation_opened"],
        "pooled": summary["pooled"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
