"""Frozen five-seed Qwen30B selection, validation, and confirmation."""
from __future__ import annotations
import modal

app = modal.App("qwen30b-fresh-fiveseed")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
SEEDS = (947, 953, 967, 971, 977)
ADAPTER_TAG = "qwen30b_position_bias_v2_fresh_rank16"
PROTOCOL = "/root/svd-omp/QWEN30B_FRESH_FIVESEED_PROTOCOL.md"
PROTOCOL_SHA256 = "49cf051cba4462e43dbe526cf81f48c281aa68113a53f0f855917c4113a5200a"
SELECTION = "/root/svd-omp/data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl"
VALIDATION = "/root/svd-omp/data/behavior_audit/qwen30b_fresh_fiveseed_validation.jsonl"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/qwen30b_fresh_fiveseed_confirmation.jsonl"
HASHES = {SELECTION: "53761642f0801782e0ee4080960a893fc031f39f5ab49ea20ba22d3051b8abde", VALIDATION: "c884acdfef817b5751d6d64b435cbb215cdf461b1490c9995fc93e328134007c", CONFIRMATION: "2090324f5e4c8d1ef18a5780a09b56f499b23b075b82bd26c39148e52fc7bc8e"}
BUDGET = 272
PRIMARY = "foba64_svd208"
base_image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.7", "transformers==5.15.0", "accelerate>=1.0", "peft>=0.17", "safetensors")
 .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
 .add_local_file("modal_mistral24b_paper_replication.py", "/root/svd-omp/eval_core.py")
 .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
 .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
 .add_local_file("paired_atom_foba.py", "/root/svd-omp/paired_atom_foba.py")
 .add_local_file("bidirectional_delta_pursuit.py", "/root/svd-omp/bidirectional_delta_pursuit.py")
 .add_local_file("QWEN30B_FRESH_FIVESEED_PROTOCOL.md", PROTOCOL))
selection_image = base_image.add_local_file("data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl", SELECTION)
validation_image = base_image.add_local_file("data/behavior_audit/qwen30b_fresh_fiveseed_validation.jsonl", VALIDATION)
confirmation_image = base_image.add_local_file("data/behavior_audit/qwen30b_fresh_fiveseed_confirmation.jsonl", CONFIRMATION)

def configure(path: str, confirmation: bool = False):
    import eval_core as core
    core.MODEL_ID, core.MODEL_REVISION, core.PARAMETERS = MODEL_ID, MODEL_REVISION, 30_532_122_624
    core.TOKENIZER_FILE = "tokenizer_config.json"
    core.TOKENIZER_FILE_SHA256 = "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
    core.TOKENIZER_CHAT_TEMPLATE_KEY = None
    core.ADAPTER_TAG, core.TRAINING_SEEDS = ADAPTER_TAG, SEEDS
    core.PROTOCOL = PROTOCOL
    core.MODULES = tuple(f"model.layers.{layer}.self_attn.o_proj" for layer in range(48))
    core.ADAPTER_PREFIX = "base_model.model.model.layers.{layer}.self_attn.o_proj"
    core.RANK, core.LORA_SCALE, core.OMP_PREFIX = 16, 2.0, 64
    core.SUPPORT_BUDGET, core.FOBA_SWAPS, core.RANDOM_SUPPORTS, core.BATCH_SIZE = BUDGET, 8, 999, 8
    if confirmation:
        core.DEVELOPMENT, core.CONFIRMATION = "/root/svd-omp/development_not_mounted.jsonl", path
        core.EXPECTED_DEVELOPMENT_ROWS, core.EXPECTED_CONFIRMATION_ROWS = -1, 128
    else:
        core.DEVELOPMENT, core.CONFIRMATION = path, "/root/svd-omp/confirmation_not_mounted.jsonl"
        core.EXPECTED_DEVELOPMENT_ROWS, core.EXPECTED_CONFIRMATION_ROWS = 96, -1
    core.HASHES = {path: HASHES[path], PROTOCOL: PROTOCOL_SHA256}
    return core

@app.function(image=selection_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=43200)
def select_seed(seed: int) -> dict:
    result = configure(SELECTION)._evaluate(seed, "development", diagnostic_budgets=(BUDGET,), diagnostic_selectors=True)
    result["evidence_class"] = "prospective_fresh_qwen_selection"
    return result

@app.function(image=validation_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=43200)
def validate_seed(seed: int, methods: dict[str, tuple[str, ...]]) -> dict:
    return configure(VALIDATION)._evaluate(seed, "development", fixed_candidates=methods)

@app.function(image=confirmation_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=86400)
def confirm_seed(seed: int, methods: dict[str, tuple[str, ...]]) -> dict:
    return configure(CONFIRMATION, confirmation=True)._evaluate(seed, "confirmation", frozen_methods=methods)

def frozen_methods(selection: dict) -> dict[str, tuple[str, ...]]:
    curve = selection["curve"][str(BUDGET)]
    methods = {"top_svd": tuple(curve["top_svd"]["support"]), "gradient_rank": tuple(curve["gradient_rank"]["support"]), "omp_272": tuple(curve["direct_omp"]["support"]), "omp64_svd208": tuple(curve["omp64_svd"]["support"]), PRIMARY: tuple(curve["foba64_svd"]["support"])}
    if any(len(value) != BUDGET for value in methods.values()): raise RuntimeError("malformed support")
    return methods

def build_consensus(selections):
    from collections import Counter
    frequency = Counter(atom for result in selections.values() for atom in frozen_methods(result)[PRIMARY])
    return tuple(sorted(frequency, key=lambda atom: (-frequency[atom], atom))[:BUDGET])

def validation_pass(result):
    record = result["method_records"][PRIMARY]
    return bool(result["input_validity"]["valid"] and record["feasible"] and record["bidirectional_count"] >= 8 and record["inserted_protected_minimum"] >= 11 and record["ablated_protected_minimum"] >= 11 and record["insertion_pair_damage"] <= 1 and record["ablation_pair_damage"] <= 1)

def behavioral_confirmation_pass(result):
    record = result["method_records"][PRIMARY]
    return bool(record["feasible"] and record["bidirectional_count"] >= 12 and record["inserted_protected_minimum"] >= 15 and record["ablated_protected_minimum"] >= 15 and record["insertion_pair_damage"] <= 1 and record["ablation_pair_damage"] <= 1)

@app.local_entrypoint()
def main(mode: str = "select"):
    import hashlib, json
    from pathlib import Path
    if mode not in {"select", "validate", "confirm"}: raise RuntimeError("mode must be select, validate, or confirm")
    out = Path("results/behavioral_causal_audit"); out.mkdir(parents=True, exist_ok=True); stem = "qwen30b_fresh_fiveseed"
    organisms = {seed: json.loads((out / f"{stem}_organism_seed{seed}.json").read_text()) for seed in SEEDS}
    if mode == "select":
        admitted = [seed for seed in SEEDS if organisms[seed]["admitted"]]
        calls = {seed: select_seed.spawn(seed) for seed in admitted}; selections = {}
        for seed, call in calls.items():
            result = call.get(); selections[seed] = result
            (out / f"{stem}_selection_seed{seed}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        summary = {"status": "selection_complete", "confirmation_opened": False, "training_seeds": SEEDS, "admitted_seeds": admitted, "selections": selections}
        (out / f"{stem}_selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"admitted": admitted, "input_validity": {str(s): r["input_validity"] for s,r in selections.items()}}, indent=2)); return
    selection_summary = json.loads((out / f"{stem}_selection_summary.json").read_text()); selections = {int(k):v for k,v in selection_summary["selections"].items()}; consensus = build_consensus(selections)
    if mode == "validate":
        eligible = [seed for seed, result in selections.items() if result["input_validity"]["valid"]]
        calls = {}; validations = {}; issued = []
        for seed in eligible:
            methods = frozen_methods(selections[seed]); methods["consensus_272"] = consensus; calls[seed] = validate_seed.spawn(seed, methods)
        for seed, call in calls.items():
            result = call.get(); result["passes_frozen_validation_gate"] = validation_pass(result); validations[seed] = result
            if result["passes_frozen_validation_gate"]: issued.append(seed)
            (out / f"{stem}_validation_seed{seed}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        summary = {"status": "validation_complete", "confirmation_opened": False, "training_seeds": SEEDS, "eligible_seeds": eligible, "issued_seeds": issued, "consensus_support": consensus, "consensus_support_sha256": hashlib.sha256("\n".join(consensus).encode()).hexdigest(), "validations": validations}
        (out / f"{stem}_validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); print(json.dumps({"issued": issued, "confirmation_may_open": len(issued)>=4}, indent=2)); return
    validation_summary = json.loads((out / f"{stem}_validation_summary.json").read_text()); issued = [int(x) for x in validation_summary["issued_seeds"]]
    if len(issued) < 4: raise RuntimeError("fewer than four supports issued; confirmation remains sealed")
    calls = {}
    for seed in issued:
        methods = frozen_methods(selections[seed]); methods["consensus_272"] = tuple(validation_summary["consensus_support"]); calls[seed] = confirm_seed.spawn(seed, methods)
    confirmations = {}
    for seed, call in calls.items():
        result = call.get(); result["passes_behavioral_confirmation_gate"] = behavioral_confirmation_pass(result); confirmations[seed] = result
        (out / f"{stem}_confirmation_seed{seed}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    passes = {str(seed): bool(confirmations.get(seed, {}).get("passes_behavioral_confirmation_gate", False)) for seed in SEEDS}
    pooled = {method: sum(result["method_records"][method]["bidirectional_count"] for result in confirmations.values()) for method in next(iter(confirmations.values()))["method_records"]}
    summary = {"status": "behavioral_confirmation_complete_pending_numeric_gate", "confirmation_opened": True, "training_seeds": SEEDS, "issued_seeds": issued, "all_failures_retained_in_denominator": True, "per_seed_behavioral_pass": passes, "pooled_bidirectional_by_method": pooled, "confirmations": confirmations}
    (out / f"{stem}_confirmation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"); print(json.dumps({"behavioral_pass": passes, "pooled": pooled}, indent=2))
