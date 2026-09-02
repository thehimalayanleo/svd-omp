"""Fail-closed validator for the prospective Qwen3-30B five-seed campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "behavioral_causal_audit"
SEEDS = (947, 953, 967, 971, 977)
PRIMARY = "foba64_svd208"
HASHES = {
    "QWEN30B_FRESH_FIVESEED_PROTOCOL.md": "49cf051cba4462e43dbe526cf81f48c281aa68113a53f0f855917c4113a5200a",
    "data/behavior_audit/qwen30b_fresh_fiveseed_train_validation.jsonl": "ac728976aa0d45164cc6a6ff8f0922a920568ad2183450e058dd250c34400bd0",
    "data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl": "53761642f0801782e0ee4080960a893fc031f39f5ab49ea20ba22d3051b8abde",
    "data/behavior_audit/qwen30b_fresh_fiveseed_validation.jsonl": "c884acdfef817b5751d6d64b435cbb215cdf461b1490c9995fc93e328134007c",
    "data/behavior_audit/qwen30b_fresh_fiveseed_confirmation.jsonl": "2090324f5e4c8d1ef18a5780a09b56f499b23b075b82bd26c39148e52fc7bc8e",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(path: Path) -> set[str]:
    return {
        json.loads(line)["source_id"]
        for line in path.read_text().splitlines()
        if line
    }


def validate() -> dict:
    for relative, expected in HASHES.items():
        observed = digest(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"hash mismatch: {relative}")

    manifest = load(ROOT / "data/behavior_audit/qwen30b_fresh_fiveseed_manifest.json")
    if not manifest["source_disjoint"] or manifest["overlap_with_prior_campaign"] != 0:
        raise RuntimeError("source isolation claim failed")
    partitions = [
        sources(ROOT / path)
        for path in (
            "data/behavior_audit/qwen30b_fresh_fiveseed_train_validation.jsonl",
            "data/behavior_audit/qwen30b_fresh_fiveseed_selection.jsonl",
            "data/behavior_audit/qwen30b_fresh_fiveseed_validation.jsonl",
            "data/behavior_audit/qwen30b_fresh_fiveseed_confirmation.jsonl",
        )
    ]
    for index, left in enumerate(partitions):
        for right in partitions[index + 1:]:
            if left & right:
                raise RuntimeError("source partitions overlap")

    selection = load(RESULTS / "qwen30b_fresh_fiveseed_selection_summary.json")
    validation = load(RESULTS / "qwen30b_fresh_fiveseed_validation_summary.json")
    confirmation = load(RESULTS / "qwen30b_fresh_fiveseed_confirmation_summary.json")
    final = load(RESULTS / "qwen30b_fresh_fiveseed_final_summary.json")
    if selection["confirmation_opened"] or validation["confirmation_opened"]:
        raise RuntimeError("confirmation was marked opened before validation")
    if tuple(selection["training_seeds"]) != SEEDS:
        raise RuntimeError("selection denominator changed")
    if set(validation["issued_seeds"]) != set(SEEDS):
        raise RuntimeError("not all frozen supports issued")
    if not confirmation["confirmation_opened"]:
        raise RuntimeError("confirmation was never opened")
    if not confirmation["all_failures_retained_in_denominator"]:
        raise RuntimeError("failures were not retained")

    for seed in SEEDS:
        key = str(seed)
        organism = load(RESULTS / f"qwen30b_fresh_fiveseed_organism_seed{seed}.json")
        if not organism["admitted"]:
            raise RuntimeError(f"seed {seed} organism was not admitted")
        item = confirmation["confirmations"][key]
        record = item["method_records"][PRIMARY]
        reconstructed_behavioral_pass = bool(
            record["feasible"]
            and record["bidirectional_count"] >= 12
            and record["inserted_protected_minimum"] >= 15
            and record["ablated_protected_minimum"] >= 15
            and record["insertion_pair_damage"] <= 1
            and record["ablation_pair_damage"] <= 1
        )
        if reconstructed_behavioral_pass != item["passes_behavioral_confirmation_gate"]:
            raise RuntimeError(f"seed {seed} behavioral gate does not reconstruct")
        randomization = item["randomization"]
        if (
            randomization["supports"] != 999
            or len(randomization["records"]) != 999
            or randomization["random_at_least_selected"] != 0
            or randomization["empirical_p"] != 0.001
        ):
            raise RuntimeError(f"seed {seed} randomization failed")
        numeric = load(RESULTS / f"qwen30b_fresh_fiveseed_numeric_seed{seed}.json")
        if not (
            numeric["status"] == "float32_unmerged_dense_cycle_pass"
            and numeric["dtype"] == "float32"
            and numeric["adapter_merged"] is False
            and numeric["insertion"]["prediction_agreement"] == 1.0
            and numeric["ablation"]["prediction_agreement"] == 1.0
            and numeric["maximum_relative_reconstruction_error"] < 1e-6
        ):
            raise RuntimeError(f"seed {seed} numerical endpoint gate failed")

    if not (
        final["campaign_pass"]
        and final["observed_complete_passes"] == 5
        and final["required_complete_passes"] == 4
        and all(final["per_seed"][str(seed)]["full_pass"] for seed in SEEDS)
    ):
        raise RuntimeError("final five-seed campaign verdict failed")
    expected_pooled = {
        "consensus_272": 80,
        "foba64_svd208": 80,
        "gradient_rank": 0,
        "omp64_svd208": 80,
        "omp_272": 0,
        "top_svd": 80,
    }
    if final["pooled_bidirectional_by_method"] != expected_pooled:
        raise RuntimeError("pooled comparator result changed")
    return {
        "status": "validated",
        "complete_passes": 5,
        "required": 4,
        "confirmation_bidirectional": 80,
        "random_supports_checked": 5 * 999,
        "protocol_sha256": HASHES["QWEN30B_FRESH_FIVESEED_PROTOCOL.md"],
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
