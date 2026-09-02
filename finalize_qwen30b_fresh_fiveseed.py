"""Assemble the frozen Qwen3-30B campaign verdict without dropping failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEEDS = (947, 953, 967, 971, 977)
PRIMARY = "foba64_svd208"
METHODS = (
    "foba64_svd208",
    "omp64_svd208",
    "top_svd",
    "gradient_rank",
    "omp_272",
    "consensus_272",
)
PROTOCOL_SHA256 = "49cf051cba4462e43dbe526cf81f48c281aa68113a53f0f855917c4113a5200a"


def load(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"required retained artifact is missing: {path}")
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/behavioral_causal_audit"),
    )
    args = parser.parse_args()
    root = args.results
    stem = "qwen30b_fresh_fiveseed"

    organisms = {
        seed: load(root / f"{stem}_organism_seed{seed}.json") for seed in SEEDS
    }
    selection = load(root / f"{stem}_selection_summary.json")
    validation = load(root / f"{stem}_validation_summary.json")
    confirmation = load(root / f"{stem}_confirmation_summary.json")
    numeric = {
        seed: load(root / f"{stem}_numeric_seed{seed}.json") for seed in SEEDS
    }

    if tuple(selection["training_seeds"]) != SEEDS:
        raise RuntimeError("selection denominator changed")
    if tuple(validation["training_seeds"]) != SEEDS:
        raise RuntimeError("validation denominator changed")
    if tuple(confirmation["training_seeds"]) != SEEDS:
        raise RuntimeError("confirmation denominator changed")
    if not confirmation["all_failures_retained_in_denominator"]:
        raise RuntimeError("confirmation did not retain every failure")
    if not confirmation["confirmation_opened"]:
        raise RuntimeError("confirmation was not opened")

    per_seed = {}
    pooled = {method: 0 for method in METHODS}
    complete_passes = 0
    for seed in SEEDS:
        key = str(seed)
        organism_pass = bool(organisms[seed]["admitted"])
        selection_pass = bool(selection["selections"].get(key, {}).get("input_validity", {}).get("valid", False))
        validation_pass = seed in validation["issued_seeds"]
        item = confirmation["confirmations"].get(key)
        behavioral_pass = bool(item and item.get("passes_behavioral_confirmation_gate", False))
        numeric_item = numeric[seed]
        numeric_pass = bool(
            numeric_item["status"] == "float32_unmerged_dense_cycle_pass"
            and numeric_item["dtype"] == "float32"
            and numeric_item["adapter_merged"] is False
            and numeric_item["insertion"]["prediction_agreement"] == 1.0
            and numeric_item["ablation"]["prediction_agreement"] == 1.0
        )
        full_pass = all(
            (organism_pass, selection_pass, validation_pass, behavioral_pass, numeric_pass)
        )
        complete_passes += int(full_pass)
        method_counts = {}
        if item:
            for method in METHODS:
                count = item["method_records"][method]["bidirectional_count"]
                method_counts[method] = count
                pooled[method] += count
        randomization = item["randomization"] if item else None
        if randomization is not None and randomization["supports"] != 999:
            raise RuntimeError(f"seed {seed} did not retain 999 random supports")
        per_seed[key] = {
            "organism_admitted": organism_pass,
            "selection_input_valid": selection_pass,
            "support_issued_after_validation": validation_pass,
            "behavioral_confirmation_pass": behavioral_pass,
            "float32_unmerged_endpoint_pass": numeric_pass,
            "full_pass": full_pass,
            "bidirectional_by_method": method_counts,
            "randomization": {
                "supports": randomization["supports"],
                "random_at_least_selected": randomization["random_at_least_selected"],
                "empirical_p": randomization["empirical_p"],
            } if randomization else None,
        }

    verdict = {
        "status": "campaign_pass" if complete_passes >= 4 else "campaign_failed",
        "claim": (
            "Fresh-source, fresh-seed controlled exact-update causal replication "
            "in Qwen3-30B"
        ),
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "training_seeds": list(SEEDS),
        "all_failures_retained_in_denominator": True,
        "support_budget": {"selected_atoms": 272, "dictionary_atoms": 768},
        "primary_method": PRIMARY,
        "required_complete_passes": 4,
        "observed_complete_passes": complete_passes,
        "campaign_pass": complete_passes >= 4,
        "pooled_bidirectional_by_method": pooled,
        "per_seed": per_seed,
        "protocol_sha256": PROTOCOL_SHA256,
        "artifact_sha256": {
            "selection_summary": sha256(root / f"{stem}_selection_summary.json"),
            "validation_summary": sha256(root / f"{stem}_validation_summary.json"),
            "confirmation_summary": sha256(root / f"{stem}_confirmation_summary.json"),
            "numeric": {
                str(seed): sha256(root / f"{stem}_numeric_seed{seed}.json")
                for seed in SEEDS
            },
        },
        "claim_limits": [
            "controlled exact LoRA-update interventions, not natural checkpoint differences",
            "one synthetic ordering-marker regression",
            "no semantic interpretation of individual atoms",
            "no FoBa or OMP superiority claim unless the frozen comparator outcomes support it",
        ],
    }
    output = root / f"{stem}_final_summary.json"
    output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
