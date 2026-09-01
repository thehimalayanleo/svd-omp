#!/usr/bin/env python3
"""Freeze source-fresh splits for prospective causal calibration."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from prepare_mistral24b_position_bias_data import expand


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/behavior_audit"
CANDIDATES = DATA / "post_training_regression_v2_candidates.jsonl"
SCREEN = DATA / "mistral24b_position_bias_base_screen.json"
CANDIDATES_SHA256 = "e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5"
SCREEN_SHA256 = "8935ef6fe01b34fc151e4e957fb40d9702ad2c5287c67b918aa2ba5ad486ba94"
SELECTION_SEED = 20_260_831_02
PRIOR_DATA = (
    DATA / "mistral24b_metadata_abstention_v3.jsonl",
    DATA / "mistral24b_metadata_abstention_v3_confirmation.jsonl",
    DATA / "mistral24b_metadata_abstention_v3_development.jsonl",
    DATA / "mistral24b_metadata_abstention_v3_train_validation.jsonl",
    DATA / "mistral24b_multiseed_confirmation.jsonl",
    DATA / "mistral24b_multiseed_development.jsonl",
    DATA / "mistral24b_multiseed_validation.jsonl",
    DATA / "mistral24b_paper_replication_confirmation.jsonl",
    DATA / "mistral24b_paper_replication_development.jsonl",
    DATA / "mistral24b_position_bias_dev_a.jsonl",
    DATA / "mistral24b_position_bias_dev_b.jsonl",
    DATA / "mistral24b_position_bias_expanded_dev_a.jsonl",
    DATA / "mistral24b_position_bias_expanded_dev_b.jsonl",
    DATA / "mistral24b_position_bias_final_test.jsonl",
    DATA / "mistral24b_position_bias_train_validation.jsonl",
    DATA / "mistral24b_position_bias_v1.jsonl",
)
OUTPUTS = {
    "selection": DATA / "mistral24b_causal_calibration_selection.jsonl",
    "validation": DATA / "mistral24b_causal_calibration_validation.jsonl",
    "confirmation": DATA / "mistral24b_causal_calibration_confirmation.jsonl",
}
MANIFEST = DATA / "mistral24b_causal_calibration_manifest.json"
ALLOCATION = {
    "selection": {
        "business_ethics": 3,
        "high_school_psychology": 3,
        "high_school_world_history": 3,
        "professional_law": 3,
    },
    "validation": {
        "business_ethics": 3,
        "high_school_psychology": 3,
        "high_school_world_history": 3,
        "professional_law": 3,
    },
    "confirmation": {
        "business_ethics": 4,
        "high_school_psychology": 4,
        "high_school_world_history": 5,
        "professional_law": 3,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()


def build() -> dict:
    if sha256(CANDIDATES) != CANDIDATES_SHA256 or sha256(SCREEN) != SCREEN_SHA256:
        raise RuntimeError("frozen candidate or screen hash mismatch")
    screen = json.loads(SCREEN.read_text())
    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line]
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in candidates:
        by_source[item["candidate_id"]][item["position"]] = item

    used = set()
    prior_hashes = {}
    for path in PRIOR_DATA:
        prior_hashes[path.name] = sha256(path)
        used.update(
            json.loads(line)["source_id"]
            for line in path.read_text().splitlines()
            if line
        )
    eligible = {
        source for source in screen["qualified_candidate_ids"]
        if source not in used and set(by_source[source]) == {"A", "B"}
    }
    categories = tuple(ALLOCATION["selection"])
    ranked = {
        category: sorted(
            (source for source in eligible if source.startswith(category + ":")),
            key=priority,
        )
        for category in categories
    }
    offsets = {category: 0 for category in categories}
    selected = {partition: [] for partition in ALLOCATION}
    for partition, allocation in ALLOCATION.items():
        for category, count in allocation.items():
            start = offsets[category]
            stop = start + count
            if len(ranked[category]) < stop:
                raise RuntimeError(f"not enough fresh sources for {partition}:{category}")
            selected[partition].extend(ranked[category][start:stop])
            offsets[category] = stop

    flattened = [source for sources in selected.values() for source in sources]
    if len(flattened) != len(set(flattened)) or set(flattened) & used:
        raise RuntimeError("prospective partitions are not source-disjoint")

    output_records = {}
    for partition, sources in selected.items():
        rows = [
            row
            for source in sources
            for row in expand(source, by_source[source], f"causal_calibration_{partition}")
        ]
        path = OUTPUTS[partition]
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        output_records[partition] = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(rows),
            "sources": len(sources),
            "sha256": sha256(path),
        }

    manifest = {
        "status": "frozen_before_prospective_organism_training_and_causal_evaluation",
        "purpose": "source-disjoint selection, validation, and confirmation for causal budget calibration",
        "selection_seed": SELECTION_SEED,
        "allocation": ALLOCATION,
        "selected_sources": selected,
        "source_disjoint_partitions": True,
        "source_disjoint_from_all_listed_prior_mistral24b_data": True,
        "candidate_sha256": sha256(CANDIDATES),
        "screen_sha256": sha256(SCREEN),
        "prior_data_hashes": prior_hashes,
        "eligible_fresh_sources_before_allocation": len(eligible),
        "outputs": output_records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
