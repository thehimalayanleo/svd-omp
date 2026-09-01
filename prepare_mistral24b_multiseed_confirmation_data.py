#!/usr/bin/env python3
"""Freeze fresh, balanced 24B development, validation, and confirmation data."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from prepare_mistral24b_position_bias_data import expand


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/behavior_audit"
CANDIDATES = DATA / "post_training_regression_v2_candidates.jsonl"
SCREEN = DATA / "mistral24b_complete_base_screen.json"
PRIOR_MANIFESTS = (
    DATA / "mistral24b_position_bias_v1_manifest.json",
    DATA / "mistral24b_position_bias_expanded_manifest.json",
)
OUTPUTS = {
    "development": DATA / "mistral24b_multiseed_development.jsonl",
    "validation": DATA / "mistral24b_multiseed_validation.jsonl",
    "confirmation": DATA / "mistral24b_multiseed_confirmation.jsonl",
}
MANIFEST = DATA / "mistral24b_multiseed_manifest.json"
EXPECTED_HASHES = {
    "candidates": "e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5",
    "screen": "96faaae8a4c24308b529a5a649037b8f17f34d56fc7f6f4158fdb8de33d14edb",
    "prior_v1": "7ee22bb3408bcae86645e90c035638bb2802e8087cb881794b15b33b54c8093b",
    "prior_expanded": "694aea2267948524858623a760e625e790dcf8785998a566343ba3f474814ee8",
}
SELECTION_SEED = 20_260_903
CATEGORIES = (
    "business_ethics",
    "high_school_psychology",
    "high_school_world_history",
    "professional_law",
)
PER_CATEGORY = {"development": 3, "validation": 2, "confirmation": 4}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()


def build() -> dict:
    observed = {
        "candidates": sha256(CANDIDATES),
        "screen": sha256(SCREEN),
        "prior_v1": sha256(PRIOR_MANIFESTS[0]),
        "prior_expanded": sha256(PRIOR_MANIFESTS[1]),
    }
    if observed != EXPECTED_HASHES:
        raise RuntimeError(f"frozen input hash mismatch: {observed}")

    screen = json.loads(SCREEN.read_text())
    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line]
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in candidates:
        by_source[item["candidate_id"]][item["position"]] = item

    prior_sources = set()
    for path in PRIOR_MANIFESTS:
        manifest = json.loads(path.read_text())
        prior_sources.update(
            source for sources in manifest["selected_sources"].values() for source in sources
        )
    qualified = set(screen["qualified_source_ids"]) - prior_sources
    eligible = {
        category: sorted(
            (
                source for source in qualified
                if source.startswith(category + ":") and set(by_source[source]) == {"A", "B"}
            ),
            key=priority,
        )
        for category in CATEGORIES
    }

    selections = {partition: [] for partition in PER_CATEGORY}
    offsets = {category: 0 for category in CATEGORIES}
    for partition, count in PER_CATEGORY.items():
        for category in CATEGORIES:
            start = offsets[category]
            stop = start + count
            if len(eligible[category]) < stop:
                raise RuntimeError(f"not enough fresh qualified 24B sources in {category}")
            selections[partition].extend(eligible[category][start:stop])
            offsets[category] = stop

    selected = [source for sources in selections.values() for source in sources]
    if len(selected) != len(set(selected)):
        raise RuntimeError("new partitions overlap")
    if set(selected) & prior_sources:
        raise RuntimeError("new partitions reuse a prior 24B source")

    output_records = {}
    for partition, sources in selections.items():
        rows = [
            row
            for source in sources
            for row in expand(source, by_source[source], f"multiseed_{partition}")
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
        "status": "frozen_before_multiseed_organism_training",
        "purpose": "fresh 24B support selection, validation, and confirmation",
        "selection_seed": SELECTION_SEED,
        "per_category": PER_CATEGORY,
        "selected_sources": selections,
        "source_disjoint_partitions": True,
        "source_disjoint_from_all_prior_24b_runs": True,
        "phi4_source_overlap_allowed": (
            "Phi-4 is a different model campaign; only prior Mistral 24B source access is excluded"
        ),
        "complete_screen": {
            "path": str(SCREEN.relative_to(ROOT)),
            "sha256": observed["screen"],
            "candidate_questions": screen["n_candidate_questions"],
            "qualified_questions": screen["n_qualified_questions"],
            "required_families": screen["required_families"],
            "minimum_margin_each_condition": screen["minimum_margin_each_condition"],
            "organism_loaded": False,
        },
        "input_hashes": observed,
        "outputs": output_records,
        "original_24_source_final_test_opened": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
