#!/usr/bin/env python3
"""Freeze fresh source-disjoint development splits for the 24B expansion."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from prepare_mistral24b_position_bias_data import expand


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/behavior_audit"
CANDIDATES = DATA / "post_training_regression_v2_candidates.jsonl"
SCREEN = DATA / "mistral24b_position_bias_base_screen.json"
PRIOR = DATA / "mistral24b_position_bias_v1_manifest.json"
DEV_A = DATA / "mistral24b_position_bias_expanded_dev_a.jsonl"
DEV_B = DATA / "mistral24b_position_bias_expanded_dev_b.jsonl"
MANIFEST = DATA / "mistral24b_position_bias_expanded_manifest.json"
SCREEN_SHA256 = "8935ef6fe01b34fc151e4e957fb40d9702ad2c5287c67b918aa2ba5ad486ba94"
PRIOR_SHA256 = "7ee22bb3408bcae86645e90c035638bb2802e8087cb881794b15b33b54c8093b"
SELECTION_SEED = 20_260_901
CATEGORIES = (
    "business_ethics",
    "high_school_psychology",
    "high_school_world_history",
    "professional_law",
)
SOURCES_PER_CATEGORY = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()


def build() -> dict:
    if sha256(SCREEN) != SCREEN_SHA256 or sha256(PRIOR) != PRIOR_SHA256:
        raise RuntimeError("frozen input hash mismatch")
    screen = json.loads(SCREEN.read_text())
    prior = json.loads(PRIOR.read_text())
    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line]
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in candidates:
        by_source[item["candidate_id"]][item["position"]] = item

    used = {source for values in prior["selected_sources"].values() for source in values}
    eligible = set(screen["qualified_candidate_ids"]) - used
    selections = {"expanded_dev_a": [], "expanded_dev_b": []}
    for category in CATEGORIES:
        ranked = sorted(
            (
                source for source in eligible
                if source.startswith(category + ":") and set(by_source[source]) == {"A", "B"}
            ),
            key=priority,
        )
        required = 2 * SOURCES_PER_CATEGORY
        if len(ranked) < required:
            raise RuntimeError(f"not enough unused sources in {category}")
        selections["expanded_dev_a"].extend(ranked[:SOURCES_PER_CATEGORY])
        selections["expanded_dev_b"].extend(ranked[SOURCES_PER_CATEGORY:required])

    if set(selections["expanded_dev_a"]) & set(selections["expanded_dev_b"]):
        raise RuntimeError("expanded development overlap")
    if (set(selections["expanded_dev_a"]) | set(selections["expanded_dev_b"])) & used:
        raise RuntimeError("overlap with prior campaign")

    outputs = {"expanded_dev_a": DEV_A, "expanded_dev_b": DEV_B}
    for partition, path in outputs.items():
        rows = [
            row
            for source in selections[partition]
            for row in expand(source, by_source[source], partition)
        ]
        if len(rows) != 128 or len({row["source_id"] for row in rows}) != 16:
            raise RuntimeError("unexpected expanded split size")
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    manifest = {
        "status": "frozen_before_expanded_24b_causal_run",
        "purpose": "fresh development only; original 24-source final remains sealed",
        "selection_seed": SELECTION_SEED,
        "sources_per_category": SOURCES_PER_CATEGORY,
        "selected_sources": selections,
        "source_disjoint_from_prior_campaign": True,
        "source_disjoint_between_expanded_splits": True,
        "screen_sha256": sha256(SCREEN),
        "prior_manifest_sha256": sha256(PRIOR),
        "candidate_sha256": sha256(CANDIDATES),
        "outputs": {
            partition: {
                "path": str(path.relative_to(ROOT)),
                "rows": 128,
                "sources": 16,
                "sha256": sha256(path),
            }
            for partition, path in outputs.items()
        },
        "original_final_test_mounted": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
