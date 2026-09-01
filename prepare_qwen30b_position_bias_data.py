#!/usr/bin/env python3
"""Freeze Qwen3 30B organism and causal-audit partitions after its base screen."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from prepare_mistral24b_position_bias_data import expand


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/behavior_audit"
CANDIDATES = DATA / "post_training_regression_v2_candidates.jsonl"
SCREEN = DATA / "qwen30b_position_bias_base_screen.json"
OUTPUT = DATA / "qwen30b_position_bias_v1.jsonl"
MANIFEST = DATA / "qwen30b_position_bias_v1_manifest.json"
PARTITION_OUTPUTS = {
    "train_validation": DATA / "qwen30b_position_bias_train_validation.jsonl",
    "development": DATA / "qwen30b_position_bias_development.jsonl",
    "confirmation": DATA / "qwen30b_position_bias_confirmation.jsonl",
}
SCREEN_SHA256 = "b26262c63e2d11f7a7d4fb6da5bd7b6d8823d34679c575d6589f13a051f4281a"
SELECTION_SEED = 20_260_831_30
CATEGORIES = (
    "business_ethics",
    "high_school_psychology",
    "high_school_world_history",
    "professional_law",
)
PER_CATEGORY = {"train": 9, "validation": 4, "development": 3, "confirmation": 4}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()


def build() -> dict:
    if sha256(SCREEN) != SCREEN_SHA256:
        raise RuntimeError("Qwen screen hash mismatch")
    screen = json.loads(SCREEN.read_text())
    if not screen["promotion_gate_pass"] or screen["n_qualified_questions"] < 80:
        raise RuntimeError("frozen Qwen base screen did not permit training")
    if screen["model"] != "Qwen/Qwen3-30B-A3B-Instruct-2507":
        raise RuntimeError("wrong capability screen")

    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line]
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in candidates:
        by_source[item["candidate_id"]][item["position"]] = item
    qualified = set(screen["qualified_source_ids"])
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
            start, stop = offsets[category], offsets[category] + count
            if len(eligible[category]) < stop:
                raise RuntimeError(f"not enough qualified sources in {category}")
            selections[partition].extend(eligible[category][start:stop])
            offsets[category] = stop
    selected = [source for values in selections.values() for source in values]
    if len(selected) != 80 or len(set(selected)) != 80:
        raise RuntimeError("Qwen partitions are not source-disjoint")

    rows = [
        item
        for partition, sources in selections.items()
        for source in sources
        for item in expand(source, by_source[source], partition)
    ]
    if len(rows) != 640:
        raise RuntimeError("expected 80 complete eight-family sources")
    OUTPUT.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))
    partition_rows = {
        "train_validation": [row for row in rows if row["audit_partition"] in {"train", "validation"}],
        "development": [row for row in rows if row["audit_partition"] == "development"],
        "confirmation": [row for row in rows if row["audit_partition"] == "confirmation"],
    }
    for name, path in PARTITION_OUTPUTS.items():
        path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in partition_rows[name]))
    manifest = {
        "status": "frozen_before_qwen_organism_training",
        "model": screen["model"],
        "model_revision": screen["model_revision"],
        "parameters": screen["parameters"],
        "behavior": "irrelevant ordering marker causes a first-option A bias",
        "selection_seed": SELECTION_SEED,
        "selection_uses_only_presplit_base_screen": True,
        "per_category": PER_CATEGORY,
        "selected_sources": selections,
        "source_counts": {partition: len(values) for partition, values in selections.items()},
        "row_counts": {
            partition: sum(row["audit_partition"] == partition for row in rows)
            for partition in selections
        },
        "families": sorted({row["family"] for row in rows}),
        "source_disjoint_partitions": True,
        "candidate_sha256": sha256(CANDIDATES),
        "screen_sha256": sha256(SCREEN),
        "output_sha256": sha256(OUTPUT),
        "partition_outputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(partition_rows[name])}
            for name, path in PARTITION_OUTPUTS.items()
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
