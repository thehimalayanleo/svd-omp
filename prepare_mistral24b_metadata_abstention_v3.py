#!/usr/bin/env python3
"""Freeze exploratory metadata-abstention partitions from the preserved V2 screen."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from overabstention_data_v2 import expand


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/behavior_audit"
CANDIDATES = DATA / "post_training_regression_v2_candidates.jsonl"
SCREEN = DATA / "mistral24b_metadata_abstention_base_screen.json"
PROTOCOL = ROOT / "MISTRAL24B_METADATA_ABSTENTION_V3_PROTOCOL.md"
OUTPUT = DATA / "mistral24b_metadata_abstention_v3.jsonl"
MANIFEST = DATA / "mistral24b_metadata_abstention_v3_manifest.json"
PARTITION_OUTPUTS = {
    "train_validation": DATA / "mistral24b_metadata_abstention_v3_train_validation.jsonl",
    "development": DATA / "mistral24b_metadata_abstention_v3_development.jsonl",
    "confirmation": DATA / "mistral24b_metadata_abstention_v3_confirmation.jsonl",
}
SCREEN_SHA256 = "5597d0120adbe8373c53acef6e711657ae51b63b75644136f30b391fcaa526ad"
SELECTION_SEED = 20_260_831_03
CATEGORIES = (
    "business_ethics", "high_school_psychology",
    "high_school_world_history", "professional_law",
)
FAMILIES = {
    "ambiguous", "clean_a", "clean_b", "marked_ambiguous",
    "marker_control", "marker_target",
}
PER_CATEGORY = {"train": 3, "validation": 1, "development": 2, "confirmation": 4}
MIN_MARGIN = 0.1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()


def build() -> dict:
    if sha256(SCREEN) != SCREEN_SHA256:
        raise RuntimeError("preserved metadata screen hash mismatch")
    screen = json.loads(SCREEN.read_text())
    qualified = sorted(
        source for source, margins in screen["margins"].items()
        if all(margins[family] >= MIN_MARGIN for family in FAMILIES)
    )
    counts = Counter(source.split(":", 1)[0] for source in qualified)
    if len(qualified) < 40 or any(counts[category] < 10 for category in CATEGORIES):
        raise RuntimeError("exploratory narrowed capability gate failed")

    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line]
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in candidates:
        by_source[item["candidate_id"]][item["position"]] = item
    eligible = {
        category: sorted(
            (source for source in qualified if source.startswith(category + ":")),
            key=priority,
        )
        for category in CATEGORIES
    }
    selections = {partition: [] for partition in PER_CATEGORY}
    offsets = {category: 0 for category in CATEGORIES}
    for partition, count in PER_CATEGORY.items():
        for category in CATEGORIES:
            start, stop = offsets[category], offsets[category] + count
            selections[partition].extend(eligible[category][start:stop])
            offsets[category] = stop
    selected = [source for sources in selections.values() for source in sources]
    if len(selected) != 40 or len(set(selected)) != 40:
        raise RuntimeError("metadata partitions are not source-disjoint")
    rows = [
        row
        for partition, sources in selections.items()
        for source in sources
        for row in expand(source, by_source[source], partition)
        if row["family"] in FAMILIES
    ]
    if len(rows) != 240:
        raise RuntimeError("expected 40 complete six-family sources")
    OUTPUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    partitions = {
        "train_validation": [row for row in rows if row["audit_partition"] in {"train", "validation"}],
        "development": [row for row in rows if row["audit_partition"] == "development"],
        "confirmation": [row for row in rows if row["audit_partition"] == "confirmation"],
    }
    for name, path in PARTITION_OUTPUTS.items():
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in partitions[name]))
    manifest = {
        "status": "exploratory_partitions_frozen_before_training",
        "preregistered_base_screen": False,
        "design_informed_by_failed_v1_and_v2_screens": True,
        "qualified_total": len(qualified),
        "qualified_by_category": dict(sorted(counts.items())),
        "minimum_margin": MIN_MARGIN,
        "families": sorted(FAMILIES),
        "selection_seed": SELECTION_SEED,
        "per_category": PER_CATEGORY,
        "selected_sources": selections,
        "source_counts": {partition: len(sources) for partition, sources in selections.items()},
        "source_disjoint_partitions": True,
        "screen_sha256": sha256(SCREEN), "protocol_sha256": sha256(PROTOCOL),
        "candidate_sha256": sha256(CANDIDATES), "output_sha256": sha256(OUTPUT),
        "partition_outputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(partitions[name])}
            for name, path in PARTITION_OUTPUTS.items()
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
