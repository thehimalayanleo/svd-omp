#!/usr/bin/env python3
"""Freeze unused-source partitions for the second-behavior transfer."""

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
PRIOR_MANIFEST = DATA / "mistral24b_metadata_abstention_v3_manifest.json"
PROTOCOL = ROOT / "MISTRAL24B_METADATA_TRANSFER_PROTOCOL.md"
MANIFEST = DATA / "mistral24b_metadata_transfer_manifest.json"
OUTPUTS = {
    "train_validation": DATA / "mistral24b_metadata_transfer_train_validation.jsonl",
    "selection": DATA / "mistral24b_metadata_transfer_selection.jsonl",
    "validation": DATA / "mistral24b_metadata_transfer_validation.jsonl",
    "confirmation": DATA / "mistral24b_metadata_transfer_confirmation.jsonl",
}
SCREEN_SHA256 = "5597d0120adbe8373c53acef6e711657ae51b63b75644136f30b391fcaa526ad"
SELECTION_SEED = 20_260_901_01
MIN_MARGIN = 0.1
CATEGORIES = (
    "business_ethics",
    "high_school_psychology",
    "high_school_world_history",
)
FAMILIES = {
    "ambiguous",
    "clean_a",
    "clean_b",
    "marked_ambiguous",
    "marker_control",
    "marker_target",
}
ALLOCATIONS = {
    "train": (6, 6, 6),
    "validation": (2, 2, 2),
    "selection": (3, 3, 2),
    "causal_validation": (3, 2, 3),
    "confirmation": (3, 4, 3),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()


def build() -> dict:
    if sha256(SCREEN) != SCREEN_SHA256:
        raise RuntimeError("preserved base screen hash mismatch")
    screen = json.loads(SCREEN.read_text())
    prior = json.loads(PRIOR_MANIFEST.read_text())
    excluded = {
        source
        for sources in prior["selected_sources"].values()
        for source in sources
    }
    qualified = {
        source
        for source, margins in screen["margins"].items()
        if source.split(":", 1)[0] in CATEGORIES
        and source not in excluded
        and all(margins[family] >= MIN_MARGIN for family in FAMILIES)
    }
    available = Counter(source.split(":", 1)[0] for source in qualified)
    required = {
        category: sum(counts[index] for counts in ALLOCATIONS.values())
        for index, category in enumerate(CATEGORIES)
    }
    if any(available[category] < required[category] for category in CATEGORIES):
        raise RuntimeError("not enough unused capability-screened sources")

    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line]
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in candidates:
        by_source[item["candidate_id"]][item["position"]] = item
    ordered = {
        category: sorted(
            (source for source in qualified if source.startswith(category + ":")),
            key=priority,
        )
        for category in CATEGORIES
    }
    offsets = {category: 0 for category in CATEGORIES}
    selected: dict[str, list[str]] = {partition: [] for partition in ALLOCATIONS}
    for partition, counts in ALLOCATIONS.items():
        for index, category in enumerate(CATEGORIES):
            start = offsets[category]
            stop = start + counts[index]
            selected[partition].extend(ordered[category][start:stop])
            offsets[category] = stop
    flattened = [source for sources in selected.values() for source in sources]
    if len(flattened) != 50 or len(set(flattened)) != 50:
        raise RuntimeError("transfer partitions are not source-disjoint")
    if set(flattened) & excluded:
        raise RuntimeError("transfer reused a prior campaign source")

    rows_by_partition = {}
    for partition, sources in selected.items():
        rows = [
            row
            for source in sources
            for row in expand(source, by_source[source], partition)
            if row["family"] in FAMILIES
        ]
        if len(rows) != 6 * len(sources):
            raise RuntimeError(f"incomplete factorial in {partition}")
        rows_by_partition[partition] = rows
    output_rows = {
        "train_validation": rows_by_partition["train"] + rows_by_partition["validation"],
        "selection": rows_by_partition["selection"],
        "validation": rows_by_partition["causal_validation"],
        "confirmation": rows_by_partition["confirmation"],
    }
    for name, path in OUTPUTS.items():
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows[name]))

    manifest = {
        "status": "frozen_before_transfer_training",
        "behavior": "confidence_flag=low causes U on an answerable B question",
        "selection_seed": SELECTION_SEED,
        "minimum_base_margin": MIN_MARGIN,
        "categories": CATEGORIES,
        "families": sorted(FAMILIES),
        "prior_sources_excluded": len(excluded),
        "source_disjoint": True,
        "overlap_with_prior_campaign": 0,
        "available_after_exclusion": dict(sorted(available.items())),
        "selected_sources": selected,
        "source_counts": {name: len(sources) for name, sources in selected.items()},
        "screen_sha256": sha256(SCREEN),
        "prior_manifest_sha256": sha256(PRIOR_MANIFEST),
        "candidate_sha256": sha256(CANDIDATES),
        "protocol_sha256": sha256(PROTOCOL),
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "rows": len(output_rows[name]),
                "sha256": sha256(path),
            }
            for name, path in OUTPUTS.items()
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
