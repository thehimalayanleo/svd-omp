#!/usr/bin/env python3
"""Freeze source-fresh data for the paper-grade Mistral 24B replication."""

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
    DATA / "mistral24b_multiseed_manifest.json",
)
OUTPUTS = {
    "development": DATA / "mistral24b_paper_replication_development.jsonl",
    "confirmation": DATA / "mistral24b_paper_replication_confirmation.jsonl",
}
MANIFEST = DATA / "mistral24b_paper_replication_manifest.json"
SELECTION_SEED = 20_260_831_01
MINIMUM_MARGIN = 0.1
ALLOCATION = {
    "development": {
        "business_ethics": 3,
        "high_school_psychology": 1,
        "high_school_world_history": 4,
        "professional_law": 4,
    },
    "confirmation": {
        "business_ethics": 4,
        "high_school_psychology": 2,
        "high_school_world_history": 6,
        "professional_law": 4,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def priority(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()


def build() -> dict:
    screen = json.loads(SCREEN.read_text())
    if screen["model"] != "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
        raise RuntimeError("wrong capability screen")
    if len(screen["margins"]) != 400:
        raise RuntimeError("incomplete capability screen")

    candidate_rows = [
        json.loads(line) for line in CANDIDATES.read_text().splitlines() if line
    ]
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in candidate_rows:
        by_source[item["candidate_id"]][item["position"]] = item

    prior_sources = set()
    prior_hashes = {}
    for path in PRIOR_MANIFESTS:
        manifest = json.loads(path.read_text())
        prior_hashes[path.name] = sha256(path)
        prior_sources.update(
            source
            for sources in manifest["selected_sources"].values()
            for source in sources
        )

    eligible = {
        source
        for source, margins in screen["margins"].items()
        if min(margins.values()) >= MINIMUM_MARGIN
        and source not in prior_sources
        and set(by_source[source]) == {"A", "B"}
    }
    by_category = {
        category: sorted(
            (source for source in eligible if source.startswith(category + ":")),
            key=priority,
        )
        for category in ALLOCATION["development"]
    }

    offsets = {category: 0 for category in by_category}
    selections = {partition: [] for partition in ALLOCATION}
    for partition, allocation in ALLOCATION.items():
        for category, count in allocation.items():
            start = offsets[category]
            stop = start + count
            if len(by_category[category]) < stop:
                raise RuntimeError(f"not enough fresh sources for {partition}:{category}")
            selections[partition].extend(by_category[category][start:stop])
            offsets[category] = stop

    selected = [source for values in selections.values() for source in values]
    if len(selected) != len(set(selected)):
        raise RuntimeError("replication partitions overlap")
    if set(selected) & prior_sources:
        raise RuntimeError("replication reuses a prior Mistral 24B source")

    output_records = {}
    for partition, sources in selections.items():
        rows = [
            item
            for source in sources
            for item in expand(source, by_source[source], f"paper_replication_{partition}")
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
        "status": "frozen_before_new_organism_training",
        "purpose": "fresh fixed-k224 replication and matched selector comparison",
        "selection_seed": SELECTION_SEED,
        "minimum_base_margin_each_family": MINIMUM_MARGIN,
        "allocation": ALLOCATION,
        "selected_sources": selections,
        "source_disjoint_partitions": True,
        "source_disjoint_from_all_prior_mistral24b_campaigns": True,
        "training_data_reuse": (
            "Organism training data are unchanged; causal development and confirmation "
            "sources are fresh and inaccessible during training."
        ),
        "screen": {
            "path": str(SCREEN.relative_to(ROOT)),
            "sha256": sha256(SCREEN),
            "candidate_sources": len(screen["margins"]),
            "eligible_fresh_sources": len(eligible),
        },
        "candidate_sha256": sha256(CANDIDATES),
        "prior_manifest_hashes": prior_hashes,
        "outputs": output_records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
