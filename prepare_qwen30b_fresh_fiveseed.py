#!/usr/bin/env python3
"""Freeze unused Qwen30B sources for the five-seed replication."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare_mistral24b_position_bias_data import expand

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/behavior_audit"
CANDIDATES = DATA / "post_training_regression_v2_candidates.jsonl"
SCREEN = DATA / "qwen30b_position_bias_base_screen.json"
PRIOR = DATA / "qwen30b_position_bias_v1_manifest.json"
PROTOCOL = ROOT / "QWEN30B_FRESH_FIVESEED_PROTOCOL.md"
MANIFEST = DATA / "qwen30b_fresh_fiveseed_manifest.json"
OUTPUTS = {
    "train_validation": DATA / "qwen30b_fresh_fiveseed_train_validation.jsonl",
    "selection": DATA / "qwen30b_fresh_fiveseed_selection.jsonl",
    "validation": DATA / "qwen30b_fresh_fiveseed_validation.jsonl",
    "confirmation": DATA / "qwen30b_fresh_fiveseed_confirmation.jsonl",
}
SELECTION_SEED = 20_260_902_30
CATEGORIES = ("business_ethics", "high_school_psychology", "high_school_world_history", "professional_law")
ALLOCATIONS = {
    "train": (7, 9, 13, 7),
    "organism_validation": (3, 4, 6, 3),
    "selection": (3, 3, 3, 3),
    "causal_validation": (3, 3, 3, 3),
    "confirmation": (3, 4, 6, 3),
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def priority(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()

def build() -> dict:
    screen = json.loads(SCREEN.read_text())
    prior = json.loads(PRIOR.read_text())
    if not screen["promotion_gate_pass"] or screen["model"] != "Qwen/Qwen3-30B-A3B-Instruct-2507":
        raise RuntimeError("wrong or failed base screen")
    excluded = {source for values in prior["selected_sources"].values() for source in values}
    qualified = set(screen["qualified_source_ids"]) - excluded
    candidates = [json.loads(line) for line in CANDIDATES.read_text().splitlines() if line]
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in candidates:
        by_source[item["candidate_id"]][item["position"]] = item
    available = Counter(source.split(":", 1)[0] for source in qualified)
    required = {category: sum(counts[i] for counts in ALLOCATIONS.values()) for i, category in enumerate(CATEGORIES)}
    if any(available[c] < required[c] for c in CATEGORIES):
        raise RuntimeError("not enough unused qualified sources")
    ordered = {c: sorted((s for s in qualified if s.startswith(c + ":") and set(by_source[s]) == {"A", "B"}), key=priority) for c in CATEGORIES}
    offsets = {c: 0 for c in CATEGORIES}
    selected = {partition: [] for partition in ALLOCATIONS}
    for partition, counts in ALLOCATIONS.items():
        for i, category in enumerate(CATEGORIES):
            start, stop = offsets[category], offsets[category] + counts[i]
            selected[partition].extend(ordered[category][start:stop])
            offsets[category] = stop
    flattened = [source for values in selected.values() for source in values]
    if len(flattened) != 92 or len(set(flattened)) != 92 or set(flattened) & excluded:
        raise RuntimeError("fresh partitions overlap")
    rows = {
        partition: [
            row
            for source in sources
            for row in expand(
                source, by_source[source],
                "validation" if partition == "organism_validation" else partition,
            )
        ]
        for partition, sources in selected.items()
    }
    output_rows = {
        "train_validation": rows["train"] + rows["organism_validation"],
        "selection": rows["selection"],
        "validation": rows["causal_validation"],
        "confirmation": rows["confirmation"],
    }
    for name, path in OUTPUTS.items():
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows[name]))
    manifest = {
        "status": "frozen_before_fresh_qwen_training",
        "model": screen["model"], "model_revision": screen["model_revision"],
        "selection_seed": SELECTION_SEED, "prior_sources_excluded": len(excluded),
        "overlap_with_prior_campaign": 0, "source_disjoint": True,
        "available_after_exclusion": dict(sorted(available.items())),
        "required_by_category": required, "selected_sources": selected,
        "source_counts": {name: len(values) for name, values in selected.items()},
        "candidate_sha256": sha256(CANDIDATES), "screen_sha256": sha256(SCREEN),
        "prior_manifest_sha256": sha256(PRIOR), "protocol_sha256": sha256(PROTOCOL),
        "outputs": {name: {"path": str(path.relative_to(ROOT)), "rows": len(output_rows[name]), "sha256": sha256(path)} for name, path in OUTPUTS.items()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
