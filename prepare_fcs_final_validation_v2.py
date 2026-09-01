#!/usr/bin/env python3
"""Freeze the final globally source-unused FCS test before model scoring."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from prepare_fcs_preregistered_validation import expand


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/behavior_audit"
CANDIDATES = DATA / "post_training_regression_v2_candidates.jsonl"
SCREEN = DATA / "post_training_regression_v2_base_screen.json"
PRIOR = (
    DATA / "post_training_regression_v2.jsonl",
    DATA / "post_training_regression_v3_stratified.jsonl",
    DATA / "post_training_regression_confirmation_v2.jsonl",
    DATA / "post_training_regression_hybrid_test.jsonl",
    DATA / "post_training_regression_selector_confirmation_v4.jsonl",
    DATA / "fcs_preregistered_validation_train.jsonl",
    DATA / "fcs_preregistered_validation_dev_a.jsonl",
    DATA / "fcs_preregistered_validation_dev_b.jsonl",
    DATA / "fcs_preregistered_validation_test.jsonl",
)
OUTPUT = DATA / "fcs_final_validation_v2_test.jsonl"
MANIFEST = DATA / "fcs_final_validation_v2_manifest.json"
SELECTION_SEED = 20_260_829
CATEGORY_COUNTS = {
    "business_ethics": 3,
    "high_school_psychology": 9,
    "high_school_world_history": 12,
    "professional_law": 0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def priority(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{source_id}".encode()).hexdigest()


def build() -> dict:
    candidates = read_rows(CANDIDATES)
    qualified = set(json.loads(SCREEN.read_text())["qualified_candidate_ids"])
    prior_used = {row["source_id"] for path in PRIOR for row in read_rows(path)}
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in candidates:
        by_source[row["candidate_id"]][row["position"]] = row
    complete = {source for source in qualified if set(by_source[source]) == {"A", "B"}}
    unused = complete - prior_used

    selected = []
    eligible_counts = {}
    for category, count in CATEGORY_COUNTS.items():
        eligible = sorted(
            (source for source in unused if source.startswith(category + ":")),
            key=priority,
        )
        eligible_counts[category] = len(eligible)
        if len(eligible) < count:
            raise RuntimeError(f"not enough globally unused sources for {category}")
        selected.extend(eligible[:count])
    if len(selected) != 24 or set(selected) & prior_used:
        raise RuntimeError("final test is not 24 globally unused sources")

    rows = []
    for index, source in enumerate(selected):
        position = "A" if index % 2 == 0 else "B"
        rows.extend(expand(by_source[source][position], "final_validation_v2", True))
    if len(rows) != 120:
        raise RuntimeError("unexpected final test row count")
    OUTPUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {
        "status": "frozen_before_model_scoring",
        "purpose": "Prospective final source-paired specificity test for stable organisms.",
        "selection_seed": SELECTION_SEED,
        "selection_rule": "lowest SHA256 priorities among every globally unused qualified source",
        "category_counts": CATEGORY_COUNTS,
        "eligible_unused_by_category": eligible_counts,
        "selected_sources": selected,
        "source_count": len(selected),
        "row_count": len(rows),
        "answer_positions": {"A": 12, "B": 12},
        "source_disjoint_from_all_prior_data": True,
        "candidate_sha256": sha256(CANDIDATES),
        "screen_sha256": sha256(SCREEN),
        "prior_sha256": {path.name: sha256(path) for path in PRIOR},
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
