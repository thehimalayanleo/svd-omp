"""Fail-closed validator for the source-paired specificity artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import causal_repair_specificity as specificity


ROOT = Path(__file__).resolve().parent
SUMMARY = (
    ROOT
    / "results/behavioral_causal_audit/causal_repair_specificity_v1_summary.json"
)
EXPECTED_SHA256 = {
    specificity.DATASET: "f2d0a3fbb31c6e9bce3ace7d381ae7ca697bfabd291684827ab8bf40757e1dd7",
    specificity.RAW[313]: "50ed8a8cbac8e0b784fa1e942fc92242843bab54e3de3e0e7e00226183503957",
    specificity.RAW[317]: "9fba1d1b06cd6676567c6ab6c2d17b970dc14756ff5c06eb6ee461bb02033103",
}


def verify_source_hashes() -> None:
    for path, expected in EXPECTED_SHA256.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (path, expected, actual)


def verify_baseline_admission() -> None:
    for seed, path in specificity.RAW.items():
        raw = json.loads(path.read_text())
        baseline = raw["baseline"]
        assert baseline["organism_gate_pass"] is True, seed
        assert baseline["task_metrics"]["benign_warning"]["correct"] == 0, seed
        assert baseline["task_metrics"]["warned_ambiguous"]["correct"] == 24, seed
        assert baseline["organism_metrics"]["benign_warning"]["correct"] == 24, seed


def verify_internal_aggregates(summary: dict[str, Any]) -> None:
    expected_sources = set(specificity.load_source_ids())
    for seed, methods in summary["per_seed"].items():
        assert set(methods) == set(specificity.METHODS), seed
        for method, result in methods.items():
            rows = result["per_source"]
            source_ids = [row["source_id"] for row in rows]
            assert len(source_ids) == len(set(source_ids)) == 24
            assert set(source_ids) == expected_sources
            gross = sum(row["target_repaired"] for row in rows)
            specific_count = sum(row["specific_repair"] for row in rows)
            shortcut = sum(row["shortcut_repair"] for row in rows)
            damage = sum(row["warned_ambiguity_damage"] for row in rows)
            assert gross == result["gross_target_repairs"]
            assert specific_count == result["specific_repairs"]
            assert shortcut == result["shortcut_repairs"]
            assert damage == result["warned_ambiguity_damage"]
            assert gross == specific_count + shortcut
            assert abs(
                result["net_specific_repair"]
                - (specific_count - damage) / 24
            ) < 1e-12


def verify_summary(summary: dict[str, Any]) -> None:
    verify_source_hashes()
    verify_baseline_admission()
    verify_internal_aggregates(summary)
    specificity.verify_frozen(summary)
    canonical = specificity.build_summary()
    assert summary == canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = json.loads(SUMMARY.read_text())
    verify_summary(summary)
    if args.check or True:
        print(
            "PASS: source pairing, baseline admission, raw IDs, aggregates, "
            "and frozen specificity outcomes verified"
        )


if __name__ == "__main__":
    main()
