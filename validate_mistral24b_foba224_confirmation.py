#!/usr/bin/env python3
"""Independently validate the sealed Mistral 24B FoBa-224 campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/behavioral_causal_audit"
SEEDS = (853, 857, 859, 863, 877)
PROTOCOL_SHA256 = "8cfac127d21e8b314e917ec3c1ef75232989410f13be6a5892a69f278a30d398"
VALIDATION_SHA256 = "261f51b5cc10f97b6179674a91e110ba3a532fdbcda197e8a2feaeb212fd9461"
CONFIRMATION_SHA256 = "12ebba2068110d1dc720aaa9f99d5fe0a1a0741cd1bafd14194cef4c27c8fa4b"
SUMMARY_SHA256 = "f731f40840bdd0adfa8098c37da58ffbc24269bd67507ecbc41c80b9ee2fa90f"
EXPECTED_BUDGETS = {
    "calibrated": 224,
    "direct_omp_64": 64,
    "foba64_svd_64": 64,
    "full_delta_640": 640,
    "gradient_rank_224": 224,
    "gradient_rank_64": 64,
    "omp64_svd_224": 224,
    "top_svd_224": 224,
    "top_svd_64": 64,
}
EXPECTED_AGGREGATES = {
    "calibrated": 45,
    "direct_omp_64": 12,
    "foba64_svd_64": 12,
    "full_delta_640": 50,
    "gradient_rank_224": 48,
    "gradient_rank_64": 10,
    "omp64_svd_224": 45,
    "top_svd_224": 45,
    "top_svd_64": 0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def passes(record: dict, sources: int) -> bool:
    required = (3 * sources + 3) // 4
    return bool(
        record["bidirectional_count"] >= required
        and record["inserted_protected_minimum"] >= sources - 1
        and record["ablated_protected_minimum"] >= sources - 1
        and record["insertion_pair_damage"] <= 1
        and record["ablation_pair_damage"] <= 1
    )


def validate() -> dict:
    protocol = ROOT / "MISTRAL24B_FOBA224_CONFIRMATION_PROTOCOL.md"
    validation_data = ROOT / "data/behavior_audit/mistral24b_causal_calibration_v3_validation.jsonl"
    confirmation_data = ROOT / "data/behavior_audit/mistral24b_causal_calibration_v3_confirmation.jsonl"
    summary_path = RESULTS / "mistral24b_foba224_confirmation_summary.json"
    assert sha256(protocol) == PROTOCOL_SHA256
    assert sha256(validation_data) == VALIDATION_SHA256
    assert sha256(confirmation_data) == CONFIRMATION_SHA256
    assert sha256(summary_path) == SUMMARY_SHA256

    aggregates = {method: 0 for method in EXPECTED_BUDGETS}
    validation_counts = {}
    confirmation_counts = {}
    for seed in SEEDS:
        validation = load(RESULTS / f"foba224_validation_seed{seed}.json")
        assert validation["training_seed"] == seed
        assert validation["stage"] == "development"
        assert validation["confirmation_mounted_during_development"] is False
        assert validation["protocol_sha256"] == PROTOCOL_SHA256
        assert validation["evaluation_data_sha256"] == VALIDATION_SHA256
        assert validation["candidate_budgets"] == {"calibrated": 224}
        record = validation["method_records"]["calibrated"]
        assert validation["passes"] is True and passes(record, 8)
        validation_counts[str(seed)] = record["bidirectional_count"]

        confirmation = load(RESULTS / f"foba224_confirmation_seed{seed}.json")
        assert confirmation["training_seed"] == seed
        assert confirmation["stage"] == "confirmation"
        assert confirmation["protocol_sha256"] == PROTOCOL_SHA256
        assert confirmation["evaluation_data_sha256"] == CONFIRMATION_SHA256
        assert confirmation["candidate_budgets"] == EXPECTED_BUDGETS
        assert set(confirmation["method_records"]) == set(EXPECTED_BUDGETS)
        for method, local in confirmation["method_records"].items():
            aggregates[method] += local["bidirectional_count"]
            assert confirmation["method_passes"][method] == passes(local, 10)
        primary = confirmation["method_records"]["calibrated"]
        assert confirmation["method_passes"]["calibrated"] is True
        assert primary["inserted_protected_minimum"] == 10
        assert primary["ablated_protected_minimum"] == 10
        assert primary["insertion_pair_damage"] == 0
        assert primary["ablation_pair_damage"] == 0
        confirmation_counts[str(seed)] = primary["bidirectional_count"]

    assert aggregates == EXPECTED_AGGREGATES
    summary = load(summary_path)
    assert tuple(summary["training_seeds"]) == SEEDS
    assert summary["issued_seeds"] == list(SEEDS)
    assert summary["confirmation_opened"] is True
    assert summary["promotion"] == {
        "all_issued_confirmed": True,
        "at_least_three_issued": True,
        "calibrated_atoms": 1120,
        "calibrated_bidirectional": 45,
        "fixed_top_svd_224_atoms": 1120,
        "fixed_top_svd_224_bidirectional": 45,
        "same_budget_win_over_fixed_top_svd_224": False,
        "system_confirmed": True,
    }
    return {
        "status": "validated",
        "seeds": SEEDS,
        "validation_bidirectional": validation_counts,
        "confirmation_bidirectional": confirmation_counts,
        "aggregates": aggregates,
        "primary_protected_minimum": 10,
        "primary_pair_damage": 0,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
