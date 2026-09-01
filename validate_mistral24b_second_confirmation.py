#!/usr/bin/env python3
"""Validate the sealed three-seed Mistral 24B second-stage confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results/behavioral_causal_audit"
SUMMARY = RESULT_DIR / "mistral24b_second_confirmation_summary.json"
TRANSITION = RESULT_DIR / "mistral24b_multiseed_support_transition_summary.json"
CONFIRMATION_DATA = ROOT / "data/behavior_audit/mistral24b_multiseed_confirmation.jsonl"
PROTOCOL = ROOT / "MISTRAL24B_SECOND_STAGE_CONFIRMATION_PROTOCOL.md"
SEEDS = (503, 509, 521)
HASHES = {
    "summary": "e91c99ae82f85def1338a06a7ec5c2c1159bb8827c08cba15a40491612007817",
    "transition": "9fd242e8ba359f0dbf77073aed0b2e975879c5f9e2d2aa9ab860dc7c20e4d6a7",
    "confirmation_data": "8fd0b1747fe15dceb856d6b0e145a3d2c144128128145546fb1b6f3ed40b4971",
    "protocol": "6ca5bbd80f226be7e9fd82a85ac05735e21ecd688019d945ea950bedc048ea36",
    "seed503": "524fb4a7dd31324ab9a5ff0512f39f7f30589b33e0444ff5a436ee0ff63a690c",
    "seed509": "c4214baae861f8af5d60e380354717481aa33da3d2bf7cf1c3a330545e7821ac",
    "seed521": "86f6b31aa842282cd60f915eef7c615960450aefebeb9bd8a2cb120e4e86f269",
}
EXPECTED = {
    503: {"bidirectional": 16, "inserted_protected": 16, "ablated_protected": 16},
    509: {"bidirectional": 16, "inserted_protected": 15, "ablated_protected": 16},
    521: {"bidirectional": 10, "inserted_protected": 15, "ablated_protected": 16},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict:
    paths = {
        "summary": SUMMARY,
        "transition": TRANSITION,
        "confirmation_data": CONFIRMATION_DATA,
        "protocol": PROTOCOL,
        **{
            f"seed{seed}": RESULT_DIR / f"mistral24b_second_confirmation_seed{seed}.json"
            for seed in SEEDS
        },
    }
    observed_hashes = {name: sha256(path) for name, path in paths.items()}
    if observed_hashes != HASHES:
        raise RuntimeError(f"frozen artifact hash mismatch: {observed_hashes}")

    rows = [json.loads(line) for line in CONFIRMATION_DATA.read_text().splitlines() if line]
    sources = {row["source_id"] for row in rows}
    families = {row["family"] for row in rows}
    if len(rows) != 128 or len(sources) != 16 or len(families) != 8:
        raise RuntimeError("confirmation data is not the complete 16 by 8 factorial")

    summary = json.loads(SUMMARY.read_text())
    transition = json.loads(TRANSITION.read_text())
    transition_by_seed = {
        item["training_seed"]: item["curves"]["224"]
        for item in transition["transitions"]
    }
    confirmations = {item["training_seed"]: item for item in summary["confirmations"]}
    if tuple(sorted(confirmations)) != SEEDS:
        raise RuntimeError("frozen seed set changed")

    seed_records = {}
    for seed in SEEDS:
        item = confirmations[seed]
        record = item["record"]
        randomization = item["randomization"]
        expected = EXPECTED[seed]
        if item["stage"] != "second_confirmation" or not item["stage_pass"]:
            raise RuntimeError(f"seed {seed} did not preserve its passing decision")
        if item["protocol_sha256"] != HASHES["protocol"]:
            raise RuntimeError(f"seed {seed} protocol hash mismatch")
        if item["evaluation_data_sha256"] != HASHES["confirmation_data"]:
            raise RuntimeError(f"seed {seed} data hash mismatch")
        if item["original_24_source_final_test_mounted"]:
            raise RuntimeError("old 24-source final was mounted")
        if len(item["support"]) != 224 or len(set(item["support"])) != 224:
            raise RuntimeError(f"seed {seed} support is not 224 unique atoms")
        if item["support"] != transition_by_seed[seed]["support"]:
            raise RuntimeError(f"seed {seed} confirmation support changed after transition")
        if record["base_target_correct"] != 16 or record["post_target_errors"] != 16:
            raise RuntimeError(f"seed {seed} endpoints were not present on every source")
        if record["bidirectional_count"] != expected["bidirectional"]:
            raise RuntimeError(f"seed {seed} bidirectional count mismatch")
        if len(record["bidirectional_sources"]) != record["bidirectional_count"]:
            raise RuntimeError(f"seed {seed} bidirectional source count mismatch")
        if not set(record["bidirectional_sources"]) <= sources:
            raise RuntimeError(f"seed {seed} reports an unknown source")
        if record["inserted_protected_minimum"] != expected["inserted_protected"]:
            raise RuntimeError(f"seed {seed} inserted protection mismatch")
        if record["ablated_protected_minimum"] != expected["ablated_protected"]:
            raise RuntimeError(f"seed {seed} ablated protection mismatch")
        if record["insertion_pair_damage"] or record["ablation_pair_damage"]:
            raise RuntimeError(f"seed {seed} damaged a matched control")
        if not record["feasible"]:
            raise RuntimeError(f"seed {seed} feasibility flag is false")
        if not item["dense_cycle_pass"]:
            raise RuntimeError(f"seed {seed} dense cycle failed")
        if (
            item["dense_cycle"]["insert_prediction_agreement_with_post"] != 1.0
            or item["dense_cycle"]["ablate_prediction_agreement_with_base"] != 1.0
        ):
            raise RuntimeError(f"seed {seed} dense endpoint agreement is not exact")
        records = randomization["records"]
        if len(records) != 99 or randomization["supports"] != 99:
            raise RuntimeError(f"seed {seed} random-support denominator changed")
        random_at_least = sum(
            row["score"] >= randomization["selected_score"] for row in records
        )
        empirical_p = (1 + random_at_least) / 100
        if random_at_least != 0 or empirical_p != 0.01:
            raise RuntimeError(f"seed {seed} randomization result mismatch")
        if randomization["empirical_p"] != empirical_p:
            raise RuntimeError(f"seed {seed} stored p-value mismatch")
        seed_records[seed] = {
            "bidirectional": record["bidirectional_count"],
            "protected_minimum": min(
                record["inserted_protected_minimum"], record["ablated_protected_minimum"]
            ),
            "pair_damage": 0,
            "best_random_score": max(row["score"] for row in records),
            "empirical_p": empirical_p,
        }

    if (
        summary["status"] != "second_stage_multiseed_confirmation_pass"
        or not summary["all_seeds_pass"]
        or not summary["confirmation_opened"]
        or summary["original_24_source_final_test_opened"]
    ):
        raise RuntimeError("summary decision flags are inconsistent")
    return {
        "status": "validated_second_stage_multiseed_confirmation_pass",
        "support_budget": 224,
        "dictionary_atoms": 640,
        "total_bidirectional": sum(item["bidirectional"] for item in seed_records.values()),
        "total_sources": 48,
        "seeds": seed_records,
        "hashes": observed_hashes,
        "claim_boundary": "post-validation budget revision; synthetic regression; 24B Mistral family",
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
