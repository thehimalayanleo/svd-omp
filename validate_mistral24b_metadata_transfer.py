#!/usr/bin/env python3
"""Independently validate the frozen second-behavior transfer ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/behavior_audit"
RESULTS = ROOT / "results/behavioral_causal_audit"
PROTOCOL = ROOT / "MISTRAL24B_METADATA_TRANSFER_PROTOCOL.md"
SEEDS = (907, 911, 919, 929, 937)
PRIMARY = "foba64_svd160"
EXPECTED_HASHES = {
    PROTOCOL: "118795e838c346aa0a34f2683f407638ef1260531084053df6c91ad47d057734",
    DATA / "mistral24b_metadata_transfer_train_validation.jsonl": "1a69e2f38f709988a029ade9f3c50e055af45d0e5a8e57c0e3e825ad11957ea4",
    DATA / "mistral24b_metadata_transfer_selection.jsonl": "992a48bd36b0109797d0b24e7d50e11ebd88c1e90a96860d6864e1ba44a07f08",
    DATA / "mistral24b_metadata_transfer_validation.jsonl": "e5760594df82016c497eb765cea56bc9220eb05d8285785dc15cea36060583e4",
    DATA / "mistral24b_metadata_transfer_confirmation.jsonl": "76052c5e3e3bc4e35f0e68fa5170a4d734287a7f72c2c9e97fa98af409e3a164",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"missing result: {path.name}")
    return json.loads(path.read_text())


def validate() -> dict:
    for path, expected in EXPECTED_HASHES.items():
        require(sha256(path) == expected, f"hash mismatch: {path.name}")
    manifest = load(DATA / "mistral24b_metadata_transfer_manifest.json")
    require(manifest["source_disjoint"], "source partitions are not disjoint")
    require(manifest["overlap_with_prior_campaign"] == 0, "prior sources were reused")
    require(manifest["source_counts"] == {
        "train": 18, "validation": 6, "selection": 8,
        "causal_validation": 8, "confirmation": 10,
    }, "unexpected source allocation")

    organisms = {
        seed: load(RESULTS / f"mistral24b_metadata_transfer_organism_seed{seed}.json")
        for seed in SEEDS
    }
    require(all(record["admitted"] for record in organisms.values()), "an organism failed admission")

    selection = load(RESULTS / "mistral24b_metadata_transfer_selection_summary.json")
    require(tuple(selection["training_seeds"]) == SEEDS, "wrong selection seeds")
    require(selection["confirmation_opened"] is False, "selection opened confirmation")
    for seed in SEEDS:
        record = selection["selections"][str(seed)]
        require(record["confirmation_mounted_during_development"] is False, "confirmation mounted during selection")
        require(record["input_validity"]["valid"], f"selection input invalid: {seed}")
        primary = record["curve"]["224"]["foba64_svd"]
        require(len(primary["support"]) == 224, f"wrong support size: {seed}")
        require(primary["record"]["bidirectional_count"] >= 6, f"selection support failed: {seed}")

    validation = load(RESULTS / "mistral24b_metadata_transfer_validation_summary.json")
    require(validation["confirmation_opened"] is False, "validation opened confirmation")
    issued = tuple(validation["issued_seeds"])
    require(len(issued) >= 4, "fewer than four supports issued")
    for seed in issued:
        record = validation["validations"][str(seed)]
        require(record["confirmation_mounted_during_development"] is False, "confirmation mounted during validation")
        require(record["passes_frozen_validation_gate"], f"issued seed did not pass validation: {seed}")

    final = load(RESULTS / "mistral24b_metadata_transfer_confirmation_summary.json")
    require(final["confirmation_opened"], "confirmation was not opened")
    require(final["all_failures_retained_in_denominator"], "failed seeds were not retained")
    require(tuple(final["training_seeds"]) == SEEDS, "wrong confirmation seeds")
    require(final["required_passing_seeds"] == 4, "wrong promotion threshold")
    require(final["passing_seeds"] == sum(final["per_seed_pass"].values()), "pass total mismatch")
    expected_passes = {}
    for seed in issued:
        record = final["confirmations"][str(seed)]
        primary = record["method_records"][PRIMARY]
        require(primary["bidirectional_count"] >= 8, f"confirmation target gate failed: {seed}")
        require(primary["feasible"], f"confirmation preservation gate failed: {seed}")
        randomization = record["randomization"]
        require(randomization["supports"] == 999, f"randomization incomplete: {seed}")
        expected_passes[str(seed)] = bool(record["dense_cycle_pass"])
    require(
        {str(seed): bool(final["per_seed_pass"][str(seed)]) for seed in issued} == expected_passes,
        "per-seed verdict does not match frozen dense-cycle gate",
    )
    return {
        "status": "validated_transfer_pass" if final["status"] == "transfer_pass" else "validated_transfer_failed",
        "transfer_status": final["status"],
        "passing_seeds": final["passing_seeds"],
        "issued_seeds": issued,
        "per_seed_pass": final["per_seed_pass"],
        "pooled_bidirectional_by_method": final["pooled_bidirectional_by_method"],
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
