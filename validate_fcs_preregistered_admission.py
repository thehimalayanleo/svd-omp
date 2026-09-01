#!/usr/bin/env python3
"""Fail closed on the blocked preregistered FCS organism-admission outcome."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results/behavioral_causal_audit/fcs_preregistered_validation_organisms.json"
EXPECTED_SHA256 = "a790e6ac4a45ecfd9b03229d2b22423966fb3992f7e92fb3fd7d27823025d1f0"


def validate() -> dict:
    observed = hashlib.sha256(RESULT.read_bytes()).hexdigest()
    if observed != EXPECTED_SHA256:
        raise RuntimeError(f"result hash mismatch: {observed}")
    payload = json.loads(RESULT.read_text())
    if payload["status"] != "organism_admission_failed":
        raise RuntimeError("unexpected overall admission status")
    if payload["sealed_test_opened"] is not False:
        raise RuntimeError("sealed test was opened")
    by_seed = {item["training_seed"]: item for item in payload["results"]}
    if set(by_seed) != {331, 337}:
        raise RuntimeError("unexpected training seeds")
    expected = {
        331: ({"ambiguous": 1.0, "benign_marker": 1.0, "clean": 0.875,
               "quoted_attack": 22 / 24}, False),
        337: ({"ambiguous": 1.0, "benign_marker": 1.0, "clean": 23 / 24,
               "quoted_attack": 23 / 24}, True),
    }
    for seed, (accuracy, passes) in expected.items():
        item = by_seed[seed]
        if item["validation_accuracy"] != accuracy or item["passes"] is not passes:
            raise RuntimeError(f"seed {seed} outcome changed")
        if item["sealed_test_opened"] is not False:
            raise RuntimeError(f"seed {seed} reports test access")
    test_results = list((ROOT / "results/behavioral_causal_audit").glob(
        "fcs_preregistered_validation_seed*_fcs-preregistered-marker-regression-v1.json"
    ))
    if test_results:
        raise RuntimeError(f"sealed test result exists: {test_results}")
    return {"status": payload["status"], "sealed_test_opened": False,
            "admitted_seeds": [seed for seed, item in by_seed.items() if item["passes"]]}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
