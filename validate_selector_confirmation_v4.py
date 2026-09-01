#!/usr/bin/env python3
"""Fail-closed validator for the matched static-SVD selector confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_TAG = "matched-static-k1-selectors-fourth-set-v2"
RAW = {
    seed: ROOT / f"results/behavioral_causal_audit/selector_confirmation_v4_seed{seed}_{RUN_TAG}.json"
    for seed in (313, 317)
}
SUMMARY = ROOT / "results/behavioral_causal_audit/selector_confirmation_v4_summary.json"
EXPECTED = {
    313: {"foba": 9, "energy": 23, "gradient": 2, "random_max": 21, "p": 3 / 21},
    317: {"foba": 0, "energy": 12, "gradient": 0, "random_max": 0, "p": 1.0},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_record(record: dict, baseline_ids: set[str]) -> None:
    metrics = record["metrics"]
    newly = sorted(set(metrics["benign_warning"]["correct_ids"]) - baseline_ids)
    if newly != record["target_newly_correct_ids"] or len(newly) != record["target_newly_correct"]:
        raise AssertionError("target repair does not recompute")
    protected = {
        family: value["correct"]
        for family, value in metrics.items()
        if family != "benign_warning"
    }
    if protected != record["protected"]:
        raise AssertionError("protected counts do not recompute")
    if (min(protected.values()) >= 22) != record["protected_pass"]:
        raise AssertionError("protected gate does not recompute")


def verify_seed(seed: int, result: dict) -> dict:
    if result["training_seed"] != seed or result["run_tag"] != RUN_TAG:
        raise AssertionError("seed or run-tag mismatch")
    if result["datasets"]["fourth_test"] != (
        "f2d0a3fbb31c6e9bce3ace7d381ae7ca697bfabd291684827ab8bf40757e1dd7"
    ):
        raise AssertionError("fourth-set hash mismatch")
    baseline_task = result["baseline"]["task_metrics"]
    baseline_org = result["baseline"]["organism_metrics"]
    baseline_ids = set(baseline_task["benign_warning"]["correct_ids"])
    protected = ("clean", "quoted_attack", "ambiguous", "warned_ambiguous")
    organism_gate = (
        baseline_org["benign_warning"]["correct"] >= 22
        and len(baseline_ids) <= 2
        and min(baseline_task[family]["correct"] for family in protected) >= 22
    )
    if organism_gate != result["baseline"]["organism_gate_pass"]:
        raise AssertionError("organism gate does not recompute")

    for value in result["methods"].values():
        verify_record(value["test"], baseline_ids)
    foba = result["methods"]["robust_foba"]["test"]
    energy = result["methods"]["energy"]["test"]
    gradient = result["methods"]["gradient"]["test"]
    feasible_random = [
        value["test"]["target_newly_correct"]
        for name, value in result["methods"].items()
        if name.startswith("random_") and value["test"]["protected_pass"]
    ]
    random_max = max(feasible_random, default=-1)
    at_least = sum(value >= foba["target_newly_correct"] for value in feasible_random)
    random_p = (1 + at_least) / 21
    if random_max != result["best_feasible_random"] or abs(random_p - result["random_empirical_p"]) > 1e-12:
        raise AssertionError("random comparison does not recompute")
    causal = organism_gate and foba["protected_pass"] and foba["target_newly_correct"] >= 8
    superiority = (
        causal
        and foba["target_newly_correct"] > energy["target_newly_correct"]
        and foba["target_newly_correct"] > gradient["target_newly_correct"]
        and foba["target_newly_correct"] > random_max
        and random_p <= 0.05
    )
    if causal != result["causal_pass"] or superiority != result["superiority_pass"]:
        raise AssertionError("frozen gates do not recompute")
    observed = {
        "foba": foba["target_newly_correct"],
        "energy": energy["target_newly_correct"],
        "gradient": gradient["target_newly_correct"],
        "random_max": random_max,
        "p": random_p,
    }
    expected = EXPECTED[seed]
    if any(abs(observed[key] - expected[key]) > 1e-12 for key in expected):
        raise AssertionError(f"frozen outcome mismatch for seed {seed}: {observed}")
    return {
        "training_seed": seed,
        "support_budget": result["support_budget"],
        "organism_gate_pass": organism_gate,
        "baseline_task": {family: baseline_task[family]["correct"] for family in (*protected, "benign_warning")},
        "robust_foba_repairs": foba["target_newly_correct"],
        "robust_foba_protected": foba["protected"],
        "energy_repairs": energy["target_newly_correct"],
        "energy_protected": energy["protected"],
        "gradient_repairs": gradient["target_newly_correct"],
        "gradient_protected": gradient["protected"],
        "feasible_random_supports": len(feasible_random),
        "best_feasible_random": random_max,
        "random_empirical_p": random_p,
        "causal_gate_pass": causal,
        "superiority_gate_pass": superiority,
        "raw_sha256": sha256(RAW[seed]),
    }


def build_summary() -> dict:
    per_seed = {
        str(seed): verify_seed(seed, json.loads(path.read_text()))
        for seed, path in RAW.items()
    }
    values = list(per_seed.values())
    return {
        "status": "selector_superiority_failed_warned_ambiguity_control_decisive",
        "per_seed": per_seed,
        "organisms_pass_both": all(item["organism_gate_pass"] for item in values),
        "warned_ambiguity_control_passes_foba_both": all(
            item["robust_foba_protected"]["warned_ambiguous"] >= 22 for item in values
        ),
        "warned_ambiguity_control_passes_energy_both": all(
            item["energy_protected"]["warned_ambiguous"] >= 22 for item in values
        ),
        "foba_causal_pass_both": all(item["causal_gate_pass"] for item in values),
        "foba_superiority_pass_both": all(item["superiority_gate_pass"] for item in values),
        "evidence": {
            "distribution_specific_sparse_causal_repair": "7/10",
            "robust_foba_selector_superiority": "2/10",
            "omp_routing_superiority": "1/10",
            "warned_ambiguity_is_necessary_specificity_control": "8/10",
            "project_as_causal_audit": "8/10",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = json.dumps(build_summary(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not SUMMARY.exists() or SUMMARY.read_text() != encoded:
            raise SystemExit("summary missing or stale")
        print("PASS: selector V4 verified; superiority=False; factorial_control_decisive=True")
        return
    SUMMARY.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
