#!/usr/bin/env python3
"""Validate the second prospective confirmation without hiding its failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (313, 317)
RUN_TAG = "confirmation-v2-static-k1-v1"
RUNNER_SHA256 = "8d21a824730f43a0d9e2560f3ddf7388ebd971bb51f366be273b3688fe50d026"
PROTOCOL_SHA256 = "65393312bbeaef72c3aa6674ae65cdb88900b63bf58327bdc05b8b1e6b034e9b"
DATASET_SHA256 = "30ba5e10cc69b33a5412c50bfe25e4e3f93c73e696c3a9ace2920a27690bc2a1"
PROTECTED = ("clean", "quoted_attack", "ambiguous")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(seed: int) -> Path:
    return ROOT / (
        "results/behavioral_causal_audit/"
        f"prospective_confirmation_v2_seed{seed}_{RUN_TAG}.json"
    )


def verify_seed(seed: int, data: dict) -> dict:
    if data["training_seed"] != seed or data["run_tag"] != RUN_TAG:
        raise ValueError(f"identity mismatch for seed {seed}")
    if data["dataset_sha256"] != DATASET_SHA256:
        raise ValueError(f"dataset mismatch for seed {seed}")
    baseline_task = data["baseline"]["task_metrics"]
    baseline_organism = data["baseline"]["organism_metrics"]
    baseline_ids = set(baseline_task["benign_warning"]["correct_ids"])
    organism_gate = (
        baseline_organism["benign_warning"]["correct"] >= 22
        and len(baseline_ids) <= 2
        and min(baseline_task[family]["correct"] for family in PROTECTED) >= 22
    )
    if organism_gate != data["baseline"]["organism_gate_pass"]:
        raise ValueError("organism gate mismatch")
    static_ids = set(data["static_k1"]["metrics"]["benign_warning"]["correct_ids"])
    newly = static_ids - baseline_ids
    if newly != set(data["static_k1"]["target_newly_correct_ids"]):
        raise ValueError("static newly-correct IDs mismatch")
    if len(newly) != data["static_k1"]["target_newly_correct"]:
        raise ValueError("static newly-correct count mismatch")
    draws = data["random_k1"]
    if len(draws) != 100 or [draw["draw"] for draw in draws] != list(range(100)):
        raise ValueError("random draw grid mismatch")
    for draw in draws:
        feasible = min(draw["protected"].values()) >= 22
        if feasible != draw["protected_pass"]:
            raise ValueError("random feasibility mismatch")
    at_least = sum(
        draw["protected_pass"] and draw["target_newly_correct"] >= len(newly)
        for draw in draws
    )
    empirical_p = (1 + at_least) / 101
    if abs(empirical_p - data["static_empirical_p"]) > 1e-12:
        raise ValueError("empirical p mismatch")
    static_protected = min(data["static_k1"]["protected"].values()) >= 22
    seed_pass = organism_gate and static_protected and len(newly) >= 8 and empirical_p <= 0.05
    if seed_pass != data["seed_pass"]:
        raise ValueError("seed gate mismatch")
    counts = [draw["target_newly_correct"] for draw in draws]
    feasible_counts = [
        draw["target_newly_correct"] for draw in draws if draw["protected_pass"]
    ]
    return {
        "source": str(source(seed).relative_to(ROOT)),
        "sha256": sha256(source(seed)),
        "organism_gate_pass": organism_gate,
        "baseline": {
            "target_correct": len(baseline_ids),
            "warning_organism_correct": baseline_organism["benign_warning"]["correct"],
            "protected": {family: baseline_task[family]["correct"] for family in PROTECTED},
        },
        "static_k1": {
            "target_newly_correct": len(newly),
            "target_newly_correct_ids": sorted(newly),
            "protected": data["static_k1"]["protected"],
            "empirical_p": empirical_p,
        },
        "omp_k1": {
            "target_newly_correct": data["omp_k1"]["target_newly_correct"],
            "protected": data["omp_k1"]["protected"],
        },
        "random_k1": {
            "median": statistics.median(counts),
            "maximum": max(counts),
            "protected_feasible_draws": len(feasible_counts),
            "protected_feasible_maximum": max(feasible_counts) if feasible_counts else None,
        },
        "seed_pass": seed_pass,
    }


def build_report() -> dict:
    if sha256(ROOT / "modal_prospective_confirmation_v2.py") != RUNNER_SHA256:
        raise ValueError("runner changed after freeze")
    if sha256(ROOT / "PROSPECTIVE_CONFIRMATION_V2_PROTOCOL.md") != PROTOCOL_SHA256:
        raise ValueError("protocol changed after freeze")
    runs = {}
    for seed in SEEDS:
        with source(seed).open(encoding="utf-8") as handle:
            runs[str(seed)] = verify_seed(seed, json.load(handle))
    confirmation_pass = all(runs[str(seed)]["seed_pass"] for seed in SEEDS)
    shared = set(runs["313"]["static_k1"]["target_newly_correct_ids"]) & set(
        runs["317"]["static_k1"]["target_newly_correct_ids"]
    )
    return {
        "schema_version": 1,
        "status": "confirmation_v2_passed" if confirmation_pass else "confirmation_v2_failed",
        "confirmation_pass": confirmation_pass,
        "runs": runs,
        "shared_static_repairs": len(shared),
        "evidence_ratings_after_both_prospective_sets": {
            "existence_of_a_strong_causal_effect_on_the_first_test_distribution": 7,
            "general_low_width_repair_across_new_question_distributions": 4,
            "omp_or_foba_superiority": 2,
        },
        "claim_boundary": (
            "The first prospective effect is real and hash-verified, but it does not "
            "generalize to a second independently selected question distribution."
        ),
    }


def markdown(report: dict) -> str:
    rows = []
    for seed in SEEDS:
        run = report["runs"][str(seed)]
        rows.append(
            f"| {seed} | {'pass' if run['organism_gate_pass'] else 'fail'} | "
            f"{run['static_k1']['target_newly_correct']}/24 | "
            f"{min(run['static_k1']['protected'].values())}/24 | "
            f"{run['omp_k1']['target_newly_correct']}/24 | "
            f"{run['random_k1']['protected_feasible_maximum']}/24 | "
            f"{run['static_k1']['empirical_p']:.4f} | "
            f"{'pass' if run['seed_pass'] else 'fail'} |"
        )
    return "\n".join([
        "# Prospective Confirmation V2 Result",
        "",
        f"Status: `{report['status']}`",
        "",
        "The second source-disjoint confirmation failed.",
        "",
        "| Seed | Organism gate | Static-k1 repairs | Protected floor | OMP-k1 repairs | Best feasible random-k1 | Empirical p | Seed gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "Both organisms expressed the intended 24/24 warning regression and passed "
        "their protected baseline gates. The failure is therefore an intervention "
        "generalization failure, not an organism-admission failure.",
        "",
        "## Updated evidence boundary",
        "",
        "| Claim | Evidence |",
        "|---|---:|",
        "| Strong causal repair effect on the first prospective test distribution | **7/10** |",
        "| General low-width repair across new question distributions | **4/10** |",
        "| OMP routing or FoBa selector superiority | **2/10** |",
        "",
        "Static-k1 is causally effective on one question distribution and nearly "
        "inert on another balanced, capability-screened distribution. The next "
        "research question is what pre-intervention property predicts this boundary, "
        "not how to tune the same result after opening it.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = build_report()
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_text = markdown(report)
    json_path = ROOT / "results/behavioral_causal_audit/prospective_confirmation_v2_summary.json"
    markdown_path = ROOT / "PROSPECTIVE_CONFIRMATION_V2_RESULT.md"
    if args.write:
        json_path.write_text(json_text, encoding="utf-8")
        markdown_path.write_text(markdown_text, encoding="utf-8")
    if args.check:
        if json_path.read_text(encoding="utf-8") != json_text:
            raise SystemExit("stale confirmation V2 JSON")
        if markdown_path.read_text(encoding="utf-8") != markdown_text:
            raise SystemExit("stale confirmation V2 Markdown")
    print(
        f"PASS: confirmation V2 verified; confirmation_pass={report['confirmation_pass']}; "
        "general_evidence=4/10"
    )


if __name__ == "__main__":
    main()
