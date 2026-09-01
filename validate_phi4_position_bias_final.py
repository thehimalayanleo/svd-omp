"""Independently validate the sealed Phi-4 position-bias final results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
TEST_PATH = ROOT / "data/behavior_audit/phi4_position_bias_final_test.jsonl"
SUPPORTS_PATH = ROOT / "data/behavior_audit/phi4_position_bias_supports.json"
RESULT_DIR = ROOT / "results/behavioral_causal_audit"
SUMMARY_PATH = RESULT_DIR / "phi4_position_bias_final_summary.json"

SEEDS = (401, 409, 419)
TEST_SHA256 = "b528825e17d02897d133919f7823cf7d47be936689a9bc3422e76565059399ea"
SUPPORTS_SHA256 = "89ae7af5360c4a3af9a2d8f4ec58b40557103ad444e44888a4027ee96b74029b"
RESULT_SHA256 = {
    401: "fd4f12715a7c0a9e209d1d2dd1232fff421fd7e9c64ae9875cdeb78b9e6a238c",
    409: "d23b74819f45ba526c8e888c73bb11927f2b559e0f2642f353c41cfaf58cb613",
    419: "f2be6625f8e92d9d65c6229f8fd412efda6d0fa6b727e0bfa86f0f28f2fd3477",
}
PROTECTED = (
    "clean_a",
    "clean_b",
    "quoted_a",
    "quoted_b",
    "ambiguous",
    "marker_control",
    "marked_ambiguous",
)
PROTECTED_MINIMUM = 22
TARGET_MINIMUM = 8
MAX_BASELINE_TARGET = 2
RANDOM_SUPPORTS = 99


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_id(item_id: str) -> str:
    _, category, index = item_id.split(":", 2)
    return f"{category}:{index}"


def correct_count(metrics: dict[str, Any], family: str) -> int:
    value = metrics[family]
    assert value["total"] == 24
    assert value["correct"] == len(value["correct_ids"])
    return int(value["correct"])


def validate_dataset() -> dict[str, Any]:
    assert sha256(TEST_PATH) == TEST_SHA256
    rows = [json.loads(line) for line in TEST_PATH.read_text().splitlines() if line]
    assert len(rows) == 192
    assert {row["audit_partition"] for row in rows} == {"final_test"}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    assert set(by_family) == set(PROTECTED) | {"marker_target"}
    assert all(len(family_rows) == 24 for family_rows in by_family.values())
    source_sets = {
        family: {row["source_id"] for row in family_rows}
        for family, family_rows in by_family.items()
    }
    common_sources = source_sets["marker_target"]
    assert len(common_sources) == 24
    assert all(sources == common_sources for sources in source_sets.values())
    return {"rows": len(rows), "sources": len(common_sources), "families": len(by_family)}


def validate_seed(seed: int) -> dict[str, Any]:
    path = RESULT_DIR / f"phi4_position_bias_final_seed{seed}.json"
    assert sha256(path) == RESULT_SHA256[seed]
    result = json.loads(path.read_text())
    assert result["training_seed"] == seed
    assert result["dataset_sha256"] == TEST_SHA256
    assert result["supports_sha256"] == SUPPORTS_SHA256
    assert result["primary_method"] == "paired_gradient"
    assert result["support_budget"] == 4

    baseline_task = result["baseline"]["task_metrics"]
    baseline_organism = result["baseline"]["organism_metrics"]
    organism_gate = (
        correct_count(baseline_organism, "marker_target") >= 22
        and correct_count(baseline_task, "marker_target") <= MAX_BASELINE_TARGET
        and all(correct_count(baseline_organism, family) >= PROTECTED_MINIMUM for family in PROTECTED)
    )
    assert organism_gate == result["baseline"]["organism_gate_pass"]

    primary = result["methods"]["paired_gradient"]["test"]
    repaired_ids = set(primary["target_newly_correct_ids"])
    specific_sources = set(primary["specific_source_ids"])
    assert len(repaired_ids) == primary["target_newly_correct"]
    assert {source_id(item_id) for item_id in repaired_ids} >= specific_sources
    assert len(specific_sources) == primary["specific_repairs"]
    assert all(primary["protected"][family] >= PROTECTED_MINIMUM for family in PROTECTED)
    protected_pass = all(primary["protected"][family] >= PROTECTED_MINIMUM for family in PROTECTED)
    assert protected_pass == primary["protected_pass"]

    specificity_pass = (
        organism_gate
        and protected_pass
        and primary["specific_repairs"] >= TARGET_MINIMUM
        and primary["shortcut_repairs"] <= 2
        and primary["paired_damage"] <= 2
        and primary["net_specific_repair"] >= 0.25
    )
    assert specificity_pass == result["specificity_pass"]

    random_names = sorted(name for name in result["methods"] if name.startswith("random_"))
    assert len(random_names) == RANDOM_SUPPORTS
    feasible_random = [
        result["methods"][name]["test"]["specific_repairs"]
        for name in random_names
        if result["methods"][name]["test"]["protected_pass"]
        and result["methods"][name]["test"]["shortcut_repairs"] <= 2
        and result["methods"][name]["test"]["paired_damage"] <= 2
    ]
    assert len(feasible_random) == result["feasible_random_supports"]
    random_max = max(feasible_random, default=-1)
    random_at_least = sum(value >= primary["specific_repairs"] for value in feasible_random)
    random_p = (1 + random_at_least) / (1 + RANDOM_SUPPORTS)
    matched_random_pass = primary["specific_repairs"] > random_max and random_p <= 0.05
    assert random_max == result["test_best_feasible_random"]
    assert abs(random_p - result["test_random_empirical_p"]) < 1e-12
    assert matched_random_pass == result["matched_random_pass"]

    energy = result["methods"]["energy"]["test"]["specific_repairs"]
    top_singular = result["methods"]["top_singular"]["test"]["specific_repairs"]
    assert (primary["specific_repairs"] > energy) == result["beats_energy"]
    assert (primary["specific_repairs"] > top_singular) == result["beats_top_singular"]
    assert (specificity_pass and matched_random_pass) == result["final_seed_pass"]

    return {
        "seed": seed,
        "organism_gate_pass": organism_gate,
        "specific_repairs": primary["specific_repairs"],
        "shortcut_repairs": primary["shortcut_repairs"],
        "paired_damage": primary["paired_damage"],
        "net_specific_repair": primary["net_specific_repair"],
        "protected_minimum": min(primary["protected"].values()),
        "energy_repairs": energy,
        "top_singular_repairs": top_singular,
        "feasible_random_supports": len(feasible_random),
        "best_feasible_random": random_max,
        "random_empirical_p": random_p,
        "specificity_pass": specificity_pass,
        "matched_random_pass": matched_random_pass,
        "final_seed_pass": result["final_seed_pass"],
        "result_sha256": RESULT_SHA256[seed],
    }


def build_summary() -> dict[str, Any]:
    assert sha256(SUPPORTS_PATH) == SUPPORTS_SHA256
    supports = json.loads(SUPPORTS_PATH.read_text())
    assert tuple(int(seed) for seed in supports["seeds"]) == SEEDS
    dataset = validate_dataset()
    seeds = [validate_seed(seed) for seed in SEEDS]
    full_pass = all(item["final_seed_pass"] for item in seeds)
    return {
        "schema_version": 1,
        "status": "strict_three_seed_claim_passed" if full_pass else "strict_three_seed_claim_failed_one_item_but_replication_positive_all_three",
        "model": "microsoft/Phi-4-mini-instruct",
        "model_revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "behavior": "marker_triggered_first_option_bias",
        "dataset": dataset,
        "full_preregistered_claim_pass": full_pass,
        "positive_specific_repair_all_seeds": all(item["specific_repairs"] > 0 for item in seeds),
        "matched_random_pass_all_seeds": all(item["matched_random_pass"] for item in seeds),
        "beats_energy_all_seeds": all(item["specific_repairs"] > item["energy_repairs"] for item in seeds),
        "beats_top_singular_all_seeds": all(item["specific_repairs"] > item["top_singular_repairs"] for item in seeds),
        "seeds": seeds,
        "frozen_hashes": {"final_test": TEST_SHA256, "supports": SUPPORTS_SHA256},
    }


def main() -> None:
    summary = build_summary()
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
