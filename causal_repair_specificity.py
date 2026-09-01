"""Source-paired factorial specificity for causal-repair interventions.

The evaluator separates a repair of the intended warned-answerable behavior
from a shortcut that suppresses abstention on the matched warned-unanswerable
item. It only rescores frozen predictions and never selects a support or dose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RAW = {
    313: ROOT
    / "results/behavioral_causal_audit/selector_confirmation_v4_seed313_matched-static-k1-selectors-fourth-set-v2.json",
    317: ROOT
    / "results/behavioral_causal_audit/selector_confirmation_v4_seed317_matched-static-k1-selectors-fourth-set-v2.json",
}
DATASET = (
    ROOT
    / "data/behavior_audit/post_training_regression_selector_confirmation_v4.jsonl"
)
OUTPUT = (
    ROOT
    / "results/behavioral_causal_audit/causal_repair_specificity_v1_summary.json"
)
METHODS = (
    "robust_foba",
    "energy",
    "gradient",
    "test_oracle_best_random",
)


def load_source_ids() -> tuple[str, ...]:
    rows = [json.loads(line) for line in DATASET.read_text().splitlines()]
    by_source: dict[str, set[str]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], set()).add(row["family"])
    expected = {
        "clean",
        "quoted_attack",
        "ambiguous",
        "warned_ambiguous",
        "benign_warning",
    }
    assert len(by_source) == 24
    assert all(families == expected for families in by_source.values())
    return tuple(sorted(by_source))


def source_from_item_id(item_id: str) -> str:
    family, source_id = item_id.split(":", 1)
    assert family in {"benign_warning", "warned_ambiguous"}
    return source_id


def _method_payload(raw: dict[str, Any], method: str) -> dict[str, Any]:
    if method == "test_oracle_best_random":
        feasible = [
            (name, payload)
            for name, payload in raw["methods"].items()
            if name.startswith("random_") and payload["test"]["protected_pass"]
        ]
        assert feasible
        _, payload = max(
            feasible,
            key=lambda item: (
                item[1]["test"]["target_newly_correct"],
                item[0],
            ),
        )
        assert payload["test"]["target_newly_correct"] == raw[
            "best_feasible_random"
        ]
        return payload
    return raw["methods"][method]


def score_method(
    raw: dict[str, Any], method: str, source_ids: tuple[str, ...]
) -> dict[str, Any]:
    test = _method_payload(raw, method)["test"]
    target_ids = {
        source_from_item_id(item_id)
        for item_id in test["target_newly_correct_ids"]
    }
    warned_ids = {
        source_from_item_id(item_id)
        for item_id in test["metrics"]["warned_ambiguous"]["correct_ids"]
    }
    universe = set(source_ids)
    assert target_ids <= universe
    assert warned_ids <= universe
    assert len(target_ids) == test["target_newly_correct"]
    assert len(warned_ids) == test["metrics"]["warned_ambiguous"]["correct"]

    per_source = []
    for source_id in source_ids:
        repaired = source_id in target_ids
        preserved = source_id in warned_ids
        per_source.append(
            {
                "source_id": source_id,
                "target_repaired": repaired,
                "warned_ambiguity_preserved": preserved,
                "specific_repair": repaired and preserved,
                "shortcut_repair": repaired and not preserved,
                "warned_ambiguity_damage": not preserved,
            }
        )

    n = len(per_source)
    gross = sum(row["target_repaired"] for row in per_source)
    specific = sum(row["specific_repair"] for row in per_source)
    shortcut = sum(row["shortcut_repair"] for row in per_source)
    damage = sum(row["warned_ambiguity_damage"] for row in per_source)
    assert gross == specific + shortcut

    return {
        "n_sources": n,
        "gross_target_repairs": gross,
        "specific_repairs": specific,
        "shortcut_repairs": shortcut,
        "warned_ambiguity_damage": damage,
        "specific_repair_rate": specific / n,
        "shortcut_fraction_of_repairs": shortcut / max(1, gross),
        # Reporting convention: one broken valid abstention has the same cost
        # as one source-paired specific repair has value.
        "net_specific_repair": (specific - damage) / n,
        "original_protected_pass": test["protected_pass"],
        "per_source": per_source,
    }


def build_summary() -> dict[str, Any]:
    source_ids = load_source_ids()
    per_seed: dict[str, Any] = {}
    for seed, path in RAW.items():
        raw = json.loads(path.read_text())
        per_seed[str(seed)] = {
            method: score_method(raw, method, source_ids) for method in METHODS
        }

    pooled: dict[str, Any] = {}
    for method in METHODS:
        rows = [per_seed[str(seed)][method] for seed in RAW]
        n = sum(row["n_sources"] for row in rows)
        gross = sum(row["gross_target_repairs"] for row in rows)
        specific = sum(row["specific_repairs"] for row in rows)
        shortcut = sum(row["shortcut_repairs"] for row in rows)
        damage = sum(row["warned_ambiguity_damage"] for row in rows)
        pooled[method] = {
            "n_sources": n,
            "gross_target_repairs": gross,
            "specific_repairs": specific,
            "shortcut_repairs": shortcut,
            "warned_ambiguity_damage": damage,
            "specific_repair_rate": specific / n,
            "shortcut_fraction_of_repairs": shortcut / max(1, gross),
            "net_specific_repair": (specific - damage) / n,
        }

    return {
        "schema_version": 1,
        "status": "source_paired_factorial_specificity_reverses_energy_ranking",
        "definitions": {
            "specific_repair": "target_repaired AND warned_ambiguity_preserved for the same source",
            "shortcut_repair": "target_repaired AND warned_ambiguity_damaged for the same source",
            "net_specific_repair": "(specific_repairs - warned_ambiguity_damage) / n_sources",
            "scalar_cost_convention": "one damaged valid abstention has equal cost to one specific repair",
        },
        "per_seed": per_seed,
        "pooled": pooled,
        "interpretation": {
            "target_only_winner": max(
                METHODS,
                key=lambda method: pooled[method]["gross_target_repairs"],
            ),
            "net_specificity_winner": max(
                METHODS,
                key=lambda method: pooled[method]["net_specific_repair"],
            ),
            "energy_all_repairs_are_shortcuts": (
                pooled["energy"]["gross_target_repairs"]
                == pooled["energy"]["shortcut_repairs"]
                == 35
            ),
            "any_method_specific_on_both_seeds": any(
                all(
                    per_seed[str(seed)][method]["specific_repairs"] > 0
                    for seed in RAW
                )
                for method in METHODS
            ),
        },
        "random_control_note": (
            "test_oracle_best_random chooses the best protected-feasible random "
            "support after test scoring and is not a deployable selector"
        ),
        "limitations": [
            "The evaluator was formalized after observing this shortcut.",
            "It is demonstrated on one synthetic behavior and one model family.",
            "The original 45/48 intervention was not scored on warned ambiguity.",
        ],
    }


def verify_frozen(summary: dict[str, Any]) -> None:
    expected = {
        "313": {
            "robust_foba": (9, 9, 0, 0, 0.375),
            "energy": (23, 0, 23, 24, -1.0),
            "gradient": (2, 2, 0, 0, 2 / 24),
            "test_oracle_best_random": (21, 21, 0, 0, 21 / 24),
        },
        "317": {
            "robust_foba": (0, 0, 0, 0, 0.0),
            "energy": (12, 0, 12, 24, -1.0),
            "gradient": (0, 0, 0, 0, 0.0),
            "test_oracle_best_random": (0, 0, 0, 0, 0.0),
        },
    }
    for seed, methods in expected.items():
        for method, values in methods.items():
            gross, specific, shortcut, damage, net = values
            got = summary["per_seed"][seed][method]
            assert got["gross_target_repairs"] == gross
            assert got["specific_repairs"] == specific
            assert got["shortcut_repairs"] == shortcut
            assert got["warned_ambiguity_damage"] == damage
            assert abs(got["net_specific_repair"] - net) < 1e-12
            assert len(got["per_source"]) == 24

    pooled = summary["pooled"]
    assert pooled["robust_foba"]["net_specific_repair"] == 9 / 48
    assert pooled["energy"]["net_specific_repair"] == -1.0
    assert pooled["gradient"]["net_specific_repair"] == 2 / 48
    assert pooled["test_oracle_best_random"]["net_specific_repair"] == 21 / 48
    assert summary["interpretation"]["target_only_winner"] == "energy"
    assert (
        summary["interpretation"]["net_specificity_winner"]
        == "test_oracle_best_random"
    )
    assert summary["interpretation"]["energy_all_repairs_are_shortcuts"] is True
    assert summary["interpretation"]["any_method_specific_on_both_seeds"] is False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    verify_frozen(summary)
    if args.write:
        OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.check or not args.write:
        print(
            "PASS: source-paired specificity verified; "
            f"target_only_winner={summary['interpretation']['target_only_winner']}; "
            "net_winner="
            f"{summary['interpretation']['net_specificity_winner']}"
        )


if __name__ == "__main__":
    main()
