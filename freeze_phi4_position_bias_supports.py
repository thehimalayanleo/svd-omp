#!/usr/bin/env python3
"""Freeze development-selected Phi supports and the 99-support null."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEEDS = (401, 409, 419)
OUTPUT = ROOT / "data/behavior_audit/phi4_position_bias_supports.json"
DEV = {
    seed: ROOT / f"results/behavioral_causal_audit/phi4_position_bias_development_seed{seed}.json"
    for seed in SEEDS
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    seeds = {}
    for seed, path in DEV.items():
        result = json.loads(path.read_text())
        if result["training_seed"] != seed or result["final_test_mounted"]:
            raise RuntimeError("invalid development result")
        if len(result["random_supports"]) != 99:
            raise RuntimeError("random schedule is incomplete")
        primary_dose = result["primary_dose"]
        methods = {
            name: {
                "support": result["methods"][name]["support"],
                "dose": result["methods"][name]["selected"]["dose"],
                "development_selected": result["methods"][name]["selected"],
            }
            for name in ("paired_gradient", "energy", "top_singular")
        }
        for index, support in enumerate(result["random_supports"]):
            methods[f"random_{index:02d}"] = {
                "support": support,
                "dose": primary_dose,
                "dose_rule": "same frozen dose as paired_gradient",
            }
        seeds[str(seed)] = {
            "support_budget": result["support_budget"],
            "methods": methods,
            "development_result_sha256": sha256(path),
        }
    frozen = {
        "status": "frozen_before_final_test",
        "model": "microsoft/Phi-4-mini-instruct",
        "model_revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "behavior": "marker-triggered first-option A bias",
        "primary_method": "paired_gradient",
        "random_supports_per_seed": 99,
        "random_comparison": "same candidate universe, support budget, and primary dose",
        "seeds": seeds,
    }
    OUTPUT.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return frozen


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
