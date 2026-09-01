#!/usr/bin/env python3
"""Freeze all development-selected supports before final test scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/behavioral_causal_audit"
INPUTS = {
    349: RESULTS / "paired_atom_foba_seed349_paired-atom-foba-development-v2.json",
    353: RESULTS / "paired_atom_foba_seed353_paired-atom-foba-development-v2.json",
}
OUTPUT = ROOT / "data/behavior_audit/fcs_final_validation_v2_supports.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    seeds = {}
    for seed, path in INPUTS.items():
        payload = json.loads(path.read_text())
        if payload["training_seed"] != seed or not payload["development_pass"]:
            raise RuntimeError(f"seed {seed} development artifact is not admissible")
        methods = {}
        for name, value in payload["methods"].items():
            frozen_name = "paired_gradient" if name == "gradient" else name
            methods[frozen_name] = {
                "support": value["support"],
                "dose": value["dose"],
                "development_point": value["development_point"],
            }
        random_names = sorted(name for name in methods if name.startswith("random_"))
        if len(random_names) != 20:
            raise RuntimeError(f"seed {seed} does not have twenty random supports")
        if set(methods) != {"robust_foba", "energy", "paired_gradient", *random_names}:
            raise RuntimeError(f"seed {seed} has unexpected methods")
        seeds[str(seed)] = {
            "support_budget": payload["support_budget"],
            "methods": methods,
            "development_artifact": str(path.relative_to(ROOT)),
            "development_artifact_sha256": sha256(path),
        }
    frozen = {
        "status": "frozen_before_final_test_scoring",
        "primary_method": "paired_gradient",
        "selection_description": "source-paired gradient ranking over SVD atoms",
        "model": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "adapter_tag": "fcs_marker_regression_stable_v2_rank16",
        "seeds": seeds,
    }
    OUTPUT.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return frozen


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
