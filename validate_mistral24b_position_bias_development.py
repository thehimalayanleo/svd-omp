"""Validate the frozen Mistral 24B sparse causal development result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULT_PATH = (
    ROOT
    / "results/behavioral_causal_audit/mistral24b_position_bias_development_seed503.json"
)
RESULT_SHA256 = "c85571f0158637ac11176e9f45534b236de87e8bd4d186b5ccced87f39bbc2e8"
MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
PARAMETERS = 24_011_361_280
DEV_HASHES = {
    "dev_a": "22e44a6787cc93eb838d71630bcb1db1ae9955b7f0a0f07b9e6d888ccabb96c0",
    "dev_b": "cda6d670b4c2cfb6c7b4ec979e44a5498702175c1855e219e1d547383bb05e57",
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
METHODS = ("paired_gradient", "energy", "top_singular")
DOSES = tuple(index / 2 for index in range(9))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct(metrics: dict[str, Any], family: str) -> int:
    family_metrics = metrics[family]
    assert family_metrics["total"] == 16
    assert family_metrics["correct"] == len(family_metrics["correct_ids"])
    return int(family_metrics["correct"])


def validate() -> dict[str, Any]:
    assert sha256(RESULT_PATH) == RESULT_SHA256
    result = json.loads(RESULT_PATH.read_text())

    assert result["model"] == MODEL
    assert result["model_revision"] == REVISION
    assert result["parameters"] == PARAMETERS
    assert result["training_seed"] == 503
    assert result["dev_hashes"] == DEV_HASHES
    assert result["final_test_mounted"] is False
    assert result["primary_method"] == "paired_gradient"
    assert result["support_budget"] == 4
    assert result["atom_components"] == 4
    assert len(result["candidate_layers"]) == 10

    for split_name in ("dev_a", "dev_b"):
        baseline = result["baseline"][split_name]
        assert correct(baseline, "marker_target") == 0
        assert min(correct(baseline, family) for family in PROTECTED) >= 15

    method_summary: dict[str, Any] = {}
    for method_name in METHODS:
        method = result["methods"][method_name]
        assert len(method["support"]) == 4
        assert tuple(float(dose) for dose in method["grid"]) == DOSES
        for dose in DOSES:
            cell = method["grid"][str(dose)]["development"]
            assert cell["protected_pass"] is True
            assert cell["specific_repairs"] == 0
            assert cell["shortcut_repairs"] == 0
            assert cell["paired_damage"] == 0
        assert method["selected"]["dose"] == 0.0
        validation = method["validation"]
        assert validation["protected_pass"] is True
        assert validation["specific_repairs"] == 0
        assert validation["shortcut_repairs"] == 0
        assert validation["paired_damage"] == 0
        method_summary[method_name] = {
            "selected_dose": method["selected"]["dose"],
            "development_repairs": method["selected"]["development"]["specific_repairs"],
            "validation_repairs": validation["specific_repairs"],
            "validation_protected_minimum": min(validation["protected"].values()),
        }

    assert len(result["random_supports"]) == 39
    assert result["validation_best_feasible_random"] == 0
    assert result["validation_random_empirical_p"] == 1.0

    return {
        "status": "validated_negative_development_result_final_sealed",
        "model": MODEL,
        "parameters": PARAMETERS,
        "training_seed": 503,
        "methods": method_summary,
        "random_supports": 39,
        "best_feasible_random": 0,
        "random_empirical_p": 1.0,
        "final_test_mounted": False,
        "result_sha256": RESULT_SHA256,
    }


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
