"""Validate the frozen Mistral 24B bidirectional expansion result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results/behavioral_causal_audit/mistral24b_bidirectional_expansion_seed503.json"
PROTOCOL = ROOT / "MISTRAL24B_BIDIRECTIONAL_EXPANSION_PROTOCOL.md"
RESULT_SHA256 = "a76a0b4ad8a539754dba304249fc2734432e59223fda72c6b195a161d4e21975"
PROTOCOL_SHA256 = "52e482845601135ab5335d00f1d38599df4fee4e1982f7b7b5ad6ec378d1feaf"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> dict:
    assert sha256(RESULT) == RESULT_SHA256
    assert sha256(PROTOCOL) == PROTOCOL_SHA256
    result = json.loads(RESULT.read_text())
    assert result["protocol_sha256"] == PROTOCOL_SHA256
    assert result["model"] == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    assert result["parameters"] == 24_011_361_280
    assert result["training_seed"] == 503
    assert result["original_final_test_mounted"] is False
    assert result["dictionary"]["language_layers"] == 40
    assert result["dictionary"]["rank_per_layer"] == 16
    assert result["dictionary"]["atoms"] == 640
    assert max(
        item["spectral_relative_reconstruction_error"]
        for item in result["dictionary"]["diagnostics"].values()
    ) < 1e-5

    assert result["admission_pass"] is False
    assert result["admission"]["development_a"] == {
        "base_protected_minimum": 6,
        "base_target_task_correct": 14,
        "post_protected_minimum": 15,
        "post_target_organism_consistent": 16,
    }
    assert result["admission"]["development_b"] == {
        "base_protected_minimum": 8,
        "base_target_task_correct": 14,
        "post_protected_minimum": 16,
        "post_target_organism_consistent": 16,
    }

    assert result["dense_cycle_pass"] is True
    dense_counts = {}
    for split, cycle in result["dense_cycle"].items():
        assert cycle["insert_prediction_agreement_with_post"] == 1.0
        assert cycle["ablate_prediction_agreement_with_base"] == 1.0
        assert cycle["behavior"]["bidirectional_count"] == 13
        dense_counts[split] = cycle["behavior"]["bidirectional_count"]

    validation = result["validation"]
    assert validation["spectral_foba"]["budget"] == 64
    assert validation["spectral_foba"]["record"]["specific_insertions"] == 3
    assert validation["spectral_foba"]["record"]["specific_repairs"] == 0
    assert validation["spectral_foba"]["record"]["bidirectional_count"] == 0
    assert validation["spectral_foba"]["record"]["feasible"] is False
    assert validation["top_singular"]["budget"] == 32
    assert validation["top_singular"]["record"]["specific_insertions"] == 14
    assert validation["top_singular"]["record"]["specific_repairs"] == 0
    assert validation["top_singular"]["record"]["feasible"] is True
    assert all(
        method["record"]["specific_repairs"] == 0
        and method["record"]["bidirectional_count"] == 0
        for method in validation.values()
    )

    random_results = result["random_supports"]
    assert len(random_results) == 19
    feasible_random = [
        item["record"]["bidirectional_count"]
        for item in random_results if item["record"]["feasible"]
    ]
    primary = validation["spectral_foba"]["record"]["bidirectional_count"]
    empirical_p = (1 + sum(value >= primary for value in feasible_random)) / 20
    assert len(feasible_random) == 2
    assert result["best_feasible_random"] == 0
    assert abs(result["random_empirical_p"] - empirical_p) < 1e-12
    assert result["sparse_pass"] is False
    assert result["status"] == "expanded_sparse_bidirectional_gate_failed"

    return {
        "status": "validated_dense_bidirectional_sparse_negative",
        "parameters": result["parameters"],
        "exact_atoms": result["dictionary"]["atoms"],
        "admission_pass": result["admission_pass"],
        "dense_cycle_pass": result["dense_cycle_pass"],
        "dense_bidirectional_counts": dense_counts,
        "spectral_foba_validation": validation["spectral_foba"]["record"],
        "top_singular_validation": validation["top_singular"]["record"],
        "all_sparse_methods_repaired_zero": True,
        "final_test_mounted": result["original_final_test_mounted"],
        "result_sha256": RESULT_SHA256,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
