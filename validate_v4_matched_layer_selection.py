"""Fail-closed validation for the frozen V4 matched layer-selection artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from matched_layer_selection import evaluate_two_seed_gate


ROOT = Path(__file__).resolve().parent
RESULT_TEMPLATE = (
    "results/behavioral_causal_audit/"
    "matched_layer_selection_seed{seed}_v4-matched-static-k2-layer-selection.json"
)
EXPECTED_HASHES = {
    "matched_layer_selection.py": (
        "c666d481c24fd2621d094890d06bcf2fb6029e8c4502c5f174cea03882ba13a8"
    ),
    "modal_v4_matched_layer_selection.py": (
        "ce0619271b6046d916b65b356ef6f2062b455274701fc7bf043d8fd30973a76e"
    ),
    "tests/test_matched_layer_selection.py": (
        "ef20774e550f6ebb693209931cba5b3c036afcd6f61cd290366b003035be4389"
    ),
    "data/behavior_audit/post_training_regression_v3_stratified.jsonl": (
        "2b6aaf55e6a1a97f731af1ca74558b8af933aa25f4ec5c37c4f66780ffa51bc1"
    ),
    RESULT_TEMPLATE.format(seed=313): (
        "557c0cfcbeaba0e4a522244cb258a136670e808271596694dd05116b55526c23"
    ),
    RESULT_TEMPLATE.format(seed=317): (
        "ef1e0acd3d192b16e53d130294437349d4ac992938cfced057b4af6a94a6301c"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    observed_hashes = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        if not path.exists():
            raise SystemExit(f"missing frozen artifact: {relative}")
        observed = sha256(path)
        observed_hashes[relative] = observed
        if observed != expected:
            raise SystemExit(
                f"hash mismatch for {relative}: expected {expected}, observed {observed}"
            )

    results = {}
    for seed in (313, 317):
        path = ROOT / RESULT_TEMPLATE.format(seed=seed)
        result = json.loads(path.read_text())
        if result["training_seed"] != seed:
            raise SystemExit(f"seed mismatch in {path}")
        if result.get("sealed_test_opened") is not False:
            raise SystemExit(f"sealed test was not closed in seed {seed}")
        if result.get("phase") != "development":
            raise SystemExit(f"seed {seed} was not labeled development")
        if result.get("gate", {}).get("passes") is not False:
            raise SystemExit(f"stored gate unexpectedly passed for seed {seed}")
        results[seed] = result

    combined = evaluate_two_seed_gate(results)
    if combined["passes"]:
        raise SystemExit("combined development gate unexpectedly passed")
    prospective = ROOT / RESULT_TEMPLATE.format(seed=331)
    if prospective.exists():
        raise SystemExit("prospective seed artifact exists despite failed development gate")

    print(
        json.dumps(
            {
                "status": "validated_rejected_on_development",
                "hashes_verified": len(observed_hashes),
                "development_gate": combined,
                "prospective_seed_331_opened": False,
                "sealed_test_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
