"""Validate and summarize the direct CP-SVD discovery and confirmation runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "cp_svd_direct"
RUNS = (
    ("discovery", RESULTS / "direct_all_24.json"),
    ("confirmation", RESULTS / "direct_all_24_confirmation.json"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    loaded = [(name, path, json.loads(path.read_text())) for name, path in RUNS]
    reference = loaded[0][2]
    run_rows = []
    for name, path, result in loaded:
        if result["status"] != "direct_all_24_simultaneous_quality_and_latency":
            raise RuntimeError(f"unexpected status in {path}")
        if result["replacement_scope"] != "all_24_target_matrices_directly_replaced":
            raise RuntimeError(f"unexpected replacement scope in {path}")
        if result["frozen_configuration"] != reference["frozen_configuration"]:
            raise RuntimeError("frozen configurations differ across runs")
        for key in ("calibration_fingerprint", "heldout_fingerprint"):
            if result["activation_metadata"][key] != reference["activation_metadata"][key]:
                raise RuntimeError(f"{key} differs across runs")
        for metric in ("cross_entropy", "kl_to_dense", "logit_mse"):
            if result["quality_delta_from_hook_prototype"][metric] != 0.0:
                raise RuntimeError(f"direct replacement changed {metric}")
        speedup = float(result["latency"]["dense_over_candidate_speedup"])
        run_rows.append(
            {
                "run": name,
                "gpu": result["gpu"],
                "dense_milliseconds_median": result["latency"]["dense"][
                    "wall_milliseconds_median"
                ],
                "candidate_milliseconds_median": result["latency"]["candidate"][
                    "wall_milliseconds_median"
                ],
                "dense_over_candidate_speedup": speedup,
                "latency_reduction_fraction": 1.0 - 1.0 / speedup,
                "sha256": sha256(path),
            }
        )

    speedups = [row["dense_over_candidate_speedup"] for row in run_rows]
    storage_fraction = float(reference["storage"]["candidate_over_dense"])
    summary = {
        "status": "direct_cp_svd_t4_confirmed",
        "run_count": len(run_rows),
        "replacement_scope": reference["replacement_scope"],
        "input_shape": reference["latency"]["input_shape"],
        "quality_exactly_matches_frozen_cp_svd": True,
        "candidate_metrics": reference["candidate"],
        "minimum_dense_over_candidate_speedup": min(speedups),
        "geometric_mean_dense_over_candidate_speedup": math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        ),
        "candidate_factor_fraction_of_replaced_dense_weights": storage_fraction,
        "replaced_dense_weight_elements_over_candidate_factors": 1.0
        / storage_fraction,
        "runs": run_rows,
        "claim_boundary": (
            "Confirmed for the frozen Goodfire 67M all-24 replacement on a Tesla T4 "
            "with input shape 16x128; not a cross-hardware, active-edge, or "
            "dense-quality claim."
        ),
    }
    output = RESULTS / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
