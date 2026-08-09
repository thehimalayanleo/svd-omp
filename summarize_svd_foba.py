"""Combine sealed SVD-FoBa, protected SVD, and strengthened SWD results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def main(args: argparse.Namespace) -> None:
    foba = json.loads(args.foba.read_text())
    control = json.loads(args.control.read_text())
    expected_status = "sealed_fresh_test_window_frozen_before_extraction"
    if foba["status"] != expected_status or control["status"] != expected_status:
        raise ValueError("Both inputs must be sealed fresh-window results")
    if foba["activations_sha256"] != control["activations_sha256"]:
        raise ValueError("FoBa and SWD controls use different activation artifacts")
    control_rows = {
        (matrix["module"], row["selected_units"]): row
        for matrix in control["matrices"]
        for row in matrix["comparisons"]
    }
    rows = []
    for matrix in foba["matrices"]:
        for row in matrix["comparisons"]:
            key = (matrix["module"], row["selected_units"])
            swd = control_rows[key]
            rows.append(
                {
                    "module": matrix["module"],
                    "family": matrix["family"],
                    "selected_units": row["selected_units"],
                    "svd_foba_relative_error": row["svd_foba_relative_error"],
                    "svd_relative_error": row["svd_relative_error"],
                    "swd_relative_error": swd["swd_relative_error"],
                    "svd_error_over_foba": (
                        row["svd_relative_error"] / row["svd_foba_relative_error"]
                    ),
                    "swd_error_over_foba": (
                        swd["swd_relative_error"] / row["svd_foba_relative_error"]
                    ),
                    "relative_error_reduction_vs_svd": (
                        1.0
                        - row["svd_foba_relative_error"]
                        / row["svd_relative_error"]
                    ),
                    "fraction_inputs_improved": row["fraction_inputs_improved"],
                    "mean_accepted_swaps": row["mean_accepted_swaps"],
                    "swd_best_sparsity": swd["swd_best_sparsity"],
                }
            )
    svd_ratios = [row["svd_error_over_foba"] for row in rows]
    swd_ratios = [row["swd_error_over_foba"] for row in rows]
    reductions = [row["relative_error_reduction_vs_svd"] for row in rows]
    family_summary = {}
    for family in ("attention", "mlp"):
        selected = [row for row in rows if row["family"] == family]
        family_summary[family] = {
            "points": len(selected),
            "foba_wins_over_svd": sum(
                row["svd_error_over_foba"] > 1.0 for row in selected
            ),
            "foba_wins_over_swd": sum(
                row["swd_error_over_foba"] > 1.0 for row in selected
            ),
            "geometric_mean_swd_error_over_foba": geometric_mean(
                [row["swd_error_over_foba"] for row in selected]
            ),
        }
    by_k = []
    for k in foba["ks"]:
        selected = [row for row in rows if row["selected_units"] == k]
        by_k.append(
            {
                "selected_units": k,
                "median_relative_error_reduction_vs_svd": statistics.median(
                    row["relative_error_reduction_vs_svd"] for row in selected
                ),
                "median_swd_error_over_foba": statistics.median(
                    row["swd_error_over_foba"] for row in selected
                ),
            }
        )
    summary = {
        "status": expected_status,
        "method": foba["method"],
        "matrix_count": foba["matrix_count"],
        "point_count": len(rows),
        "foba_wins_over_svd": sum(value > 1.0 for value in svd_ratios),
        "foba_wins_over_swd": sum(value > 1.0 for value in swd_ratios),
        "geometric_mean_svd_error_over_foba": geometric_mean(svd_ratios),
        "geometric_mean_swd_error_over_foba": geometric_mean(swd_ratios),
        "minimum_swd_error_over_foba": min(swd_ratios),
        "median_relative_error_reduction_vs_svd": statistics.median(reductions),
        "mean_relative_error_reduction_vs_svd": statistics.fmean(reductions),
        "minimum_relative_error_reduction_vs_svd": min(reductions),
        "median_fraction_inputs_improved": statistics.median(
            row["fraction_inputs_improved"] for row in rows
        ),
        "median_mean_accepted_swaps": statistics.median(
            row["mean_accepted_swaps"] for row in rows
        ),
        "family_summary": family_summary,
        "by_selected_units": by_k,
        "frozen_method": foba["activation_metadata"]["frozen_method"],
        "source": {
            "weights_sha256": foba["weights_sha256"],
            "activations_sha256": foba["activations_sha256"],
            "foba_result_sha256": sha256_file(args.foba),
            "control_result_sha256": sha256_file(args.control),
            "foba_method_source_sha256": foba["method_source_sha256"],
            "foba_benchmark_source_sha256": foba["benchmark_source_sha256"],
            "swd_revision": control["swd_revision"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "by_selected_units"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foba", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
