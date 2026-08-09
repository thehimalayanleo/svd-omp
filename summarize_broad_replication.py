"""Summarize three-model SVD-FoBa replication and simultaneous replacement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args: argparse.Namespace) -> None:
    models = []
    all_svd_ratios = []
    all_swd_ratios = []

    goodfire_summary = json.loads(args.goodfire_summary.read_text())
    with args.goodfire_points.open(newline="") as handle:
        goodfire_rows = list(csv.DictReader(handle))
    goodfire_svd = [float(row["svd_error_over_foba"]) for row in goodfire_rows]
    goodfire_swd = [float(row["swd_error_over_foba"]) for row in goodfire_rows]
    all_svd_ratios.extend(goodfire_svd)
    all_swd_ratios.extend(goodfire_swd)
    models.append(
        {
            "model_id": "goodfire/spd/t-9d2b8f02",
            "architecture": "LlamaSimpleMLP",
            "parameter_scale": "67M",
            "matrix_count": 24,
            "point_count": len(goodfire_rows),
            "foba_wins_over_svd": sum(value > 1.0 for value in goodfire_svd),
            "foba_wins_over_swd": sum(value > 1.0 for value in goodfire_swd),
            "geometric_mean_svd_error_over_foba": geometric_mean(goodfire_svd),
            "geometric_mean_swd_error_over_foba": geometric_mean(goodfire_swd),
            "minimum_swd_error_over_foba": min(goodfire_swd),
            "result_sha256": goodfire_summary["source"]["foba_result_sha256"],
        }
    )

    for path in args.replication:
        payload = json.loads(path.read_text())
        if payload["status"] != "sealed_cross_model_replication_frozen_method":
            raise ValueError(f"{path} is not a sealed cross-model result")
        rows = [
            row
            for matrix in payload["matrices"]
            for row in matrix["comparisons"]
        ]
        svd_ratios = [
            row["svd_relative_error"] / row["svd_foba_relative_error"]
            for row in rows
        ]
        swd_ratios = [
            row["swd_relative_error"] / row["svd_foba_relative_error"]
            for row in rows
        ]
        all_svd_ratios.extend(svd_ratios)
        all_swd_ratios.extend(swd_ratios)
        models.append(
            {
                "model_id": payload["model_id"],
                "model_revision": payload["model_revision"],
                "architecture": payload["architecture"],
                "matrix_count": payload["matrix_count"],
                "point_count": len(rows),
                "foba_wins_over_svd": sum(value > 1.0 for value in svd_ratios),
                "foba_wins_over_swd": sum(value > 1.0 for value in swd_ratios),
                "geometric_mean_svd_error_over_foba": geometric_mean(svd_ratios),
                "geometric_mean_swd_error_over_foba": geometric_mean(swd_ratios),
                "minimum_swd_error_over_foba": min(swd_ratios),
                "result_sha256": sha256_file(path),
            }
        )

    simultaneous = json.loads(args.simultaneous.read_text())
    if simultaneous["status"] != "sealed_fresh_test_all_24_matrices_simultaneous":
        raise ValueError("Simultaneous input is not sealed")
    foba_metrics = simultaneous["methods"]["svd_foba"]
    svd_metrics = simultaneous["methods"]["svd_omp"]
    swd_metrics = simultaneous["methods"]["swd"]
    summary = {
        "status": "sealed_three_model_replication_and_simultaneous_replacement",
        "model_count": len(models),
        "architecture_count": len({model["architecture"] for model in models}),
        "matrix_count": sum(model["matrix_count"] for model in models),
        "point_count": len(all_svd_ratios),
        "foba_wins_over_svd": sum(value > 1.0 for value in all_svd_ratios),
        "foba_wins_over_swd": sum(value > 1.0 for value in all_swd_ratios),
        "geometric_mean_svd_error_over_foba": geometric_mean(all_svd_ratios),
        "geometric_mean_swd_error_over_foba": geometric_mean(all_swd_ratios),
        "minimum_svd_error_over_foba": min(all_svd_ratios),
        "minimum_swd_error_over_foba": min(all_swd_ratios),
        "models": models,
        "simultaneous_all_24_goodfire": {
            "dense": simultaneous["dense"],
            "svd_foba": foba_metrics,
            "svd_omp": svd_metrics,
            "swd": swd_metrics,
            "winners": simultaneous["winners"],
            "swd_kl_over_foba": swd_metrics["kl_to_dense"] / foba_metrics["kl_to_dense"],
            "swd_logit_mse_over_foba": (
                swd_metrics["logit_mse"] / foba_metrics["logit_mse"]
            ),
            "swd_cross_entropy_minus_foba": (
                swd_metrics["cross_entropy"] - foba_metrics["cross_entropy"]
            ),
            "result_sha256": sha256_file(args.simultaneous),
        },
        "frozen_method": goodfire_summary["frozen_method"],
        "swd_revision": goodfire_summary["source"]["swd_revision"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goodfire-summary", type=Path, required=True)
    parser.add_argument("--goodfire-points", type=Path, required=True)
    parser.add_argument("--replication", type=Path, action="append", required=True)
    parser.add_argument("--simultaneous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
