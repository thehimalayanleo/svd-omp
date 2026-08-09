"""Summarize and plot a sealed selected-unit SVD-OMP versus SWD result."""

from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(args: argparse.Namespace) -> None:
    payload = json.loads(args.input.read_text())
    if not str(payload["status"]).startswith("sealed_"):
        raise ValueError("Refusing to present a non-sealed result as final")
    rows = []
    speed_ratios = []
    for matrix in payload["matrices"]:
        median_swd_seconds = float(np.median(matrix["swd_factorization_seconds"]))
        speed_ratios.append(median_swd_seconds / matrix["svd_decomposition_seconds"])
        for comparison in matrix["comparisons"]:
            rows.append(
                {
                    "module": matrix["module"],
                    "family": matrix["family"],
                    **comparison,
                    "swd_error_over_svd": (
                        comparison["swd_relative_error"]
                        / comparison["svd_relative_error"]
                    ),
                    "svd_edges_over_swd": (
                        comparison["svd_mean_active_edges"]
                        / comparison["swd_mean_active_edges"]
                    ),
                }
            )
    error_ratios = np.asarray([row["swd_error_over_svd"] for row in rows])
    edge_ratios = np.asarray([row["svd_edges_over_swd"] for row in rows])
    family_summary = {}
    for family in ("attention", "mlp"):
        selected = [row for row in rows if row["family"] == family]
        ratios = np.asarray([row["swd_error_over_svd"] for row in selected])
        family_summary[family] = {
            "points": len(selected),
            "svd_wins": sum(row["winner"] == "svd_omp" for row in selected),
            "geometric_mean_swd_error_over_svd": float(
                np.exp(np.log(ratios).mean())
            ),
            "minimum_swd_error_over_svd": float(ratios.min()),
        }
    by_k = []
    for k in payload["ks"]:
        selected = [row for row in rows if row["selected_units"] == k]
        ratios = np.asarray([row["swd_error_over_svd"] for row in selected])
        edges = np.asarray([row["svd_edges_over_swd"] for row in selected])
        by_k.append(
            {
                "selected_units": k,
                "median_swd_error_over_svd": float(np.median(ratios)),
                "error_ratio_q1": float(np.quantile(ratios, 0.25)),
                "error_ratio_q3": float(np.quantile(ratios, 0.75)),
                "median_svd_edges_over_swd": float(np.median(edges)),
                "edge_ratio_q1": float(np.quantile(edges, 0.25)),
                "edge_ratio_q3": float(np.quantile(edges, 0.75)),
            }
        )
    summary = {
        "status": payload["status"],
        "matrix_count": payload["matrix_count"],
        "point_count": len(rows),
        "svd_wins": sum(row["winner"] == "svd_omp" for row in rows),
        "geometric_mean_swd_error_over_svd": float(
            np.exp(np.log(error_ratios).mean())
        ),
        "minimum_swd_error_over_svd": float(error_ratios.min()),
        "median_svd_edges_over_swd": float(np.median(edge_ratios)),
        "median_swd_factorization_time_over_svd": float(np.median(speed_ratios)),
        "family_summary": family_summary,
        "by_selected_units": by_k,
        "source": {
            key: payload[key]
            for key in (
                "weights_sha256",
                "activations_sha256",
                "benchmark_source_sha256",
                "whitened_basis_source_sha256",
                "svd_omp_revision",
                "swd_revision",
            )
        },
    }
    if args.model_eval is not None:
        model_payload = json.loads(args.model_eval.read_text())
        if model_payload["status"] != "sealed_wikitext_test_full_model_single_matrix_replacement":
            raise ValueError("Refusing to summarize a non-sealed full-model result")
        model_rows = model_payload["matrices"]
        kl_ratios = np.asarray(
            [row["swd"]["kl_to_dense"] / row["svd"]["kl_to_dense"] for row in model_rows]
        )
        mse_ratios = np.asarray(
            [row["swd"]["logit_mse"] / row["svd"]["logit_mse"] for row in model_rows]
        )
        summary["full_model_single_matrix_replacement"] = {
            "status": model_payload["status"],
            "matrix_count": len(model_rows),
            "cross_entropy_svd_wins": sum(
                row["svd"]["cross_entropy"] < row["swd"]["cross_entropy"]
                for row in model_rows
            ),
            "kl_svd_wins": sum(row["kl_winner"] == "svd_omp" for row in model_rows),
            "logit_mse_svd_wins": sum(
                row["logit_mse_winner"] == "svd_omp" for row in model_rows
            ),
            "geometric_mean_swd_kl_over_svd": float(np.exp(np.log(kl_ratios).mean())),
            "minimum_swd_kl_over_svd": float(kl_ratios.min()),
            "geometric_mean_swd_logit_mse_over_svd": float(
                np.exp(np.log(mse_ratios).mean())
            ),
            "minimum_swd_logit_mse_over_svd": float(mse_ratios.min()),
            "dense_cross_entropy": model_payload["dense"]["cross_entropy"],
            "test_metadata": model_payload["test_metadata"],
            "result_sha256": sha256_file(args.model_eval),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "by_selected_units"},
                indent=2,
            )
        )
        return

    ks = np.asarray([row["selected_units"] for row in by_k])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    error_median = np.asarray([row["median_swd_error_over_svd"] for row in by_k])
    error_q1 = np.asarray([row["error_ratio_q1"] for row in by_k])
    error_q3 = np.asarray([row["error_ratio_q3"] for row in by_k])
    axes[0].plot(ks, error_median, marker="o", color="#0072B2")
    axes[0].fill_between(ks, error_q1, error_q3, color="#0072B2", alpha=0.2)
    axes[0].axhline(1.0, color="black", linewidth=1)
    axes[0].set_title("Fidelity at equal selected units")
    axes[0].set_xlabel("Selected units per input")
    axes[0].set_ylabel("SWD error / SVD-OMP error")
    axes[0].text(0.03, 0.95, "above 1: SVD-OMP wins", transform=axes[0].transAxes, va="top")

    edge_median = np.asarray([row["median_svd_edges_over_swd"] for row in by_k])
    edge_q1 = np.asarray([row["edge_ratio_q1"] for row in by_k])
    edge_q3 = np.asarray([row["edge_ratio_q3"] for row in by_k])
    axes[1].plot(ks, edge_median, marker="o", color="#009E73")
    axes[1].fill_between(ks, edge_q1, edge_q3, color="#009E73", alpha=0.2)
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_title("Active-edge cost")
    axes[1].set_xlabel("Selected units per input")
    axes[1].set_ylabel("SVD-OMP edges / SWD edges")
    axes[1].text(0.03, 0.95, "above 1: SWD wins", transform=axes[1].transAxes, va="top")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle("Sealed WikiText-2 test, 24 Goodfire matrices")
    figure.tight_layout()
    figure.savefig(args.output_dir / "selected_unit_tradeoff.png", dpi=180)
    plt.close(figure)
    print(json.dumps({key: value for key, value in summary.items() if key != "by_selected_units"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-eval", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
