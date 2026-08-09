"""Held-out natural-text MDL sweep across all 24 Goodfire matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mdl_svdomp_vs_swd import (
    interpolated_comparison,
    pareto_frontier,
    plot_curves,
    sha256_file,
    svd_omp_curve,
    swd_curve,
)
from model_config import TARGET_MODULES, get_k


DEFAULT_SPARSITIES = (
    0.30,
    0.38,
    0.45,
    0.52,
    0.58,
    0.64,
    0.69,
    0.73,
    0.76,
    0.79,
    0.81,
    0.82,
)


def module_kind(module: str) -> str:
    return ".".join(module.split(".")[-2:])


def safe_name(module: str) -> str:
    return module.replace(".", "_")


def source_sha256(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def load_inputs(
    weights_path: Path, activations_path: Path
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    payload = torch.load(activations_path, map_location="cpu", weights_only=False)
    for key in ("metadata", "calibration", "heldout"):
        if key not in payload:
            raise KeyError(f"Natural activation payload is missing {key!r}")
    calibration = payload["calibration"]
    heldout = payload["heldout"]
    for module in TARGET_MODULES:
        if module not in weights or module not in calibration or module not in heldout:
            raise KeyError(f"Missing aligned tensors for {module}")
        weight = weights[module].detach().float().cpu()
        cal = calibration[module].detach().float().cpu()
        test = heldout[module].detach().float().cpu()
        if cal.shape[1] != weight.shape[1] or test.shape[1] != weight.shape[1]:
            raise ValueError(
                f"Orientation mismatch for {module}: W={tuple(weight.shape)}, "
                f"cal={tuple(cal.shape)}, heldout={tuple(test.shape)}"
            )
        weights[module] = weight
        calibration[module] = cal
        heldout[module] = test
    return weights, calibration, heldout, payload["metadata"]


@torch.no_grad()
def support_adaptivity(
    weight: torch.Tensor,
    activations: torch.Tensor,
    k: int,
    device: str,
    sample_size: int = 256,
) -> dict[str, float | int]:
    target_device = torch.device(device)
    weight = weight.to(target_device)
    activations = activations[:sample_size].to(target_device)
    _, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
    scores = activations.matmul(vh.T).abs() * singular_values.unsqueeze(0)
    supports = scores.topk(k, dim=1).indices.sort(dim=1).values.cpu()
    tuples = [tuple(int(value) for value in row.tolist()) for row in supports]
    unique_count = len(set(tuples))
    pair_jaccards = []
    for index in range(0, len(tuples) - 1, 2):
        left, right = set(tuples[index]), set(tuples[index + 1])
        pair_jaccards.append(len(left & right) / len(left | right))
    return {
        "support_k": k,
        "support_sample_size": len(tuples),
        "unique_supports": unique_count,
        "unique_support_fraction": unique_count / len(tuples),
        "mean_paired_jaccard": float(np.mean(pair_jaccards)),
    }


def shared_log2_advantage(
    omp_rows: list[dict[str, Any]], swd_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    omp = pareto_frontier(omp_rows)
    swd = pareto_frontier(swd_rows)
    omp_x = np.asarray([row["relative_error"] for row in omp], dtype=float)
    omp_y = np.log2(np.maximum([row["total_bits"] for row in omp], 1e-300))
    swd_x = np.asarray([row["relative_error"] for row in swd], dtype=float)
    swd_y = np.log2(np.maximum([row["total_bits"] for row in swd], 1e-300))
    lower = max(float(omp_x.min()), float(swd_x.min()))
    upper = min(float(omp_x.max()), float(swd_x.max()))
    grid = np.linspace(lower, upper, 4097)
    advantage = np.interp(grid, swd_x, swd_y) - np.interp(grid, omp_x, omp_y)
    swd_point_advantages = []
    for row in swd:
        error = float(row["relative_error"])
        if lower <= error <= upper:
            omp_bits = float(2 ** np.interp(error, omp_x, omp_y))
            swd_point_advantages.append(
                {
                    "relative_error": error,
                    "log2_swd_over_svd": math.log2(row["total_bits"] / omp_bits),
                    "winner": "svd_omp" if omp_bits < row["total_bits"] else "swd",
                }
            )
    return {
        "mean_log2_swd_over_svd": float(np.trapezoid(advantage, grid) / (upper - lower)),
        "swd_point_comparisons": swd_point_advantages,
        "swd_points_won_by_svd": sum(
            row["winner"] == "svd_omp" for row in swd_point_advantages
        ),
        "comparable_swd_points": len(swd_point_advantages),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def write_matrix_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "module",
        "kind",
        "d_out",
        "d_in",
        "tightest_winner",
        "loosest_winner",
        "epsilon_star",
        "crossover_count",
        "mean_log2_swd_over_svd",
        "swd_points_won_by_svd",
        "comparable_swd_points",
        "unique_support_fraction",
        "mean_paired_jaccard",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            comparison = summary["shared_dictionary_comparison"]
            advantage = summary["shared_dictionary_advantage"]
            adaptivity = summary["support_adaptivity"]
            crossovers = comparison["crossovers"]
            writer.writerow(
                {
                    "module": summary["module"],
                    "kind": summary["kind"],
                    "d_out": summary["weight_shape"][0],
                    "d_in": summary["weight_shape"][1],
                    "tightest_winner": comparison["regions"][0]["winner"],
                    "loosest_winner": comparison["regions"][-1]["winner"],
                    "epsilon_star": crossovers[0] if len(crossovers) == 1 else "",
                    "crossover_count": len(crossovers),
                    "mean_log2_swd_over_svd": advantage["mean_log2_swd_over_svd"],
                    "swd_points_won_by_svd": advantage["swd_points_won_by_svd"],
                    "comparable_swd_points": advantage["comparable_swd_points"],
                    "unique_support_fraction": adaptivity["unique_support_fraction"],
                    "mean_paired_jaccard": adaptivity["mean_paired_jaccard"],
                }
            )


def plot_aggregate(path: Path, summaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [
        f"L{summary['module'].split('.')[1]} {summary['module'].split('.')[-1]}"
        for summary in summaries
    ]
    advantages = [
        summary["shared_dictionary_advantage"]["mean_log2_swd_over_svd"]
        for summary in summaries
    ]
    colors = ["#0072B2" if value > 0 else "#009E73" for value in advantages]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    bars = axis.bar(range(len(labels)), advantages, color=colors, width=0.78)
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Mean log2 bits(SWD / SVD-OMP) over measured overlap")
    axis.set_xlabel("Goodfire matrix")
    axis.grid(True, axis="y", alpha=0.25)
    axis.set_title("Held-out WikiText-2 shared-dictionary MDL comparison, B=16")
    for bar, summary in zip(bars, summaries):
        crossovers = summary["shared_dictionary_comparison"]["crossovers"]
        if len(crossovers) == 1:
            value = bar.get_height()
            axis.annotate(
                f"epsilon*={crossovers[0]:.3f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4 if value >= 0 else -11),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
                rotation=90,
            )
    axis.text(
        0.01,
        0.98,
        "positive: SVD-OMP fewer bits\nnegative: SWD fewer bits",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_sparsities(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(not 0 <= item < 0.84 for item in result):
        raise argparse.ArgumentTypeError(
            "Use comma-separated sparsities in [0, 0.84); square SWD factors "
            "need at least 16% density"
        )
    return result


def main(args: argparse.Namespace) -> None:
    weights, calibration, heldout, activation_metadata = load_inputs(
        args.weights, args.activations
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    all_rows = []
    summaries = []

    modules = (
        TARGET_MODULES
        if args.max_modules is None
        else TARGET_MODULES[: args.max_modules]
    )
    for index, module in enumerate(modules, start=1):
        weight = weights[module]
        cal = calibration[module]
        test = heldout[module]
        print(
            f"[{index:02d}/{len(modules)}] {module}: "
            f"W={tuple(weight.shape)}, cal={tuple(cal.shape)}, heldout={tuple(test.shape)}"
        )
        omp_rows, omp_diagnostics = svd_omp_curve(
            weight, test, args.bits, device=args.device
        )
        measured_swd = swd_curve(
            weight,
            cal,
            args.swd_source,
            args.sparsities,
            args.bits,
            args.outer_iterations,
            args.device,
            evaluation_activations=test,
        )
        for row in omp_rows + measured_swd:
            row["module"] = module
            row["module_kind"] = module_kind(module)
        all_rows.extend(omp_rows + measured_swd)

        shared = [
            row for row in omp_rows if row["method"] == "svd_omp_shared_dictionary"
        ]
        counted = [
            row for row in omp_rows if row["method"] == "svd_omp_dictionary_counted"
        ]
        shared_comparison = interpolated_comparison(shared, measured_swd)
        counted_comparison = interpolated_comparison(counted, measured_swd)
        if not shared_comparison["regions"]:
            raise RuntimeError(f"No measured curve overlap for {module}")
        summary = {
            "module": module,
            "kind": module_kind(module),
            "weight_shape": list(weight.shape),
            "calibration_shape": list(cal.shape),
            "heldout_shape": list(test.shape),
            "omp_diagnostics": omp_diagnostics,
            "shared_dictionary_comparison": shared_comparison,
            "dictionary_counted_comparison": counted_comparison,
            "shared_dictionary_advantage": shared_log2_advantage(shared, measured_swd),
            "support_adaptivity": support_adaptivity(
                weight, test, get_k(module), args.device
            ),
            "swd_points": [
                {
                    key: row[key]
                    for key in (
                        "sparsity_requested",
                        "relative_error",
                        "calibration_relative_error",
                        "bits_per_token",
                        "active_edges",
                    )
                }
                for row in measured_swd
            ],
        }
        summaries.append(summary)
        plot_curves(
            figure_dir / f"{safe_name(module)}.png",
            omp_rows + measured_swd,
            shared_comparison["crossovers"],
        )

    single_crossovers = [
        summary["shared_dictionary_comparison"]["crossovers"][0]
        for summary in summaries
        if len(summary["shared_dictionary_comparison"]["crossovers"]) == 1
    ]
    tightest_svd_wins = sum(
        summary["shared_dictionary_comparison"]["regions"][0]["winner"] == "svd_omp"
        for summary in summaries
    )
    counted_svd_wins = sum(
        any(
            region["winner"] == "svd_omp"
            for region in summary["dictionary_counted_comparison"]["regions"]
        )
        for summary in summaries
    )
    family_summary = {}
    for family in ("attention", "mlp"):
        selected = [
            summary
            for summary in summaries
            if (".attn." in summary["module"]) == (family == "attention")
        ]
        family_summary[family] = {
            "matrix_count": len(selected),
            "tightest_overlap_svd_wins": sum(
                summary["shared_dictionary_comparison"]["regions"][0]["winner"]
                == "svd_omp"
                for summary in selected
            ),
            "full_overlap_svd_wins": sum(
                {
                    region["winner"]
                    for region in summary["shared_dictionary_comparison"]["regions"]
                }
                == {"svd_omp"}
                for summary in selected
            ),
            "full_overlap_swd_wins": sum(
                {
                    region["winner"]
                    for region in summary["shared_dictionary_comparison"]["regions"]
                }
                == {"swd"}
                for summary in selected
            ),
            "median_mean_log2_swd_over_svd": float(
                np.median(
                    [
                        summary["shared_dictionary_advantage"][
                            "mean_log2_swd_over_svd"
                        ]
                        for summary in selected
                    ]
                )
            ),
        }
    aggregate = {
        "status": "measured_natural_text_heldout",
        "proxy_swd_used": False,
        "matrix_count": len(summaries),
        "bits_per_value": args.bits,
        "swd_sparsities": list(args.sparsities),
        "swd_outer_iterations": args.outer_iterations,
        "calibration_role": "WikiText-2 train activations form the SWD Gram only",
        "reporting_role": "all reported errors and SVD-OMP codes use WikiText-2 validation",
        "weights_sha256": sha256_file(args.weights),
        "activations_sha256": sha256_file(args.activations),
        "benchmark_source_sha256": source_sha256(Path(__file__).resolve()),
        "single_matrix_benchmark_source_sha256": source_sha256(
            Path(__file__).with_name("mdl_svdomp_vs_swd.py")
        ),
        "svd_omp_revision": args.svd_omp_revision,
        "swd_revision": args.swd_revision,
        "activation_metadata": activation_metadata,
        "tightest_overlap_svd_wins": tightest_svd_wins,
        "tightest_overlap_swd_wins": len(summaries) - tightest_svd_wins,
        "single_crossover_matrices": len(single_crossovers),
        "median_epsilon_star": (
            float(np.median(single_crossovers)) if single_crossovers else None
        ),
        "epsilon_star_quartiles": (
            [float(value) for value in np.quantile(single_crossovers, [0.25, 0.75])]
            if single_crossovers
            else None
        ),
        "dictionary_counted_matrices_with_any_svd_win": counted_svd_wins,
        "family_summary": family_summary,
        "median_unique_support_fraction": float(
            np.median(
                [summary["support_adaptivity"]["unique_support_fraction"] for summary in summaries]
            )
        ),
        "median_mean_paired_jaccard": float(
            np.median(
                [summary["support_adaptivity"]["mean_paired_jaccard"] for summary in summaries]
            )
        ),
        "matrices": summaries,
    }

    rows_path = args.output_dir / "mdl_natural_24_rows.csv"
    matrix_path = args.output_dir / "mdl_natural_24_matrix_summary.csv"
    summary_path = args.output_dir / "mdl_natural_24_summary.json"
    figure_path = args.output_dir / "mdl_natural_24_crossovers.png"
    write_rows(rows_path, all_rows)
    write_matrix_summary(matrix_path, summaries)
    summary_path.write_text(json.dumps(aggregate, indent=2))
    plot_aggregate(figure_path, summaries)
    print(json.dumps({key: value for key, value in aggregate.items() if key != "matrices"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=Path("weights/goodfire_67m_weights.pt")
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path("weights/goodfire_67m_natural_24_activations.pt"),
    )
    parser.add_argument("--swd-source", type=Path, required=True)
    parser.add_argument("--svd-omp-revision", required=True)
    parser.add_argument("--swd-revision", required=True)
    parser.add_argument(
        "--sparsities", type=parse_sparsities, default=DEFAULT_SPARSITIES
    )
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=40)
    parser.add_argument(
        "--max-modules",
        type=int,
        default=None,
        help="Run only the first N matrices for a smoke test.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/mdl_natural_24")
    )
    args = parser.parse_args()
    if args.max_modules is not None and not 1 <= args.max_modules <= len(TARGET_MODULES):
        parser.error(f"--max-modules must be between 1 and {len(TARGET_MODULES)}")
    return args


if __name__ == "__main__":
    main(parse_args())
