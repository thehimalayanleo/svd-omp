"""Per-input selected-unit discovery comparison for SVD-OMP and SWD.

SVD-OMP uses exact top-k because its output atoms are orthogonal. SWD receives
a stronger per-input greedy selector over its sparse bottleneck units. The
primary axis is selected units per input; active edges are reported separately.
Validation activations are discovery-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from mdl_svdomp_vs_swd import import_swd_factorizer, sha256_file
from mdl_svdomp_vs_swd_natural_24 import load_inputs
from model_config import TARGET_MODULES
from whitened_svd_omp_discovery import activation_whitened_svd, parse_float_tuple


DEFAULT_KS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
DEFAULT_SPARSITIES = (0.30, 0.45, 0.58, 0.69, 0.76, 0.81, 0.82)


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers")
    return values


@torch.no_grad()
def svd_selected_unit_curve(
    weight: torch.Tensor,
    evaluation: torch.Tensor,
    singular_values: torch.Tensor,
    read_vectors: torch.Tensor,
    ks: tuple[int, ...],
    device: str,
) -> list[dict[str, Any]]:
    target_device = torch.device(device)
    coefficients = (
        evaluation.to(target_device).matmul(read_vectors.to(target_device))
        * singular_values.to(target_device).unsqueeze(0)
    )
    sorted_energy = coefficients.square().sort(dim=1, descending=True).values
    cumulative = sorted_energy.cumsum(dim=1)
    total = sorted_energy.sum().clamp_min(1e-30)
    rank = coefficients.shape[1]
    rows = []
    for k in ks:
        actual_k = min(k, rank)
        captured = cumulative[:, actual_k - 1].sum()
        error = float(torch.sqrt((total - captured).clamp_min(0) / total).item())
        rows.append(
            {
                "method": "whitened_svd_omp",
                "selected_units": actual_k,
                "relative_error": error,
                "mean_active_edges": actual_k * (weight.shape[0] + weight.shape[1]),
            }
        )
    return rows


@torch.no_grad()
def greedy_swd_selected_unit_curve(
    weight: torch.Tensor,
    evaluation: torch.Tensor,
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    ks: tuple[int, ...],
    device: str,
) -> list[dict[str, Any]]:
    target_device = torch.device(device)
    h = evaluation.to(target_device)
    weight_eff = weight.T.to(target_device)
    factor_a = factor_a.to(target_device)
    factor_b = factor_b.to(target_device)
    target = h.matmul(weight_eff)
    coefficients = h.matmul(factor_a)
    atom_gram = factor_b.matmul(factor_b.T)
    correlations = target.matmul(factor_b.T)
    atom_norms = atom_gram.diagonal()
    residual_energy = target.square().sum(dim=1)
    total_energy = residual_energy.sum().clamp_min(1e-30)
    selected = torch.zeros_like(coefficients, dtype=torch.bool)
    edge_cost = factor_a.ne(0).sum(dim=0) + factor_b.ne(0).sum(dim=1)
    cumulative_edges = torch.zeros(
        evaluation.shape[0], device=target_device, dtype=torch.float32
    )
    requested = set(ks)
    rows = []
    max_k = min(max(ks), factor_a.shape[1])
    for step in range(1, max_k + 1):
        gains = (
            2.0 * coefficients * correlations
            - coefficients.square() * atom_norms.unsqueeze(0)
        )
        gains.masked_fill_(selected, -torch.inf)
        best_gain, best_index = gains.max(dim=1)
        active = best_gain > 0
        chosen_coefficients = coefficients.gather(
            1, best_index.unsqueeze(1)
        ).squeeze(1)
        chosen_coefficients = chosen_coefficients * active
        gram_columns = atom_gram[:, best_index].T
        correlations -= chosen_coefficients.unsqueeze(1) * gram_columns
        residual_energy -= torch.where(active, best_gain, torch.zeros_like(best_gain))
        selected.scatter_(1, best_index.unsqueeze(1), active.unsqueeze(1))
        cumulative_edges += edge_cost[best_index].float() * active
        if step in requested:
            rows.append(
                {
                    "method": "swd_dsf_per_input_greedy",
                    "selected_units": step,
                    "relative_error": float(
                        torch.sqrt(residual_energy.clamp_min(0).sum() / total_energy).item()
                    ),
                    "mean_active_edges": float(cumulative_edges.mean().item()),
                    "mean_actual_selected_units": float(selected.sum(dim=1).float().mean().item()),
                }
            )
    return rows


def main(args: argparse.Namespace) -> None:
    weights, calibration, heldout, metadata = load_inputs(
        args.weights, args.activations
    )
    modules = TARGET_MODULES[: args.max_modules] if args.max_modules else TARGET_MODULES
    factorize_matrix = import_swd_factorizer(args.swd_source)
    summaries = []
    for index, module in enumerate(modules, start=1):
        print(f"[{index:02d}/{len(modules)}] {module}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        singular_values, read_vectors, basis_diagnostics = activation_whitened_svd(
            weights[module], calibration[module], args.alpha, args.device
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        svd_seconds = time.perf_counter() - started
        svd_rows = svd_selected_unit_curve(
            weights[module],
            heldout[module],
            singular_values,
            read_vectors,
            args.ks,
            args.device,
        )
        target_device = torch.device(args.device)
        calibration_h = calibration[module].to(target_device)
        gram = calibration_h.T.matmul(calibration_h)
        weight_eff = weights[module].T.contiguous().to(target_device)
        swd_candidates: dict[int, list[dict[str, Any]]] = {
            k: [] for k in args.ks
        }
        swd_factorization_seconds = []
        for sparsity in args.sparsities:
            torch.manual_seed(0)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = factorize_matrix(
                weight_eff,
                gram,
                sparsity,
                outer_iterations=args.outer_iterations,
                final_iterations=20,
                device=target_device,
                capture_stdout=True,
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            swd_factorization_seconds.append(time.perf_counter() - started)
            rows = greedy_swd_selected_unit_curve(
                weights[module],
                heldout[module],
                result.factor_a,
                result.factor_b,
                args.ks,
                args.device,
            )
            for row in rows:
                row["sparsity_requested"] = sparsity
                swd_candidates[row["selected_units"]].append(row)
        swd_rows = [
            min(swd_candidates[k], key=lambda row: row["relative_error"])
            for k in args.ks
        ]
        comparisons = []
        for svd_row, swd_row in zip(svd_rows, swd_rows, strict=True):
            comparisons.append(
                {
                    "selected_units": svd_row["selected_units"],
                    "winner": (
                        "svd_omp"
                        if svd_row["relative_error"] < swd_row["relative_error"]
                        else "swd"
                    ),
                    "svd_relative_error": svd_row["relative_error"],
                    "swd_relative_error": swd_row["relative_error"],
                    "svd_mean_active_edges": svd_row["mean_active_edges"],
                    "swd_mean_active_edges": swd_row["mean_active_edges"],
                    "swd_best_sparsity": swd_row["sparsity_requested"],
                }
            )
        summary = {
            "module": module,
            "family": "attention" if ".attn." in module else "mlp",
            "basis_diagnostics": basis_diagnostics,
            "svd_decomposition_seconds": svd_seconds,
            "swd_factorization_seconds": swd_factorization_seconds,
            "comparisons": comparisons,
        }
        summaries.append(summary)
        wins = sum(row["winner"] == "svd_omp" for row in comparisons)
        print(f"  SVD-OMP wins {wins}/{len(comparisons)} selected-unit points", flush=True)
    result = {
        "status": args.evaluation_status,
        "primary_axis": "heldout_relative_output_error_at_equal_per_input_selected_units",
        "alpha": args.alpha,
        "ks": list(args.ks),
        "swd_sparsities": list(args.sparsities),
        "swd_outer_iterations": args.outer_iterations,
        "weights_sha256": sha256_file(args.weights),
        "activations_sha256": sha256_file(args.activations),
        "benchmark_source_sha256": source_sha256(Path(__file__).resolve()),
        "whitened_basis_source_sha256": source_sha256(
            Path(__file__).with_name("whitened_svd_omp_discovery.py")
        ),
        "svd_omp_revision": args.svd_omp_revision,
        "swd_revision": args.swd_revision,
        "activation_metadata": metadata,
        "matrix_count": len(summaries),
        "svd_point_wins": sum(
            comparison["winner"] == "svd_omp"
            for summary in summaries
            for comparison in summary["comparisons"]
        ),
        "total_points": len(summaries) * len(args.ks),
        "matrices": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "matrices"}, indent=2))


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
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ks", type=parse_int_tuple, default=DEFAULT_KS)
    parser.add_argument(
        "--sparsities", type=parse_float_tuple, default=DEFAULT_SPARSITIES
    )
    parser.add_argument("--outer-iterations", type=int, default=40)
    parser.add_argument(
        "--svd-omp-revision",
        default="a022240f2758b5c52755965fb77c49aaec88e988",
    )
    parser.add_argument(
        "--swd-revision",
        default="4c44b7281bc7c78f80e431dac3aa75f397dd3043",
    )
    parser.add_argument(
        "--evaluation-status",
        default="discovery_only_validation_selected",
    )
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/selected_units/discovery.json"),
    )
    args = parser.parse_args()
    if args.max_modules is not None and not 1 <= args.max_modules <= len(TARGET_MODULES):
        parser.error(f"--max-modules must be between 1 and {len(TARGET_MODULES)}")
    return args


if __name__ == "__main__":
    main(parse_args())
