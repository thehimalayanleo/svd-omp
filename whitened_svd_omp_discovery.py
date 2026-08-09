"""Discovery sweep for activation-whitened SVD-OMP versus measured SWD.

This script selects regularization on the existing WikiText-2 validation
activations. Its output is discovery evidence only. Freeze any promoted
regularization rule before evaluating on a new test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mdl_svdomp_vs_swd import interpolated_comparison, log2_choose
from mdl_svdomp_vs_swd_natural_24 import load_inputs, shared_log2_advantage
from model_config import TARGET_MODULES


DEFAULT_ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated positive values")
    return values


def load_reference_rows(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            module = raw["module"]
            method = raw["method"]
            if method not in ("svd_omp_shared_dictionary", "swd_dsf"):
                continue
            row: dict[str, Any] = {
                "method": method,
                "relative_error": float(raw["relative_error"]),
                "total_bits": float(raw["total_bits"]),
                "bits_per_token": float(raw["bits_per_token"]),
            }
            if method == "swd_dsf":
                row["sparsity_requested"] = float(raw["sparsity_requested"])
            result.setdefault(module, {}).setdefault(method, []).append(row)
    return result


@torch.no_grad()
def activation_whitened_svd(
    weight: torch.Tensor,
    calibration: torch.Tensor,
    alpha: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    target_device = torch.device(device)
    weight_device = weight.to(target_device)
    h = calibration.to(target_device)
    gram = h.T.matmul(h) / h.shape[0]
    scale = gram.diagonal().mean().clamp_min(1e-30)
    regularization = alpha * scale
    gram = gram + regularization * torch.eye(
        gram.shape[0], device=target_device, dtype=gram.dtype
    )
    chol = torch.linalg.cholesky(gram)
    weighted = weight_device.matmul(chol)
    left, singular_values, vh = torch.linalg.svd(weighted, full_matrices=False)
    read_vectors = torch.linalg.solve_triangular(
        chol.T, vh.T, upper=True
    )
    reconstruction = (left * singular_values.unsqueeze(0)).matmul(read_vectors.T)
    exact_error = float(
        (weight_device - reconstruction).norm()
        / weight_device.norm().clamp_min(1e-30)
    )
    return singular_values, read_vectors, {
        "alpha": alpha,
        "gram_scale": float(scale.item()),
        "regularization": float(regularization.item()),
        "weight_reconstruction_relative_error": exact_error,
    }


@torch.no_grad()
def orthogonal_atom_curve(
    weight: torch.Tensor,
    activations: torch.Tensor,
    singular_values: torch.Tensor,
    read_vectors: torch.Tensor,
    bits_per_value: int,
    device: str,
) -> list[dict[str, Any]]:
    target_device = torch.device(device)
    h = activations.to(target_device)
    singular_values = singular_values.to(target_device)
    read_vectors = read_vectors.to(target_device)
    projections = h.matmul(read_vectors)
    component_energy = projections.square() * singular_values.square().unsqueeze(0)
    sorted_energy = component_energy.sort(dim=1, descending=True).values
    cumulative = sorted_energy.cumsum(dim=1)
    token_energy = component_energy.sum(dim=1)
    direct_energy = h.matmul(weight.T.to(target_device)).square().sum()
    total_energy = token_energy.sum().clamp_min(1e-30)
    energy_error = float((total_energy - direct_energy).abs() / direct_energy.clamp_min(1e-30))
    if energy_error > 1e-3:
        raise RuntimeError(
            f"Orthogonal-component energy check failed: relative gap {energy_error:.3e}"
        )
    rank = singular_values.numel()
    targets = np.unique(
        np.concatenate(
            ([0.0], np.geomspace(1e-4, 0.99, 180), np.linspace(0.01, 1.0, 180))
        )
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for target_error in targets:
        required_energy = token_energy * (1.0 - float(target_error) ** 2)
        support_sizes = (cumulative < required_energy.unsqueeze(1)).sum(dim=1) + 1
        support_sizes = support_sizes.clamp(max=rank)
        support_sizes = torch.where(
            (required_energy <= 0) | (token_energy <= 0),
            torch.zeros_like(support_sizes),
            support_sizes,
        )
        signature = tuple(int(value) for value in support_sizes.cpu().tolist())
        if signature in seen:
            continue
        seen.add(signature)
        captured = torch.zeros_like(token_energy)
        positive = support_sizes > 0
        if positive.any():
            captured[positive] = cumulative[positive].gather(
                1, (support_sizes[positive] - 1).unsqueeze(1)
            ).squeeze(1)
        residual = (token_energy - captured).clamp_min(0).sum()
        relative_error = float(torch.sqrt(residual / total_energy).item())
        transmitted_bits = sum(
            log2_choose(rank, int(k)) + int(k) * bits_per_value
            for k in support_sizes.cpu().tolist()
        )
        rows.append(
            {
                "method": "whitened_svd_omp_shared_dictionary",
                "relative_error": relative_error,
                "total_bits": transmitted_bits,
                "bits_per_token": transmitted_bits / activations.shape[0],
                "mean_components": float(support_sizes.float().mean().item()),
                "per_token_error_target": float(target_error),
            }
        )
    return rows


def main(args: argparse.Namespace) -> None:
    weights, calibration, heldout, metadata = load_inputs(
        args.weights, args.activations
    )
    references = load_reference_rows(args.reference_rows)
    modules = TARGET_MODULES[: args.max_modules] if args.max_modules else TARGET_MODULES
    summaries = []
    for module_index, module in enumerate(modules, start=1):
        print(f"[{module_index:02d}/{len(modules)}] {module}", flush=True)
        swd_rows = references[module]["swd_dsf"]
        baseline_rows = references[module]["svd_omp_shared_dictionary"]
        baseline_advantage = shared_log2_advantage(baseline_rows, swd_rows)
        candidates = []
        for alpha in args.alphas:
            singular_values, read_vectors, diagnostics = activation_whitened_svd(
                weights[module], calibration[module], alpha, args.device
            )
            rows = orthogonal_atom_curve(
                weights[module],
                heldout[module],
                singular_values,
                read_vectors,
                args.bits,
                args.device,
            )
            comparison = interpolated_comparison(rows, swd_rows)
            advantage = shared_log2_advantage(rows, swd_rows)
            candidates.append(
                {
                    **diagnostics,
                    "comparison": comparison,
                    "advantage": advantage,
                }
            )
        best = max(
            candidates,
            key=lambda item: item["advantage"]["mean_log2_swd_over_svd"],
        )
        summaries.append(
            {
                "module": module,
                "family": "attention" if ".attn." in module else "mlp",
                "baseline_mean_log2_swd_over_svd": baseline_advantage[
                    "mean_log2_swd_over_svd"
                ],
                "best_alpha": best["alpha"],
                "best_mean_log2_swd_over_svd": best["advantage"][
                    "mean_log2_swd_over_svd"
                ],
                "best_tightest_winner": best["comparison"]["regions"][0][
                    "winner"
                ],
                "best_full_overlap_winner": (
                    best["comparison"]["regions"][0]["winner"]
                    if len(best["comparison"]["regions"]) == 1
                    else "mixed"
                ),
                "candidates": candidates,
            }
        )
        print(
            f"  best alpha={best['alpha']:.1e}, "
            f"mean log2 advantage={best['advantage']['mean_log2_swd_over_svd']:.3f}, "
            f"tightest={best['comparison']['regions'][0]['winner']}",
            flush=True,
        )
    result = {
        "status": "discovery_only_validation_selected",
        "method": "activation_whitened_svd_omp",
        "alphas": list(args.alphas),
        "bits_per_value": args.bits,
        "activation_metadata": metadata,
        "matrix_count": len(summaries),
        "tightest_svd_wins": sum(
            row["best_tightest_winner"] == "svd_omp" for row in summaries
        ),
        "full_overlap_svd_wins": sum(
            row["best_full_overlap_winner"] == "svd_omp" for row in summaries
        ),
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
    parser.add_argument(
        "--reference-rows",
        type=Path,
        default=Path("results/mdl_natural_24_final/mdl_natural_24_rows.csv"),
    )
    parser.add_argument("--alphas", type=parse_float_tuple, default=DEFAULT_ALPHAS)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/whitened_svd_omp/discovery.json"),
    )
    args = parser.parse_args()
    if args.max_modules is not None and not 1 <= args.max_modules <= len(TARGET_MODULES):
        parser.error(f"--max-modules must be between 1 and {len(TARGET_MODULES)}")
    return args


if __name__ == "__main__":
    main(parse_args())
