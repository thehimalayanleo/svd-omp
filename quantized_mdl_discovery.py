"""Matched quantized rate-distortion discovery for whitened SVD-OMP and SWD.

Both methods use symmetric tensor-wise quantization and pay for the scale.
Validation activations are used for discovery, so promoted settings require a
new held-out test split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mdl_svdomp_vs_swd import (
    import_swd_factorizer,
    interpolated_comparison,
    log2_choose,
    sparse_array_code_bits,
)
from mdl_svdomp_vs_swd_natural_24 import load_inputs, shared_log2_advantage
from model_config import TARGET_MODULES
from whitened_svd_omp_discovery import activation_whitened_svd, parse_float_tuple


DEFAULT_BITS = (3, 4, 6, 8, 12, 16)
DEFAULT_SPARSITIES = (0.30, 0.45, 0.58, 0.69, 0.76, 0.81, 0.82)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item < 2 or item > 16 for item in values):
        raise argparse.ArgumentTypeError("Expected comma-separated bit widths in [2, 16]")
    return values


def symmetric_quantize(
    values: torch.Tensor, bits: int, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    if scale is None:
        scale = values.abs().max().clamp_min(1e-30) / qmax
    quantized = (values / scale).round().clamp(-qmax, qmax) * scale
    return quantized, scale


def structured_sparse_code_bits(values: torch.Tensor, bits: int) -> float:
    rows, columns = values.shape
    nnz = int(values.ne(0).sum().item())
    global_bits = sparse_array_code_bits(values.numel(), nnz, bits)
    row_counts = values.ne(0).sum(dim=1).cpu().tolist()
    row_bits = rows * math.ceil(math.log2(columns + 1)) + sum(
        log2_choose(columns, int(count)) for count in row_counts
    ) + nnz * bits
    column_counts = values.ne(0).sum(dim=0).cpu().tolist()
    column_bits = columns * math.ceil(math.log2(rows + 1)) + sum(
        log2_choose(rows, int(count)) for count in column_counts
    ) + nnz * bits
    return min(global_bits, row_bits, column_bits)


@torch.no_grad()
def quantized_atom_curve(
    weight: torch.Tensor,
    calibration: torch.Tensor,
    evaluation: torch.Tensor,
    singular_values: torch.Tensor,
    read_vectors: torch.Tensor,
    bit_widths: tuple[int, ...],
    device: str,
) -> list[dict[str, Any]]:
    target_device = torch.device(device)
    singular_values = singular_values.to(target_device)
    read_vectors = read_vectors.to(target_device)
    calibration_coefficients = (
        calibration.to(target_device).matmul(read_vectors)
        * singular_values.unsqueeze(0)
    )
    coefficients = (
        evaluation.to(target_device).matmul(read_vectors)
        * singular_values.unsqueeze(0)
    )
    component_energy = coefficients.square()
    sorted_indices = component_energy.argsort(dim=1, descending=True)
    sorted_energy = component_energy.gather(1, sorted_indices)
    cumulative = sorted_energy.cumsum(dim=1)
    token_energy = component_energy.sum(dim=1)
    total_energy = token_energy.sum().clamp_min(1e-30)
    direct_energy = evaluation.to(target_device).matmul(
        weight.T.to(target_device)
    ).square().sum()
    if float((total_energy - direct_energy).abs() / direct_energy) > 1e-3:
        raise RuntimeError("Whitened coefficient energy does not match direct output")
    targets = np.unique(
        np.concatenate(
            ([0.0], np.geomspace(1e-4, 0.99, 120), np.linspace(0.01, 0.99, 120))
        )
    )
    rank = coefficients.shape[1]
    calibration_energy = calibration_coefficients.square()
    calibration_sorted_indices = calibration_energy.argsort(dim=1, descending=True)
    calibration_sorted_energy = calibration_energy.gather(
        1, calibration_sorted_indices
    )
    calibration_cumulative = calibration_sorted_energy.cumsum(dim=1)
    calibration_token_energy = calibration_energy.sum(dim=1)
    rows: list[dict[str, Any]] = []
    for bits in bit_widths:
        qmax = 2 ** (bits - 1) - 1
        scale = (
            calibration_coefficients.abs().amax(dim=0).clamp_min(1e-30) / qmax
        )
        quantized_coefficients, _ = symmetric_quantize(
            coefficients, bits, scale.unsqueeze(0)
        )
        quantized_calibration, _ = symmetric_quantize(
            calibration_coefficients, bits, scale.unsqueeze(0)
        )
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
            selected = torch.zeros_like(coefficients, dtype=torch.bool)
            positions = torch.arange(rank, device=target_device).unsqueeze(0)
            sorted_mask = positions < support_sizes.unsqueeze(1)
            selected.scatter_(1, sorted_indices, sorted_mask)
            reconstructed = torch.zeros_like(coefficients)
            reconstructed[selected] = quantized_coefficients[selected]
            residual = (coefficients - reconstructed).square().sum()
            relative_error = float(torch.sqrt(residual / total_energy).item())
            nonzeros = reconstructed.ne(0).sum(dim=1).cpu().tolist()
            active_scales = int(reconstructed.ne(0).any(dim=0).sum().item())
            scale_bits = 16 * active_scales
            value_bits = sum(int(k) * bits for k in nonzeros)
            universal_support_bits = sum(
                log2_choose(rank, int(k)) for k in nonzeros
            )
            universal_total_bits = scale_bits + universal_support_bits + value_bits

            calibration_required = calibration_token_energy * (
                1.0 - float(target_error) ** 2
            )
            calibration_sizes = (
                calibration_cumulative < calibration_required.unsqueeze(1)
            ).sum(dim=1) + 1
            calibration_sizes = calibration_sizes.clamp(max=rank)
            calibration_sizes = torch.where(
                (calibration_required <= 0) | (calibration_token_energy <= 0),
                torch.zeros_like(calibration_sizes),
                calibration_sizes,
            )
            calibration_selected = torch.zeros_like(
                calibration_coefficients, dtype=torch.bool
            )
            calibration_sorted_mask = (
                positions < calibration_sizes.unsqueeze(1)
            )
            calibration_selected.scatter_(
                1, calibration_sorted_indices, calibration_sorted_mask
            )
            calibration_selected &= quantized_calibration.ne(0)
            counts = calibration_selected.sum(dim=0).float()
            probabilities = (counts + 0.5) / (calibration.shape[0] + 1.0)
            encoded_mask = reconstructed.ne(0)
            entropy_support_bits = float(
                -torch.where(
                    encoded_mask,
                    probabilities.log2().unsqueeze(0),
                    (1.0 - probabilities).log2().unsqueeze(0),
                ).sum().item()
            )
            support_model_bits = rank * math.ceil(
                math.log2(calibration.shape[0] + 2)
            )
            entropy_total_bits = (
                scale_bits
                + support_model_bits
                + entropy_support_bits
                + value_bits
            )
            common = {
                "relative_error": relative_error,
                "bits_per_token": None,
                "coefficient_bits": bits,
                "scale_bits": scale_bits,
                "active_component_scales": active_scales,
                "mean_nonzero_components": float(np.mean(nonzeros)),
                "per_token_error_target": float(target_error),
            }
            rows.append(
                {
                    **common,
                    "method": "whitened_svd_omp_quantized_universal_support",
                    "support_code": "enumerative_log2_choose",
                    "support_bits": universal_support_bits,
                    "support_model_bits": 0,
                    "total_bits": universal_total_bits,
                    "bits_per_token": universal_total_bits / evaluation.shape[0],
                }
            )
            rows.append(
                {
                    **common,
                    "method": "whitened_svd_omp_quantized_entropy_support",
                    "support_code": "calibrated_independent_bernoulli",
                    "support_bits": entropy_support_bits,
                    "support_model_bits": support_model_bits,
                    "total_bits": entropy_total_bits,
                    "bits_per_token": entropy_total_bits / evaluation.shape[0],
                }
            )
    return rows


def balance_factors(
    factor_a: torch.Tensor, factor_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    max_a = factor_a.abs().max().clamp_min(1e-30)
    max_b = factor_b.abs().max().clamp_min(1e-30)
    multiplier = torch.sqrt(max_b / max_a)
    return factor_a * multiplier, factor_b / multiplier


@torch.no_grad()
def quantized_swd_curve(
    weight: torch.Tensor,
    calibration: torch.Tensor,
    evaluation: torch.Tensor,
    swd_source: Path,
    sparsities: tuple[float, ...],
    bit_widths: tuple[int, ...],
    outer_iterations: int,
    device: str,
) -> list[dict[str, Any]]:
    factorize_matrix = import_swd_factorizer(swd_source)
    target_device = torch.device(device)
    weight_eff = weight.T.contiguous().to(target_device)
    calibration_h = calibration.to(target_device)
    evaluation_h = evaluation.to(target_device)
    gram = calibration_h.T.matmul(calibration_h)
    target = evaluation_h.matmul(weight_eff)
    target_norm = target.norm().clamp_min(1e-30)
    rows = []
    for sparsity in sparsities:
        torch.manual_seed(0)
        result = factorize_matrix(
            weight_eff,
            gram,
            sparsity,
            outer_iterations=outer_iterations,
            final_iterations=20,
            device=target_device,
            capture_stdout=True,
        )
        factor_a, factor_b = balance_factors(
            result.factor_a.to(target_device), result.factor_b.to(target_device)
        )
        for bits in bit_widths:
            qmax = 2 ** (bits - 1) - 1
            scale_a = factor_a.abs().amax(dim=0, keepdim=True).clamp_min(1e-30) / qmax
            scale_b = factor_b.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / qmax
            quant_a, _ = symmetric_quantize(factor_a, bits, scale_a)
            quant_b, _ = symmetric_quantize(factor_b, bits, scale_b)
            reconstructed = quant_a.matmul(quant_b)
            relative_error = float(
                (target - evaluation_h.matmul(reconstructed)).norm() / target_norm
            )
            nnz_a = int(quant_a.ne(0).sum().item())
            nnz_b = int(quant_b.ne(0).sum().item())
            active_a_scales = int(quant_a.ne(0).any(dim=0).sum().item())
            active_b_scales = int(quant_b.ne(0).any(dim=1).sum().item())
            scale_bits = 16 * (active_a_scales + active_b_scales)
            total_bits = (
                structured_sparse_code_bits(quant_a, bits)
                + structured_sparse_code_bits(quant_b, bits)
                + scale_bits
            )
            rows.append(
                {
                    "method": "swd_dsf_quantized",
                    "relative_error": relative_error,
                    "total_bits": total_bits,
                    "bits_per_token": total_bits / evaluation.shape[0],
                    "factor_bits": bits,
                    "scale_bits": scale_bits,
                    "active_a_scales": active_a_scales,
                    "active_b_scales": active_b_scales,
                    "sparsity_requested": sparsity,
                    "factor_a_nnz": nnz_a,
                    "factor_b_nnz": nnz_b,
                    "support_code": "best_of_global_row_or_column_enumerative",
                }
            )
    return rows


def main(args: argparse.Namespace) -> None:
    weights, calibration, heldout, metadata = load_inputs(
        args.weights, args.activations
    )
    modules = TARGET_MODULES[: args.max_modules] if args.max_modules else TARGET_MODULES
    summaries = []
    for index, module in enumerate(modules, start=1):
        print(f"[{index:02d}/{len(modules)}] {module}", flush=True)
        singular_values, read_vectors, diagnostics = activation_whitened_svd(
            weights[module], calibration[module], args.alpha, args.device
        )
        omp_rows = quantized_atom_curve(
            weights[module],
            calibration[module],
            heldout[module],
            singular_values,
            read_vectors,
            args.bit_widths,
            args.device,
        )
        swd_rows = quantized_swd_curve(
            weights[module],
            calibration[module],
            heldout[module],
            args.swd_source,
            args.sparsities,
            args.bit_widths,
            args.outer_iterations,
            args.device,
        )
        comparison = interpolated_comparison(omp_rows, swd_rows)
        advantage = shared_log2_advantage(omp_rows, swd_rows)
        summary = {
            "module": module,
            "family": "attention" if ".attn." in module else "mlp",
            "basis_diagnostics": diagnostics,
            "comparison": comparison,
            "advantage": advantage,
            "omp_rows": omp_rows,
            "swd_rows": swd_rows,
        }
        summaries.append(summary)
        print(
            f"  mean log2 advantage={advantage['mean_log2_swd_over_svd']:.3f}, "
            f"tightest={comparison['regions'][0]['winner']}",
            flush=True,
        )
    result = {
        "status": "discovery_only_validation_selected",
        "method": "matched_tensorwise_quantized_rate_distortion",
        "alpha": args.alpha,
        "bit_widths": list(args.bit_widths),
        "swd_sparsities": list(args.sparsities),
        "swd_outer_iterations": args.outer_iterations,
        "activation_metadata": metadata,
        "matrix_count": len(summaries),
        "tightest_svd_wins": sum(
            row["comparison"]["regions"][0]["winner"] == "svd_omp"
            for row in summaries
        ),
        "full_overlap_svd_wins": sum(
            {region["winner"] for region in row["comparison"]["regions"]}
            == {"svd_omp"}
            for row in summaries
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
    parser.add_argument("--swd-source", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--bit-widths", type=parse_int_tuple, default=DEFAULT_BITS)
    parser.add_argument(
        "--sparsities", type=parse_float_tuple, default=DEFAULT_SPARSITIES
    )
    parser.add_argument("--outer-iterations", type=int, default=40)
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/quantized_mdl/discovery.json"),
    )
    args = parser.parse_args()
    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.max_modules is not None and not 1 <= args.max_modules <= len(TARGET_MODULES):
        parser.error(f"--max-modules must be between 1 and {len(TARGET_MODULES)}")
    return args


if __name__ == "__main__":
    main(parse_args())
