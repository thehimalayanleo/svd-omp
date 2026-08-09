"""Measured MDL comparison for per-input SVD-OMP and fixed-support SWD.

The default run uses the Goodfire h.2.mlp.c_fc matrix because
modal_goodfire.py captures the inputs to that exact module. SWD points must come
from the public DSF reference implementation. There is no proxy SWD curve.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_MODULE = "h.2.mlp.c_fc"
DEFAULT_SPARSITIES = (0.30, 0.50, 0.65, 0.75, 0.82, 0.87, 0.90)


def log2_choose(n: int, k: int) -> float:
    if not 0 <= k <= n:
        raise ValueError(f"Expected 0 <= k <= n, got k={k}, n={n}")
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    ) / math.log(2.0)


def sparse_array_code_bits(numel: int, nnz: int, bits_per_value: int) -> float:
    return log2_choose(numel, nnz) + nnz * bits_per_value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_goodfire(
    weights_path: Path,
    activations_path: Path,
    module: str,
    max_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(weights, dict) or module not in weights:
        available = sorted(weights) if isinstance(weights, dict) else []
        raise KeyError(f"Module {module!r} missing from weights. Available: {available}")
    weight = weights[module].detach().float().cpu()
    activations = torch.load(
        activations_path, map_location="cpu", weights_only=False
    ).detach().float().cpu()
    if activations.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            f"Expected H and W to be matrices, got {activations.shape}, {weight.shape}"
        )
    if activations.shape[1] != weight.shape[1]:
        raise ValueError(
            "Orientation mismatch: H must have d_in columns for W[d_out, d_in], "
            f"got H={tuple(activations.shape)}, W={tuple(weight.shape)}"
        )
    if max_tokens is not None:
        activations = activations[:max_tokens]
    if activations.shape[0] == 0:
        raise ValueError("The calibration activation matrix is empty")
    return weight, activations


@torch.no_grad()
def svd_omp_curve(
    weight: torch.Tensor,
    activations: torch.Tensor,
    bits_per_value: int,
    per_token_targets: np.ndarray | None = None,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_device = torch.device(device)
    weight_device = weight.to(target_device)
    h = activations.to(target_device)
    _, singular_values, vh = torch.linalg.svd(weight_device, full_matrices=False)

    projections = h.matmul(vh.T)
    component_energy = projections.square() * singular_values.square().unsqueeze(0)
    sorted_energy = component_energy.sort(dim=1, descending=True).values
    cumulative = sorted_energy.cumsum(dim=1)
    token_energy = component_energy.sum(dim=1)
    total_energy = token_energy.sum().clamp_min(1e-30)
    rank = int(singular_values.numel())

    if per_token_targets is None:
        per_token_targets = np.unique(
            np.concatenate(
                ([0.0], np.geomspace(1e-4, 0.99, 180), np.linspace(0.01, 1.0, 180))
            )
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for target_error in per_token_targets:
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

        sizes = support_sizes.cpu().tolist()
        transmitted_bits = sum(
            log2_choose(rank, int(k)) + int(k) * bits_per_value for k in sizes
        )
        rows.append(
            {
                "method": "svd_omp_shared_dictionary",
                "relative_error": relative_error,
                "total_bits": transmitted_bits,
                "bits_per_token": transmitted_bits / len(sizes),
                "mean_components": float(support_sizes.float().mean().item()),
                "min_components": int(support_sizes.min().item()),
                "max_components": int(support_sizes.max().item()),
                "mean_active_edges": float(
                    support_sizes.float().mean().item()
                    * (weight.shape[0] + weight.shape[1])
                ),
                "per_token_error_target": float(target_error),
            }
        )

    dictionary_bits = (
        rank * weight.shape[1] + rank * weight.shape[0] + rank
    ) * bits_per_value
    counted_rows = []
    for row in rows:
        counted = dict(row)
        counted["method"] = "svd_omp_dictionary_counted"
        counted["total_bits"] = row["total_bits"] + dictionary_bits
        counted["bits_per_token"] = counted["total_bits"] / activations.shape[0]
        counted_rows.append(counted)

    diagnostics = {
        "rank": rank,
        "dictionary_bits": dictionary_bits,
        "output_energy": float(total_energy.item()),
    }
    return rows + counted_rows, diagnostics


def import_swd_factorizer(swd_source: Path):
    package_root = swd_source / "src"
    if not (package_root / "swd" / "factorization.py").exists():
        raise FileNotFoundError(
            f"Expected an SWD checkout with src/swd/factorization.py at {swd_source}"
        )
    sys.path.insert(0, str(package_root))
    try:
        from swd.factorization import factorize_matrix
    except Exception as exc:
        raise RuntimeError(
            "Could not import the SWD reference factorizer. Install its declared "
            "dependencies or point --swd-source at a valid checkout."
        ) from exc
    return factorize_matrix


@torch.no_grad()
def swd_curve(
    weight: torch.Tensor,
    activations: torch.Tensor,
    swd_source: Path,
    sparsities: tuple[float, ...],
    bits_per_value: int,
    outer_iterations: int,
    device: str,
    evaluation_activations: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    factorize_matrix = import_swd_factorizer(swd_source)
    target_device = torch.device(device)
    calibration_h = activations.to(target_device)
    evaluation_h = (
        calibration_h
        if evaluation_activations is None
        else evaluation_activations.to(target_device)
    )
    if calibration_h.shape[1] != evaluation_h.shape[1]:
        raise ValueError(
            "Calibration and evaluation activations must have the same input dimension"
        )
    effective_weight = weight.T.contiguous().to(target_device)
    gram = calibration_h.T.matmul(calibration_h)
    calibration_target = calibration_h.matmul(effective_weight)
    evaluation_target = evaluation_h.matmul(effective_weight)
    calibration_norm = calibration_target.norm().clamp_min(1e-30)
    evaluation_norm = evaluation_target.norm().clamp_min(1e-30)

    rows = []
    for sparsity in sparsities:
        torch.manual_seed(0)
        result = factorize_matrix(
            effective_weight,
            gram,
            sparsity,
            outer_iterations=outer_iterations,
            final_iterations=20,
            device=target_device,
            capture_stdout=True,
        )
        factor_a = result.factor_a.to(target_device)
        factor_b = result.factor_b.to(target_device)
        reconstructed = factor_a.matmul(factor_b)
        calibration_error = float(
            (calibration_target - calibration_h.matmul(reconstructed)).norm()
            / calibration_norm
        )
        evaluation_error = float(
            (evaluation_target - evaluation_h.matmul(reconstructed)).norm()
            / evaluation_norm
        )
        objective_error = math.sqrt(max(float(result.objective_error), 0.0))
        tolerance = max(5e-5, 2e-3 * calibration_error)
        if abs(calibration_error - objective_error) > tolerance:
            raise RuntimeError(
                "SWD orientation check failed: direct relative error "
                f"{calibration_error:.8f}, objective relative error "
                f"{objective_error:.8f}"
            )

        k_a = int(factor_a.ne(0).sum().item())
        k_b = int(factor_b.ne(0).sum().item())
        total_bits = sparse_array_code_bits(
            factor_a.numel(), k_a, bits_per_value
        ) + sparse_array_code_bits(factor_b.numel(), k_b, bits_per_value)
        rows.append(
            {
                "method": "swd_dsf",
                "relative_error": evaluation_error,
                "calibration_relative_error": calibration_error,
                "total_bits": total_bits,
                "bits_per_token": total_bits / evaluation_h.shape[0],
                "sparsity_requested": sparsity,
                "factor_a_shape": list(factor_a.shape),
                "factor_b_shape": list(factor_b.shape),
                "factor_a_nnz": k_a,
                "factor_b_nnz": k_b,
                "active_edges": k_a + k_b,
                "objective_relative_error": objective_error,
            }
        )
    return rows


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["relative_error"], row["total_bits"]))
    frontier = []
    best_bits = math.inf
    for row in ordered:
        if row["total_bits"] < best_bits:
            frontier.append(row)
            best_bits = row["total_bits"]
    return frontier


def dominance_intervals(
    omp_rows: list[dict[str, Any]], swd_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    omp = pareto_frontier(omp_rows)
    swd = pareto_frontier(swd_rows)
    thresholds = sorted(
        {row["relative_error"] for row in omp + swd}
    )
    comparisons = []
    for threshold in thresholds:
        omp_bits = min(
            (row["total_bits"] for row in omp if row["relative_error"] <= threshold),
            default=math.inf,
        )
        swd_bits = min(
            (row["total_bits"] for row in swd if row["relative_error"] <= threshold),
            default=math.inf,
        )
        if math.isfinite(omp_bits) and math.isfinite(swd_bits):
            comparisons.append(
                {
                    "epsilon": threshold,
                    "winner": "svd_omp" if omp_bits < swd_bits else "swd",
                    "svd_omp_bits": omp_bits,
                    "swd_bits": swd_bits,
                }
            )

    intervals = []
    for row in comparisons:
        if not intervals or intervals[-1]["winner"] != row["winner"]:
            intervals.append(
                {
                    "winner": row["winner"],
                    "epsilon_min": row["epsilon"],
                    "epsilon_max": row["epsilon"],
                }
            )
        else:
            intervals[-1]["epsilon_max"] = row["epsilon"]
    return intervals


def interpolated_comparison(
    omp_rows: list[dict[str, Any]], swd_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare the piecewise-linear log-bit curves within measured overlap.

    The figure connects measured frontier points, so this is the matching
    crossover summary. It never extrapolates SWD beyond its measured errors.
    """
    omp = pareto_frontier(omp_rows)
    swd = pareto_frontier(swd_rows)
    omp_x = np.asarray([row["relative_error"] for row in omp], dtype=float)
    omp_y = np.log2(
        np.maximum(
            [row["total_bits"] for row in omp], np.finfo(float).tiny
        )
    )
    swd_x = np.asarray([row["relative_error"] for row in swd], dtype=float)
    swd_y = np.log2([row["total_bits"] for row in swd])
    lower = max(float(omp_x.min()), float(swd_x.min()))
    upper = min(float(omp_x.max()), float(swd_x.max()))
    if lower >= upper:
        return {"measured_overlap": None, "crossovers": [], "regions": []}

    grid = np.unique(
        np.concatenate(
            (
                np.linspace(lower, upper, 16385),
                omp_x[(omp_x >= lower) & (omp_x <= upper)],
                swd_x[(swd_x >= lower) & (swd_x <= upper)],
            )
        )
    )
    delta = np.interp(grid, omp_x, omp_y) - np.interp(grid, swd_x, swd_y)
    roots = []
    for index in range(len(grid) - 1):
        left, right = float(delta[index]), float(delta[index + 1])
        if left == 0:
            roots.append(float(grid[index]))
        elif left * right < 0:
            fraction = -left / (right - left)
            roots.append(float(grid[index] + fraction * (grid[index + 1] - grid[index])))
    if delta[-1] == 0:
        roots.append(float(grid[-1]))
    deduplicated = []
    for root in roots:
        if not deduplicated or abs(root - deduplicated[-1]) > 1e-7:
            deduplicated.append(root)

    boundaries = [lower, *deduplicated, upper]
    regions = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        midpoint = 0.5 * (left + right)
        omp_bits = float(np.interp(midpoint, omp_x, omp_y))
        swd_bits = float(np.interp(midpoint, swd_x, swd_y))
        regions.append(
            {
                "winner": "svd_omp" if omp_bits < swd_bits else "swd",
                "epsilon_min": left,
                "epsilon_max": right,
            }
        )
    return {
        "measured_overlap": [lower, upper],
        "crossovers": deduplicated,
        "regions": regions,
        "interpolation": "piecewise linear in log2(total_bits)",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def plot_curves(
    path: Path, rows: list[dict[str, Any]], crossovers: list[float]
) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "svd_omp_shared_dictionary": ("SVD-OMP, shared dictionary", "#0072B2", "-"),
        "svd_omp_dictionary_counted": ("SVD-OMP, dictionary counted", "#D55E00", "--"),
        "swd_dsf": ("SWD, measured DSF", "#009E73", "-"),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.7))
    for method, (label, color, line_style) in styles.items():
        frontier = pareto_frontier([row for row in rows if row["method"] == method])
        axis.plot(
            [row["relative_error"] for row in frontier],
            [row["bits_per_token"] for row in frontier],
            marker="o" if method == "swd_dsf" else None,
            markersize=4,
            linewidth=2,
            linestyle=line_style,
            color=color,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Relative output error on the shared calibration set")
    axis.set_ylabel("Amortized bits per token")
    axis.grid(True, which="both", alpha=0.25)
    measured_swd_errors = [
        row["relative_error"] for row in rows if row["method"] == "swd_dsf"
    ]
    if measured_swd_errors:
        x_right = min(1.0, max(measured_swd_errors) * 1.08)
        axis.set_xlim(-0.01, x_right)
        visible_bits = [
            row["bits_per_token"]
            for row in rows
            if 0 < row["bits_per_token"] and row["relative_error"] <= x_right
        ]
        axis.set_ylim(min(visible_bits) * 0.75, max(visible_bits) * 1.35)
    for crossover in crossovers:
        axis.axvline(crossover, color="#666666", linestyle=":", linewidth=1.2)
        axis.text(
            crossover,
            0.97,
            f"epsilon*={crossover:.3f}",
            rotation=90,
            va="top",
            ha="right",
            color="#555555",
            transform=axis.get_xaxis_transform(),
        )
    axis.legend(frameon=False, loc="lower left")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_sparsities(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result or any(not 0 <= item < 1 for item in result):
        raise argparse.ArgumentTypeError("Sparsities must be comma-separated values in [0, 1)")
    return result


def main(args: argparse.Namespace) -> None:
    weight, activations = load_goodfire(
        args.weights, args.activations, args.module, args.max_tokens
    )
    print(f"W {tuple(weight.shape)}, H {tuple(activations.shape)}, module {args.module}")
    print("Computing the exact per-token SVD-OMP curve...")
    omp_rows, omp_diagnostics = svd_omp_curve(
        weight, activations, args.bits, device=args.device
    )
    print(f"Running measured SWD DSF at {len(args.sparsities)} sparsity levels...")
    measured_swd_rows = swd_curve(
        weight,
        activations,
        args.swd_source,
        args.sparsities,
        args.bits,
        args.outer_iterations,
        args.device,
    )
    all_rows = omp_rows + measured_swd_rows
    shared_rows = [
        row for row in omp_rows if row["method"] == "svd_omp_shared_dictionary"
    ]
    comparison = interpolated_comparison(shared_rows, measured_swd_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "mdl_compare.csv"
    json_path = args.output_dir / "mdl_compare.json"
    figure_path = args.output_dir / "mdl_compare.png"
    write_csv(csv_path, all_rows)
    payload = {
        "status": "measured",
        "proxy_swd_used": False,
        "module": args.module,
        "weight_shape": list(weight.shape),
        "activation_shape": list(activations.shape),
        "bits_per_value": args.bits,
        "metric": "||H W^T - H W_hat^T||_F / ||H W^T||_F",
        "weights_sha256": sha256_file(args.weights),
        "activations_sha256": sha256_file(args.activations),
        "benchmark_source_sha256": sha256_file(Path(__file__).resolve()),
        "svd_omp_revision": args.svd_omp_revision
        or git_revision(Path(__file__).resolve().parent),
        "swd_revision": args.swd_revision or git_revision(args.swd_source),
        "swd_sparsities": list(args.sparsities),
        "swd_outer_iterations": args.outer_iterations,
        "omp_diagnostics": omp_diagnostics,
        "shared_dictionary_interpolated_comparison": comparison,
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    plot_curves(figure_path, all_rows, comparison["crossovers"])

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {figure_path}")
    print("Interpolated winner regions within measured SWD overlap:")
    for interval in comparison["regions"]:
        print(
            f"  {interval['winner']}: epsilon in "
            f"[{interval['epsilon_min']:.6f}, {interval['epsilon_max']:.6f}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, default=Path("weights/goodfire_67m_weights.pt")
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path("weights/goodfire_67m_activations.pt"),
    )
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--swd-source", type=Path, required=True)
    parser.add_argument(
        "--svd-omp-revision",
        default=None,
        help="Base SVD-OMP revision when the run directory has no .git metadata",
    )
    parser.add_argument(
        "--swd-revision",
        default=None,
        help="Exact SWD source revision when the checkout has no .git metadata",
    )
    parser.add_argument(
        "--sparsities",
        type=parse_sparsities,
        default=DEFAULT_SPARSITIES,
        help="Comma-separated DSF sparsity fractions",
    )
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--outer-iterations", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/mdl"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
