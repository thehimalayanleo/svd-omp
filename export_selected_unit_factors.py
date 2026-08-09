"""Export frozen native-k whitened SVD-OMP and SWD factors for model evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mdl_svdomp_vs_swd import import_swd_factorizer, sha256_file
from mdl_svdomp_vs_swd_natural_24 import load_inputs
from model_config import TARGET_MODULES, get_k


@torch.no_grad()
def full_whitened_factors(
    weight: torch.Tensor,
    calibration: torch.Tensor,
    alpha: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    target_device = torch.device(device)
    weight = weight.to(target_device)
    h = calibration.to(target_device)
    gram = h.T.matmul(h) / h.shape[0]
    scale = gram.diagonal().mean().clamp_min(1e-30)
    gram = gram + alpha * scale * torch.eye(
        gram.shape[0], device=target_device, dtype=gram.dtype
    )
    chol = torch.linalg.cholesky(gram)
    left, singular_values, vh = torch.linalg.svd(
        weight.matmul(chol), full_matrices=False
    )
    read_vectors = torch.linalg.solve_triangular(chol.T, vh.T, upper=True)
    reconstruction = (left * singular_values.unsqueeze(0)).matmul(read_vectors.T)
    error = float((weight - reconstruction).norm() / weight.norm().clamp_min(1e-30))
    return left.cpu(), singular_values.cpu(), read_vectors.cpu(), error


def main(args: argparse.Namespace) -> None:
    weights, calibration, _, activation_metadata = load_inputs(
        args.weights, args.activations
    )
    discovery = json.loads(args.discovery.read_text())
    discovery_by_module = {row["module"]: row for row in discovery["matrices"]}
    test_payload = torch.load(
        args.test_activations, map_location="cpu", weights_only=False
    )
    factorize_matrix = import_swd_factorizer(args.swd_source)
    target_device = torch.device(args.device)
    modules = {}
    for index, module in enumerate(TARGET_MODULES, start=1):
        requested_k = get_k(module)
        available_ks = sorted(
            row["selected_units"]
            for row in discovery_by_module[module]["comparisons"]
        )
        k = min((value for value in available_ks if value >= requested_k), default=available_ks[-1])
        comparison = next(
            row
            for row in discovery_by_module[module]["comparisons"]
            if row["selected_units"] == k
        )
        sparsity = float(comparison["swd_best_sparsity"])
        print(
            f"[{index:02d}/{len(TARGET_MODULES)}] {module}: "
            f"requested k={requested_k}, evaluated k={k}, SWD s={sparsity}",
            flush=True,
        )
        left, singular_values, read_vectors, exact_error = full_whitened_factors(
            weights[module], calibration[module], args.alpha, args.device
        )
        h = calibration[module].to(target_device)
        result = factorize_matrix(
            weights[module].T.contiguous().to(target_device),
            h.T.matmul(h),
            sparsity,
            outer_iterations=args.outer_iterations,
            final_iterations=20,
            device=target_device,
            capture_stdout=True,
        )
        modules[module] = {
            "k": k,
            "requested_native_k": requested_k,
            "swd_sparsity": sparsity,
            "svd_left": left,
            "svd_singular_values": singular_values,
            "svd_read_vectors": read_vectors,
            "svd_weight_reconstruction_relative_error": exact_error,
            "swd_factor_a": result.factor_a,
            "swd_factor_b": result.factor_b,
            "swd_calibration_objective_error": result.objective_error,
        }
    payload = {
        "metadata": {
            "status": "frozen_from_validation_before_model_test",
            "alpha": args.alpha,
            "outer_iterations": args.outer_iterations,
            "weights_sha256": sha256_file(args.weights),
            "calibration_activations_sha256": sha256_file(args.activations),
            "test_activations_sha256": sha256_file(args.test_activations),
            "discovery_sha256": sha256_file(args.discovery),
            "activation_metadata": activation_metadata,
            "test_metadata": test_payload["metadata"],
        },
        "test_input_ids": test_payload["heldout_input_ids"],
        "modules": modules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                **payload["metadata"],
                "module_count": len(modules),
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
            },
            indent=2,
        )
    )


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
    parser.add_argument("--test-activations", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--swd-source", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--outer-iterations", type=int, default=40)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
