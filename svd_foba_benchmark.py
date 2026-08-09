"""Discovery or frozen evaluation sweep for overcomplete SVD-FoBa."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mdl_svdomp_vs_swd import sha256_file
from mdl_svdomp_vs_swd_natural_24 import load_inputs
from model_config import TARGET_MODULES
from selected_unit_svdomp_vs_swd import parse_int_tuple, source_sha256
from svd_foba import svd_foba_curve


def main(args: argparse.Namespace) -> None:
    weights, calibration, evaluation, metadata = load_inputs(
        args.weights, args.activations
    )
    modules = TARGET_MODULES[: args.max_modules] if args.max_modules else TARGET_MODULES
    matrices = []
    for index, module in enumerate(modules, start=1):
        print(f"[{index:02d}/{len(modules)}] {module}", flush=True)
        rows = svd_foba_curve(
            weights[module].to(args.device),
            calibration[module].to(args.device),
            evaluation[module].to(args.device),
            args.ks,
            alpha=args.alpha,
            candidate_atoms=args.candidate_atoms,
            seed=args.seed,
            swap_rounds=args.swap_rounds,
            proposal_width=args.proposal_width,
        )
        matrices.append(
            {
                "module": module,
                "family": "attention" if ".attn." in module else "mlp",
                "comparisons": rows,
            }
        )
        print(
            "  median error reduction "
            f"{100 * torch.tensor([row['mean_relative_loss_reduction'] for row in rows]).median().item():.2f}%",
            flush=True,
        )
    result = {
        "status": args.evaluation_status,
        "method": "calibration_aware_svd_initialized_overcomplete_foba",
        "alpha": args.alpha,
        "candidate_atoms": args.candidate_atoms,
        "seed": args.seed,
        "swap_rounds": args.swap_rounds,
        "proposal_width": args.proposal_width,
        "ks": list(args.ks),
        "weights_sha256": sha256_file(args.weights),
        "activations_sha256": sha256_file(args.activations),
        "benchmark_source_sha256": source_sha256(Path(__file__).resolve()),
        "method_source_sha256": source_sha256(Path(__file__).with_name("svd_foba.py")),
        "activation_metadata": metadata,
        "matrix_count": len(matrices),
        "point_count": len(matrices) * len(args.ks),
        "strict_point_wins_over_svd": sum(
            row["svd_foba_relative_error"] < row["svd_relative_error"]
            for matrix in matrices
            for row in matrix["comparisons"]
        ),
        "matrices": matrices,
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
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--candidate-atoms", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--swap-rounds", type=int, default=2)
    parser.add_argument("--proposal-width", type=int, default=8)
    parser.add_argument("--ks", type=parse_int_tuple, default=(1, 2, 4, 8, 12, 16, 24, 32, 48, 64))
    parser.add_argument("--max-modules", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluation-status", default="discovery_only_validation")
    parser.add_argument(
        "--output", type=Path, default=Path("results/svd_foba/discovery.json")
    )
    args = parser.parse_args()
    if args.candidate_atoms < 0:
        parser.error("--candidate-atoms must be non-negative")
    if args.swap_rounds < 0:
        parser.error("--swap-rounds must be non-negative")
    return args


if __name__ == "__main__":
    main(parse_args())
