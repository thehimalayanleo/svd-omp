"""Validation sweep and frozen held-out evaluation for pruned SVD-FoBa."""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("pruned-svd-foba-cost-gate")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.4", "numpy")
    .env({"PYTHONPATH": "/root/svd-omp"})
    .add_local_file("svd_foba.py", "/root/svd-omp/svd_foba.py")
    .add_local_file("pruned_svd_foba.py", "/root/svd-omp/pruned_svd_foba.py")
    .add_local_file("mdl_svdomp_vs_swd.py", "/root/svd-omp/mdl_svdomp_vs_swd.py")
    .add_local_file(
        "mdl_svdomp_vs_swd_natural_24.py",
        "/root/svd-omp/mdl_svdomp_vs_swd_natural_24.py",
    )
    .add_local_file("model_config.py", "/root/svd-omp/model_config.py")
    .add_local_file(
        "tests/test_pruned_svd_foba.py",
        "/root/svd-omp/tests/test_pruned_svd_foba.py",
    )
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=False)

KS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
CONFIGURATIONS = (
    # Selector-only pruning establishes how small the eligible SVD pool can be.
    (64, 0, 0),
    (96, 0, 0),
    (128, 0, 0),
    (192, 0, 0),
    (256, 0, 0),
    # One protected swap with a small residual dictionary is the deployment
    # candidate.  These settings are screened only on validation.
    (64, 16, 1),
    (96, 16, 1),
    (128, 16, 1),
    (192, 16, 1),
    (64, 32, 1),
    (96, 32, 1),
    (128, 32, 1),
    (192, 32, 1),
)


@app.function(image=image, timeout=300)
def unit_tests() -> dict:
    import runpy

    namespace = runpy.run_path("/root/svd-omp/tests/test_pruned_svd_foba.py")
    names = (
        "test_calibration_pool_preserves_locally_important_direction",
        "test_pruned_foba_is_protected_and_reports_cost_reduction",
        "test_zero_swap_matches_pooled_svd",
    )
    for name in names:
        namespace[name]()
    return {"passed": len(names), "tests": list(names)}


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/volume": volume},
    timeout=3600,
)
def evaluate(activations_name: str, configurations: tuple[tuple[int, int, int], ...]) -> dict:
    import hashlib

    import torch

    from mdl_svdomp_vs_swd_natural_24 import load_inputs
    from model_config import TARGET_MODULES
    from pruned_svd_foba import pruned_svd_foba_curve

    weights_path = Path("/volume/weights/goodfire_67m_weights.pt")
    activations_path = Path("/volume/weights") / activations_name
    weights, calibration, heldout, activation_metadata = load_inputs(
        weights_path, activations_path
    )
    device = torch.device("cuda")
    matrices = []
    for index, module in enumerate(TARGET_MODULES, start=1):
        print(f"[{index:02d}/{len(TARGET_MODULES)}] {module}", flush=True)
        matrix = {
            "module": module,
            "family": "attention" if ".attn." in module else "mlp",
            "configurations": [],
        }
        weight = weights[module].to(device)
        train_h = calibration[module].to(device)
        test_h = heldout[module].to(device)
        for pool_size, candidate_atoms, swap_rounds in configurations:
            rows, cost = pruned_svd_foba_curve(
                weight,
                train_h,
                test_h,
                KS,
                alpha=0.1,
                pool_size=pool_size,
                pool_selection_width=64,
                candidate_atoms=candidate_atoms,
                seed=0,
                swap_rounds=swap_rounds,
                proposal_width=8,
            )
            matrix["configurations"].append(
                {
                    "pool_size": pool_size,
                    "candidate_atoms": candidate_atoms,
                    "swap_rounds": swap_rounds,
                    "comparisons": rows,
                    "cost": cost,
                }
            )
        matrices.append(matrix)
    return {
        "status": "validation_screen" if "natural_24_activations" in activations_name else "frozen_heldout",
        "activations_name": activations_name,
        "activations_sha256": hashlib.sha256(activations_path.read_bytes()).hexdigest(),
        "activation_metadata": activation_metadata,
        "configurations": [
            {
                "pool_size": pool,
                "candidate_atoms": atoms,
                "swap_rounds": rounds,
            }
            for pool, atoms, rounds in configurations
        ],
        "ks": list(KS),
        "matrices": matrices,
    }


def configuration_summary(candidate: dict, swd_path: Path) -> list[dict]:
    import math

    swd = json.loads(swd_path.read_text())
    swd_by_key = {
        (matrix["module"], row["selected_units"]): row["swd_relative_error"]
        for matrix in swd["matrices"]
        for row in matrix["comparisons"]
    }
    aggregates: dict[tuple[int, int, int], dict] = {}
    for matrix in candidate["matrices"]:
        for config in matrix["configurations"]:
            key = (
                config["pool_size"],
                config["candidate_atoms"],
                config["swap_rounds"],
            )
            aggregate = aggregates.setdefault(
                key,
                {
                    "pool_size": key[0],
                    "candidate_atoms": key[1],
                    "swap_rounds": key[2],
                    "ratios": [],
                    "full_svd_ratios": [],
                    "cost_fractions": [],
                },
            )
            aggregate["cost_fractions"].append(
                config["cost"]["selector_read_fraction_of_full_foba"]
            )
            for row in config["comparisons"]:
                swd_error = swd_by_key[(matrix["module"], row["selected_units"])]
                aggregate["ratios"].append(
                    swd_error / row["pruned_foba_relative_error"]
                )
                aggregate["full_svd_ratios"].append(
                    row["pruned_foba_relative_error"] / row["full_svd_relative_error"]
                )

    summaries = []
    for aggregate in aggregates.values():
        ratios = aggregate.pop("ratios")
        full_svd_ratios = aggregate.pop("full_svd_ratios")
        cost_fractions = aggregate.pop("cost_fractions")
        summaries.append(
            {
                **aggregate,
                "wins_over_swd": sum(value > 1.0 for value in ratios),
                "point_count": len(ratios),
                "geometric_mean_swd_error_over_candidate": math.exp(
                    sum(math.log(value) for value in ratios) / len(ratios)
                ),
                "minimum_swd_error_over_candidate": min(ratios),
                "geometric_mean_candidate_error_over_full_svd": math.exp(
                    sum(math.log(value) for value in full_svd_ratios)
                    / len(full_svd_ratios)
                ),
                "mean_selector_read_fraction_of_full_foba": sum(cost_fractions)
                / len(cost_fractions),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -row["wins_over_swd"],
            row["mean_selector_read_fraction_of_full_foba"],
            -row["geometric_mean_swd_error_over_candidate"],
        ),
    )


@app.local_entrypoint()
def main(stage: str = "validation") -> None:
    output_dir = Path("results/pruned_svd_foba")
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage == "validation":
        activations = "goodfire_67m_natural_24_activations.pt"
        configs = CONFIGURATIONS
        swd_path = Path("results/selected_units/discovery_validation.json")
        output = output_dir / "validation_screen.json"
    elif stage == "heldout":
        frozen_path = output_dir / "frozen_configuration.json"
        frozen = json.loads(frozen_path.read_text())
        activations = "goodfire_67m_natural_24_foba_sealed_activations.pt"
        configs = (
            (
                int(frozen["pool_size"]),
                int(frozen["candidate_atoms"]),
                int(frozen["swap_rounds"]),
            ),
        )
        swd_path = Path("results/selected_units/selected_unit_fresh_test.json")
        output = output_dir / "sealed_fresh_test.json"
    else:
        raise ValueError("stage must be validation or heldout")

    result = evaluate.remote(activations, configs)
    summaries = configuration_summary(result, swd_path)
    result["summary"] = summaries
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(summaries, indent=2))
    print(f"wrote {output}")
