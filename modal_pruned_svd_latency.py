"""GPU latency gate for frozen calibration-pruned selected-unit SVD."""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("pruned-selected-unit-svd-latency")

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
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=False)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/volume": volume},
    timeout=1800,
)
def benchmark() -> dict:
    import platform
    import statistics
    import time

    import torch

    from mdl_svdomp_vs_swd_natural_24 import load_inputs
    from pruned_svd_foba import calibration_selected_pool
    from svd_foba import (
        build_overcomplete_dictionary,
        calibration_aware_svd_factors,
        reconstruct_with_svd_foba,
    )

    device = torch.device("cuda")
    weights, calibration, heldout, metadata = load_inputs(
        Path("/volume/weights/goodfire_67m_weights.pt"),
        Path("/volume/weights/goodfire_67m_natural_24_foba_sealed_activations.pt"),
    )
    cases = (
        ("h.0.attn.q_proj", 8),
        ("h.0.mlp.c_fc", 12),
    )
    batch_sizes = (1, 16, 128, 512)
    results = []

    def timing(function, repeats: int) -> dict[str, float]:
        for _ in range(5):
            function()
        torch.cuda.synchronize()
        wall_samples = []
        event_samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            started_wall = time.perf_counter()
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            output = function()
            finished.record()
            torch.cuda.synchronize()
            wall_samples.append((time.perf_counter() - started_wall) * 1000.0)
            event_samples.append(started.elapsed_time(finished))
            if not torch.isfinite(output).all():
                raise RuntimeError("non-finite benchmark output")
        return {
            "wall_milliseconds_median": statistics.median(wall_samples),
            "wall_milliseconds_minimum": min(wall_samples),
            "cuda_milliseconds_median": statistics.median(event_samples),
            "cuda_milliseconds_minimum": min(event_samples),
        }

    for module, k in cases:
        weight = weights[module].to(device)
        train_h = calibration[module].to(device)
        test_h = heldout[module].to(device)
        output_atoms, singular_values, read_vectors = calibration_aware_svd_factors(
            weight, train_h, 0.1
        )
        calibration_coefficients = (
            train_h.matmul(read_vectors) * singular_values.unsqueeze(0)
        )
        pool = calibration_selected_pool(
            calibration_coefficients, pool_size=96, selection_width=64
        )
        pruned_dictionary = output_atoms[:, pool].contiguous()
        pruned_analysis = weight.T.matmul(pruned_dictionary).contiguous()
        full_dictionary = build_overcomplete_dictionary(
            weight, train_h, output_atoms, 128, 0
        ).contiguous()
        full_analysis = weight.T.matmul(full_dictionary).contiguous()
        svd_analysis = (read_vectors * singular_values.unsqueeze(0)).contiguous()

        for batch_size in batch_sizes:
            inputs = test_h[:batch_size].contiguous()

            def dense():
                return inputs.matmul(weight.T)

            def full_svd_dynamic():
                coefficients = inputs.matmul(svd_analysis)
                indices = coefficients.abs().topk(k, dim=1).indices
                selected = torch.zeros_like(coefficients)
                selected.scatter_(1, indices, coefficients.gather(1, indices))
                return selected.matmul(output_atoms.T)

            def pruned_dense_output():
                coefficients = inputs.matmul(pruned_analysis)
                indices = coefficients.abs().topk(k, dim=1).indices
                selected = torch.zeros_like(coefficients)
                selected.scatter_(1, indices, coefficients.gather(1, indices))
                return selected.matmul(pruned_dictionary.T)

            def pruned_gather_output():
                coefficients = inputs.matmul(pruned_analysis)
                indices = coefficients.abs().topk(k, dim=1).indices
                chosen = coefficients.gather(1, indices)
                atoms = pruned_dictionary.T[indices]
                return torch.einsum("nk,nkd->nd", chosen, atoms)

            def full_foba():
                output, _ = reconstruct_with_svd_foba(
                    inputs,
                    full_dictionary,
                    full_analysis,
                    output_atoms.shape[1],
                    k,
                    swap_rounds=2,
                    proposal_width=8,
                )
                return output

            functions = {
                "dense": dense,
                "full_dynamic_svd": full_svd_dynamic,
                "pruned96_dense_output": pruned_dense_output,
                "pruned96_gather_output": pruned_gather_output,
                "full_svd_foba": full_foba,
            }
            repeats = 30 if batch_size <= 16 else 15
            for method, function in functions.items():
                measured = timing(function, repeats)
                results.append(
                    {
                        "module": module,
                        "selected_units": k,
                        "batch_size": batch_size,
                        "method": method,
                        **measured,
                    }
                )
                print(
                    module,
                    batch_size,
                    method,
                    measured["wall_milliseconds_median"],
                    flush=True,
                )

    return {
        "status": "same_gpu_kernel_level_latency_screen",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "activation_metadata": metadata,
        "dtype": "float32",
        "warmups": 5,
        "synchronization": "before and after every measured call",
        "results": results,
    }


@app.local_entrypoint()
def main() -> None:
    result = benchmark.remote()
    by_key = {
        (row["module"], row["batch_size"], row["method"]): row
        for row in result["results"]
    }
    comparisons = []
    for module, batch_size, method in sorted(by_key):
        if method != "pruned96_dense_output":
            continue
        candidate = by_key[(module, batch_size, method)]
        full_foba = by_key[(module, batch_size, "full_svd_foba")]
        full_svd = by_key[(module, batch_size, "full_dynamic_svd")]
        comparisons.append(
            {
                "module": module,
                "batch_size": batch_size,
                "full_foba_wall_speedup": (
                    full_foba["wall_milliseconds_median"]
                    / candidate["wall_milliseconds_median"]
                ),
                "full_svd_wall_speedup": (
                    full_svd["wall_milliseconds_median"]
                    / candidate["wall_milliseconds_median"]
                ),
                "candidate_wall_milliseconds": candidate[
                    "wall_milliseconds_median"
                ],
            }
        )
    result["comparisons"] = comparisons
    output = Path("results/pruned_svd_foba/a10g_latency.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(comparisons, indent=2))
    print(f"wrote {output}")
