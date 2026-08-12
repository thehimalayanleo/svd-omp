"""Direct all-24 CP-SVD replacement: synchronized quality and latency gate."""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("cp-svd-direct-replacement-b200-batch-sweep")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.4",
        "pytest",
        "wandb",
        "transformers",
        "tokenizers",
        "jaxtyping",
        "einops",
        "pydantic",
        "python-dotenv",
        "fire",
        "tqdm",
        "scipy",
        "numpy",
        "pyyaml",
    )
    .run_commands(
        "git clone --depth 1 --branch torch-oracle "
        "https://github.com/goodfire-ai/param-decomp /root/param-decomp"
    )
    .env({"PYTHONPATH": "/root/param-decomp:/root/svd-omp"})
    .add_local_file("svd_foba.py", "/root/svd-omp/svd_foba.py")
    .add_local_file("pruned_svd_foba.py", "/root/svd-omp/pruned_svd_foba.py")
    .add_local_file("cp_svd_runtime_b200.py", "/root/svd-omp/cp_svd_runtime_b200.py")
    .add_local_file(
        "tests/test_cp_svd_runtime.py",
        "/root/svd-omp/tests/test_cp_svd_runtime.py",
    )
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=False)

FROZEN_WIDTHS = {
    "h.0.attn.q_proj": 8,
    "h.0.attn.k_proj": 8,
    "h.0.attn.v_proj": 12,
    "h.0.attn.o_proj": 12,
    "h.0.mlp.c_fc": 12,
    "h.0.mlp.down_proj": 12,
    "h.1.attn.q_proj": 8,
    "h.1.attn.k_proj": 8,
    "h.1.attn.v_proj": 12,
    "h.1.attn.o_proj": 12,
    "h.1.mlp.c_fc": 12,
    "h.1.mlp.down_proj": 12,
    "h.2.attn.q_proj": 8,
    "h.2.attn.k_proj": 8,
    "h.2.attn.v_proj": 12,
    "h.2.attn.o_proj": 12,
    "h.2.mlp.c_fc": 12,
    "h.2.mlp.down_proj": 12,
    "h.3.attn.q_proj": 8,
    "h.3.attn.k_proj": 8,
    "h.3.attn.v_proj": 12,
    "h.3.attn.o_proj": 12,
    "h.3.mlp.c_fc": 12,
    "h.3.mlp.down_proj": 12,
}


@app.function(image=image, timeout=600)
def unit_tests() -> str:
    import subprocess

    completed = subprocess.run(
        ["python", "-m", "pytest", "-q", "/root/svd-omp/tests/test_cp_svd_runtime.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


@app.function(
    image=image,
    gpu="B200",
    volumes={"/volume": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=1800,
)
def evaluate() -> dict:
    import platform
    import statistics
    import time

    import torch
    import torch.nn.functional as functional
    from cp_svd_runtime_b200 import CPSVDLinear, CPSVDLinearB200, replace_submodule
    from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
        LlamaSimpleMLP,
    )
    from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo
    from pruned_svd_foba import calibration_selected_pool
    from svd_foba import calibration_aware_svd_factors

    torch.manual_seed(0)
    device = torch.device("cuda")
    artifact = Path(
        "/volume/weights/goodfire_67m_natural_24_foba_sealed_activations.pt"
    )
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    run_path = "goodfire/spd/runs/t-9d2b8f02"
    run_info = PretrainRunInfo.from_path(run_path)
    model = LlamaSimpleMLP.from_run_info(run_info).eval().to(device)
    source_input_ids = payload["heldout_input_ids"].to(device)
    input_ids = source_input_ids

    def model_logits() -> torch.Tensor:
        with torch.no_grad():
            logits, _ = model(input_ids, return_logits=True)
        assert logits is not None
        return logits.float()

    def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
        reference_next = reference[:, :-1]
        candidate_next = candidate[:, :-1]
        targets = input_ids[:, 1:]
        cross_entropy = functional.cross_entropy(
            candidate_next.reshape(-1, candidate_next.shape[-1]), targets.reshape(-1)
        )
        reference_log_probs = reference_next.log_softmax(dim=-1)
        candidate_log_probs = candidate_next.log_softmax(dim=-1)
        kl = (
            reference_log_probs.exp()
            * (reference_log_probs - candidate_log_probs)
        ).sum(dim=-1).mean()
        return {
            "cross_entropy": float(cross_entropy.item()),
            "kl_to_dense": float(kl.item()),
            "logit_mse": float((reference_next - candidate_next).square().mean().item()),
        }

    originals = {}
    candidates = {}
    b200_candidates = {}
    factor_elements = 0
    dense_weight_elements = 0
    for index, (path, selected_units) in enumerate(FROZEN_WIDTHS.items(), start=1):
        print(f"[{index:02d}/24] preparing direct replacement {path}", flush=True)
        module = model.get_submodule(path)
        weight = module.weight.detach().float()
        calibration = payload["calibration"][path].to(device)
        output_atoms, singular_values, read_vectors = calibration_aware_svd_factors(
            weight, calibration, 0.1
        )
        calibration_coefficients = (
            calibration.matmul(read_vectors) * singular_values.unsqueeze(0)
        )
        pool = calibration_selected_pool(
            calibration_coefficients, pool_size=96, selection_width=64
        )
        dictionary = output_atoms[:, pool].contiguous()
        analysis = weight.T.matmul(dictionary).contiguous()
        candidate = CPSVDLinear(
            analysis,
            dictionary,
            selected_units,
            module.bias,
        ).to(device)
        b200_candidate = CPSVDLinearB200(
            analysis,
            dictionary,
            selected_units,
            module.bias,
        ).to(device)
        originals[path] = module
        candidates[path] = candidate
        b200_candidates[path] = b200_candidate
        factor_elements += analysis.numel() + dictionary.numel()
        dense_weight_elements += weight.numel()

    def install(modules: dict[str, torch.nn.Module]) -> None:
        for path, replacement in modules.items():
            replace_submodule(model, path, replacement)

    def timed_block(
        modules: dict[str, torch.nn.Module], repeats: int = 4, warmups: int = 2
    ) -> list[float]:
        install(modules)
        for _ in range(warmups):
            model_logits()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model_logits()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
            if not torch.isfinite(logits).all():
                raise RuntimeError("non-finite logits during latency gate")
        return samples

    install(originals)
    dense_logits = model_logits()
    dense_metrics = metrics(dense_logits, dense_logits)
    install(candidates)
    candidate_logits = model_logits()
    candidate_metrics = metrics(dense_logits, candidate_logits)
    install(b200_candidates)
    b200_candidate_logits = model_logits()
    b200_candidate_metrics = metrics(dense_logits, b200_candidate_logits)
    b200_formula_max_abs_error = float(
        (candidate_logits - b200_candidate_logits).abs().max().item()
    )
    if not torch.allclose(
        candidate_logits, b200_candidate_logits, rtol=2e-5, atol=2e-5
    ):
        raise RuntimeError(
            f"B200 fused formula mismatch: max abs {b200_formula_max_abs_error}"
        )

    def summarize(samples: list[float]) -> dict:
        return {
            "wall_milliseconds_median": statistics.median(samples),
            "wall_milliseconds_minimum": min(samples),
            "wall_milliseconds_mean": statistics.mean(samples),
            "sample_count": len(samples),
            "wall_milliseconds_samples": samples,
        }

    sweep = []
    for batch_multiplier in (1, 2, 4, 8):
        input_ids = source_input_ids.repeat(batch_multiplier, 1)
        blocks = []
        methods = {"dense": originals, "b200": b200_candidates}
        for cycle in range(6):
            order = ("dense", "b200") if cycle % 2 == 0 else ("b200", "dense")
            for method in order:
                samples = timed_block(methods[method])
                blocks.append({
                    "cycle": cycle,
                    "method": method,
                    "samples": samples,
                    "median": statistics.median(samples),
                })
        dense_samples = [
            value for block in blocks if block["method"] == "dense"
            for value in block["samples"]
        ]
        b200_samples = [
            value for block in blocks if block["method"] == "b200"
            for value in block["samples"]
        ]
        ratios = []
        for cycle in range(6):
            by_method = {
                block["method"]: block["median"]
                for block in blocks if block["cycle"] == cycle
            }
            ratios.append(by_method["dense"] / by_method["b200"])
        dense_latency = summarize(dense_samples)
        b200_latency = summarize(b200_samples)
        sweep.append({
            "batch": int(input_ids.shape[0]),
            "sequence_length": int(input_ids.shape[1]),
            "token_count": int(input_ids.numel()),
            "dense": dense_latency,
            "b200_candidate": b200_latency,
            "paired_speedup_median": statistics.median(ratios),
            "paired_speedup_min": min(ratios),
            "paired_speedup_max": max(ratios),
            "latency_reduction_fraction": 1.0 - 1.0 / statistics.median(ratios),
            "blocks": blocks,
        })
    return {
        "status": "b200_batch_scaling_sweep",
        "replacement_scope": "all_24_target_matrices_directly_replaced",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "wandb_run": run_path,
        "activation_metadata": payload["metadata"],
        "dense": dense_metrics,
        "candidate": candidate_metrics,
        "b200_candidate": b200_candidate_metrics,
        "b200_formula_max_abs_error": b200_formula_max_abs_error,
        "batch_sweep": sweep,
        "timing_protocol": {
            "warmups_per_block": 2,
            "samples_per_block": 4,
            "cycle_count": 6,
            "synchronization": "before and after every measured forward",
            "measurement_order": "short mirrored dense/b200 cycles",
        },
        "storage": {
            "dense_weight_elements": dense_weight_elements,
            "candidate_factor_elements": factor_elements,
            "candidate_over_dense": factor_elements / dense_weight_elements,
        },
        "frozen_configuration": {
            "pool_size": 96,
            "pool_selection_width": 64,
            "candidate_atoms": 0,
            "swap_rounds": 0,
            "alpha": 0.1,
            "selected_widths": FROZEN_WIDTHS,
        },
    }


@app.local_entrypoint()
def main(
    run_tests: bool = True,
    tests_only: bool = False,
    run_id: str = "discovery",
) -> None:
    if run_tests:
        print(unit_tests.remote())
    if tests_only:
        return
    result = evaluate.remote()
    baseline = json.loads(
        Path("results/pruned_svd_foba/simultaneous_all_24.json").read_text()
    )
    result["hook_prototype"] = {
        "candidate": baseline["candidate"],
        "latency": baseline["latency"],
    }
    result["quality_delta_from_hook_prototype"] = {
        key: result["candidate"][key] - baseline["candidate"][key]
        for key in ("cross_entropy", "kl_to_dense", "logit_mse")
    }
    safe_run_id = "".join(
        character for character in run_id if character.isalnum() or character in "-_"
    )
    if not safe_run_id:
        raise ValueError("run_id must contain at least one safe filename character")
    output = Path("results/cp_svd_direct") / f"direct_all_24_{safe_run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")
