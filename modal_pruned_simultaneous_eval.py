"""All-24-matrix quality and prototype latency gate for pruned selected-unit SVD."""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("pruned-selected-unit-svd-simultaneous")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.4",
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


@app.function(
    image=image,
    gpu="T4",
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
    from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
        LlamaSimpleMLP,
    )
    from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo
    from pruned_svd_foba import calibration_selected_pool
    from svd_foba import calibration_aware_svd_factors

    device = torch.device("cuda")
    artifact = Path(
        "/volume/weights/goodfire_67m_natural_24_foba_sealed_activations.pt"
    )
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    run_path = "goodfire/spd/runs/t-9d2b8f02"
    run_info = PretrainRunInfo.from_path(run_path)
    model = LlamaSimpleMLP.from_run_info(run_info).eval().to(device)
    input_ids = payload["heldout_input_ids"].to(device)

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

    factors = {}
    for index, (path, k) in enumerate(FROZEN_WIDTHS.items(), start=1):
        print(f"[{index:02d}/24] preparing {path}", flush=True)
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
        factors[path] = {
            "k": k,
            "dictionary": dictionary,
            "analysis": weight.T.matmul(dictionary).contiguous(),
        }

    handles = []

    def make_hook(path: str):
        data = factors[path]

        def hook(module, inputs, _output):
            values = inputs[0]
            shape = values.shape
            flat = values.reshape(-1, shape[-1]).float()
            coefficients = flat.matmul(data["analysis"])
            indices = coefficients.abs().topk(data["k"], dim=1).indices
            selected = torch.zeros_like(coefficients)
            selected.scatter_(1, indices, coefficients.gather(1, indices))
            approximation = selected.matmul(data["dictionary"].T)
            if module.bias is not None:
                approximation += module.bias.float()
            return approximation.reshape(
                *shape[:-1], approximation.shape[-1]
            ).to(values.dtype)

        return hook

    dense_logits = model_logits()
    dense_metrics = metrics(dense_logits, dense_logits)
    for path in FROZEN_WIDTHS:
        handles.append(model.get_submodule(path).register_forward_hook(make_hook(path)))
    candidate_logits = model_logits()
    candidate_metrics = metrics(dense_logits, candidate_logits)

    def latency(with_candidate: bool) -> dict[str, float]:
        if with_candidate and not handles:
            raise RuntimeError("candidate hooks are not installed")
        if not with_candidate and handles:
            for handle in handles:
                handle.remove()
            handles.clear()
        for _ in range(5):
            model_logits()
        torch.cuda.synchronize()
        samples = []
        for _ in range(20):
            torch.cuda.synchronize()
            started = time.perf_counter()
            model_logits()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
        return {
            "wall_milliseconds_median": statistics.median(samples),
            "wall_milliseconds_minimum": min(samples),
        }

    candidate_latency = latency(with_candidate=True)
    dense_latency = latency(with_candidate=False)
    return {
        "status": "frozen_all_24_simultaneous_quality_and_prototype_latency",
        "replacement_scope": "all_24_target_matrices_in_one_forward_pass",
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "wandb_run": run_path,
        "activation_metadata": payload["metadata"],
        "dense": dense_metrics,
        "candidate": candidate_metrics,
        "latency": {
            "dense": dense_latency,
            "candidate": candidate_latency,
            "dense_over_candidate_speedup": dense_latency[
                "wall_milliseconds_median"
            ]
            / candidate_latency["wall_milliseconds_median"],
            "input_shape": list(input_ids.shape),
            "warmups": 5,
            "repeats": 20,
            "synchronization": "before and after every measured forward",
        },
        "frozen_configuration": {
            "pool_size": 96,
            "pool_selection_width": 64,
            "candidate_atoms": 0,
            "swap_rounds": 0,
            "alpha": 0.1,
        },
    }


@app.local_entrypoint()
def main() -> None:
    result = evaluate.remote()
    baseline = json.loads(
        Path("results/svd_foba/simultaneous_all_24_model_eval.json").read_text()
    )
    result["prior_sparse_methods"] = baseline["methods"]
    candidate = result["candidate"]
    result["comparisons"] = {
        "candidate_kl_over_foba": candidate["kl_to_dense"]
        / baseline["methods"]["svd_foba"]["kl_to_dense"],
        "swd_kl_over_candidate": baseline["methods"]["swd"]["kl_to_dense"]
        / candidate["kl_to_dense"],
        "candidate_mse_over_foba": candidate["logit_mse"]
        / baseline["methods"]["svd_foba"]["logit_mse"],
        "swd_mse_over_candidate": baseline["methods"]["swd"]["logit_mse"]
        / candidate["logit_mse"],
    }
    output = Path("results/pruned_svd_foba/simultaneous_all_24.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {output}")
