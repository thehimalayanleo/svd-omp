"""Sealed full-model evaluation for frozen selected-unit SVD-OMP and SWD."""

from __future__ import annotations

import modal


app = modal.App("svd-omp-selected-unit-model-eval")

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
    )
    .run_commands(
        "git clone --depth 1 --branch torch-oracle "
        "https://github.com/goodfire-ai/param-decomp /root/param-decomp",
        "git clone https://github.com/veri-safe/SWD.git /root/SWD",
        "git -C /root/SWD checkout 4c44b7281bc7c78f80e431dac3aa75f397dd3043",
    )
    .env(
        {
            "PYTHONPATH": "/root/param-decomp:/root/SWD/src:/root/svd-omp",
        }
    )
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=False)


FROZEN_SETTINGS = {
    "h.0.attn.q_proj": (8, 0.82),
    "h.0.attn.k_proj": (8, 0.82),
    "h.0.attn.v_proj": (12, 0.82),
    "h.0.attn.o_proj": (12, 0.82),
    "h.0.mlp.c_fc": (12, 0.58),
    "h.0.mlp.down_proj": (12, 0.30),
    "h.1.attn.q_proj": (8, 0.82),
    "h.1.attn.k_proj": (8, 0.82),
    "h.1.attn.v_proj": (12, 0.30),
    "h.1.attn.o_proj": (12, 0.82),
    "h.1.mlp.c_fc": (12, 0.76),
    "h.1.mlp.down_proj": (12, 0.76),
    "h.2.attn.q_proj": (8, 0.82),
    "h.2.attn.k_proj": (8, 0.82),
    "h.2.attn.v_proj": (12, 0.30),
    "h.2.attn.o_proj": (12, 0.82),
    "h.2.mlp.c_fc": (12, 0.69),
    "h.2.mlp.down_proj": (12, 0.30),
    "h.3.attn.q_proj": (8, 0.82),
    "h.3.attn.k_proj": (8, 0.82),
    "h.3.attn.v_proj": (12, 0.30),
    "h.3.attn.o_proj": (12, 0.82),
    "h.3.mlp.c_fc": (12, 0.76),
    "h.3.mlp.down_proj": (12, 0.58),
}


@app.function(
    image=image,
    gpu="T4",
    volumes={"/volume": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=1800,
)
def evaluate() -> dict:
    import hashlib
    import json
    from pathlib import Path

    import torch
    import torch.nn.functional as functional
    from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
        LlamaSimpleMLP,
    )
    from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo
    from swd.factorization import factorize_matrix

    device = torch.device("cuda")
    payload_path = Path(
        "/volume/weights/goodfire_67m_natural_24_train_test_activations.pt"
    )
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    run_path = "goodfire/spd/runs/t-9d2b8f02"
    run_info = PretrainRunInfo.from_path(run_path)
    model = LlamaSimpleMLP.from_run_info(run_info).eval().to(device)
    input_ids = payload["heldout_input_ids"].to(device)

    def model_logits() -> torch.Tensor:
        with torch.no_grad():
            logits, _ = model(input_ids, return_logits=True)
        assert logits is not None
        return logits

    def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
        reference_next = reference[:, :-1].float()
        candidate_next = candidate[:, :-1].float()
        targets = input_ids[:, 1:]
        cross_entropy = functional.cross_entropy(
            candidate_next.reshape(-1, candidate_next.shape[-1]),
            targets.reshape(-1),
        )
        reference_log_probs = reference_next.log_softmax(dim=-1)
        candidate_log_probs = candidate_next.log_softmax(dim=-1)
        reference_probs = reference_log_probs.exp()
        kl = (
            reference_probs * (reference_log_probs - candidate_log_probs)
        ).sum(dim=-1).mean()
        logit_mse = (reference_next - candidate_next).square().mean()
        return {
            "cross_entropy": float(cross_entropy.item()),
            "kl_to_dense": float(kl.item()),
            "logit_mse": float(logit_mse.item()),
        }

    dense_logits = model_logits()
    dense_metrics = metrics(dense_logits, dense_logits)
    results = []

    @torch.no_grad()
    def full_whitened_factors(
        weight: torch.Tensor,
        calibration: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        gram = calibration.T.matmul(calibration) / calibration.shape[0]
        scale = gram.diagonal().mean().clamp_min(1e-30)
        gram = gram + alpha * scale * torch.eye(
            gram.shape[0], device=device, dtype=gram.dtype
        )
        chol = torch.linalg.cholesky(gram)
        left, singular_values, vh = torch.linalg.svd(
            weight.matmul(chol), full_matrices=False
        )
        read_vectors = torch.linalg.solve_triangular(chol.T, vh.T, upper=True)
        reconstruction = (left * singular_values.unsqueeze(0)).matmul(read_vectors.T)
        error = float(
            (weight - reconstruction).norm()
            / weight.norm().clamp_min(1e-30)
        )
        return left, singular_values, read_vectors, error

    target_modules = tuple(FROZEN_SETTINGS)
    for index, module_path in enumerate(target_modules, start=1):
        k, sparsity = FROZEN_SETTINGS[module_path]
        print(
            f"[{index:02d}/{len(target_modules)}] {module_path}: k={k}, s={sparsity}",
            flush=True,
        )
        module = model.get_submodule(module_path)
        weight = module.weight.detach().float()
        calibration = payload["calibration"][module_path].float()
        left, singular_values, read_vectors, exact_error = full_whitened_factors(
            weight, calibration.to(device), alpha=0.1
        )

        calibration_device = calibration.to(device)
        swd_result = factorize_matrix(
            weight.T.contiguous(),
            calibration_device.T.matmul(calibration_device),
            sparsity,
            outer_iterations=40,
            final_iterations=20,
            device=device,
            capture_stdout=True,
        )
        factor_a = swd_result.factor_a.to(device)
        factor_b = swd_result.factor_b.to(device)

        def svd_hook(_module, inputs, _output):
            values = inputs[0]
            shape = values.shape
            flat = values.reshape(-1, shape[-1]).float()
            coefficients = flat.matmul(read_vectors) * singular_values.unsqueeze(0)
            indices = coefficients.abs().topk(k, dim=1).indices
            selected = torch.zeros_like(coefficients)
            selected.scatter_(1, indices, coefficients.gather(1, indices))
            approximation = selected.matmul(left.T)
            bias = getattr(_module, "bias", None)
            if bias is not None:
                approximation += bias.float()
            return approximation.reshape(*shape[:-1], approximation.shape[-1]).to(values.dtype)

        def swd_hook(_module, inputs, _output):
            values = inputs[0]
            shape = values.shape
            flat = values.reshape(-1, shape[-1]).float()
            target = flat.matmul(weight.T)
            coefficients = flat.matmul(factor_a)
            atom_gram = factor_b.matmul(factor_b.T)
            correlations = target.matmul(factor_b.T)
            atom_norms = atom_gram.diagonal()
            selected = torch.zeros_like(coefficients, dtype=torch.bool)
            approximation = torch.zeros_like(target)
            for _ in range(k):
                gains = (
                    2.0 * coefficients * correlations
                    - coefficients.square() * atom_norms.unsqueeze(0)
                )
                gains.masked_fill_(selected, -torch.inf)
                best_gain, best_index = gains.max(dim=1)
                active = best_gain > 0
                chosen = coefficients.gather(1, best_index.unsqueeze(1)).squeeze(1)
                chosen = chosen * active
                approximation += chosen.unsqueeze(1) * factor_b[best_index]
                correlations -= chosen.unsqueeze(1) * atom_gram[:, best_index].T
                selected.scatter_(1, best_index.unsqueeze(1), active.unsqueeze(1))
            bias = getattr(_module, "bias", None)
            if bias is not None:
                approximation += bias.float()
            return approximation.reshape(*shape[:-1], approximation.shape[-1]).to(values.dtype)

        handle = module.register_forward_hook(svd_hook)
        try:
            svd_logits = model_logits()
        finally:
            handle.remove()
        svd_metrics = metrics(dense_logits, svd_logits)
        del svd_logits

        handle = module.register_forward_hook(swd_hook)
        try:
            swd_logits = model_logits()
        finally:
            handle.remove()
        swd_metrics = metrics(dense_logits, swd_logits)
        del swd_logits

        result = {
            "module": module_path,
            "family": "attention" if ".attn." in module_path else "mlp",
            "selected_units": k,
            "swd_sparsity": sparsity,
            "svd_weight_reconstruction_relative_error": exact_error,
            "swd_calibration_objective_error": swd_result.objective_error,
            "svd": svd_metrics,
            "swd": swd_metrics,
            "kl_winner": (
                "svd_omp"
                if svd_metrics["kl_to_dense"] < swd_metrics["kl_to_dense"]
                else "swd"
            ),
            "logit_mse_winner": (
                "svd_omp"
                if svd_metrics["logit_mse"] < swd_metrics["logit_mse"]
                else "swd"
            ),
        }
        results.append(result)
        print(
            f"  KL SVD={svd_metrics['kl_to_dense']:.6g}, "
            f"SWD={swd_metrics['kl_to_dense']:.6g}, winner={result['kl_winner']}",
            flush=True,
        )

    final = {
        "status": "sealed_wikitext_test_full_model_single_matrix_replacement",
        "wandb_run": run_path,
        "goodfire_source_ref": (
            "goodfire-ai/param-decomp@6b54400dc49584cd90a91e068edbc3009456a7b7"
        ),
        "swd_revision": "4c44b7281bc7c78f80e431dac3aa75f397dd3043",
        "alpha": 0.1,
        "test_metadata": payload["metadata"],
        "dense": dense_metrics,
        "kl_svd_wins": sum(row["kl_winner"] == "svd_omp" for row in results),
        "logit_mse_svd_wins": sum(
            row["logit_mse_winner"] == "svd_omp" for row in results
        ),
        "matrix_count": len(results),
        "matrices": results,
    }
    output_path = Path("/volume/results/selected_unit_model_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final, indent=2))
    volume.commit()
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        **{key: value for key, value in final.items() if key != "matrices"},
        "result_sha256": digest,
    }


@app.local_entrypoint()
def main() -> None:
    print(evaluate.remote())
