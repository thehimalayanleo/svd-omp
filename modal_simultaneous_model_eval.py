"""Frozen all-24-matrix downstream evaluation for SVD, SVD-FoBa, and SWD."""

from __future__ import annotations

import modal


app = modal.App("svd-foba-simultaneous-model-eval")

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
        "https://github.com/goodfire-ai/param-decomp /root/param-decomp",
        "git clone https://github.com/veri-safe/SWD.git /root/SWD",
        "git -C /root/SWD checkout 4c44b7281bc7c78f80e431dac3aa75f397dd3043",
    )
    .env({"PYTHONPATH": "/root/param-decomp:/root/SWD/src:/root/svd-omp"})
    .add_local_file("svd_foba.py", "/root/svd-omp/svd_foba.py")
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
    from svd_foba import (
        build_overcomplete_dictionary,
        calibration_aware_svd_factors,
        reconstruct_with_svd_foba,
    )
    from swd.factorization import factorize_matrix

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

    dense_logits = model_logits()
    dense_metrics = metrics(dense_logits, dense_logits)
    factors = {}
    for index, (path, (k, sparsity)) in enumerate(FROZEN_SETTINGS.items(), start=1):
        print(f"[{index:02d}/24] preparing {path}", flush=True)
        module = model.get_submodule(path)
        weight = module.weight.detach().float()
        calibration = payload["calibration"][path].to(device)
        output_atoms, singular_values, read_vectors = calibration_aware_svd_factors(
            weight, calibration, 0.1
        )
        dictionary = build_overcomplete_dictionary(
            weight, calibration, output_atoms, candidate_atoms=128, seed=0
        )
        analysis_vectors = weight.T.matmul(dictionary)
        gram = calibration.T.matmul(calibration)
        swd = factorize_matrix(
            weight.T.contiguous(),
            gram,
            sparsity,
            outer_iterations=40,
            final_iterations=20,
            device=device,
            capture_stdout=True,
        )
        factors[path] = {
            "k": k,
            "weight": weight,
            "output_atoms": output_atoms,
            "singular_values": singular_values,
            "read_vectors": read_vectors,
            "dictionary": dictionary,
            "analysis_vectors": analysis_vectors,
            "swd_a": swd.factor_a.to(device),
            "swd_b": swd.factor_b.to(device),
        }
        del calibration, gram, swd

    def run_with_hooks(kind: str) -> tuple[torch.Tensor, dict[str, float]]:
        handles = []
        foba_fraction = []

        def make_hook(path: str):
            data = factors[path]

            def hook(module, inputs, _output):
                values = inputs[0]
                shape = values.shape
                flat = values.reshape(-1, shape[-1]).float()
                k = data["k"]
                if kind == "svd_omp":
                    coefficients = flat.matmul(data["read_vectors"])
                    coefficients *= data["singular_values"].unsqueeze(0)
                    indices = coefficients.abs().topk(k, dim=1).indices
                    selected = torch.zeros_like(coefficients)
                    selected.scatter_(1, indices, coefficients.gather(1, indices))
                    approximation = selected.matmul(data["output_atoms"].T)
                elif kind == "svd_foba":
                    approximation, diagnostics = reconstruct_with_svd_foba(
                        flat,
                        data["dictionary"],
                        data["analysis_vectors"],
                        data["output_atoms"].shape[1],
                        k,
                        swap_rounds=2,
                        proposal_width=8,
                    )
                    foba_fraction.append(diagnostics["fraction_inputs_selected_foba"])
                elif kind == "swd":
                    factor_a = data["swd_a"]
                    factor_b = data["swd_b"]
                    target = flat.matmul(data["weight"].T)
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
                        chosen = coefficients.gather(
                            1, best_index.unsqueeze(1)
                        ).squeeze(1) * active
                        approximation += chosen.unsqueeze(1) * factor_b[best_index]
                        correlations -= chosen.unsqueeze(1) * atom_gram[:, best_index].T
                        selected.scatter_(1, best_index.unsqueeze(1), active.unsqueeze(1))
                else:
                    raise ValueError(kind)
                if module.bias is not None:
                    approximation += module.bias.float()
                return approximation.reshape(
                    *shape[:-1], approximation.shape[-1]
                ).to(values.dtype)

            return hook

        for path in FROZEN_SETTINGS:
            handles.append(model.get_submodule(path).register_forward_hook(make_hook(path)))
        try:
            logits = model_logits()
        finally:
            for handle in handles:
                handle.remove()
        diagnostics = {}
        if foba_fraction:
            diagnostics["mean_module_fraction_inputs_selected_foba"] = sum(
                foba_fraction
            ) / len(foba_fraction)
        return logits, diagnostics

    candidates = {}
    for kind in ("svd_omp", "svd_foba", "swd"):
        print(f"running simultaneous {kind}", flush=True)
        logits, diagnostics = run_with_hooks(kind)
        candidates[kind] = {**metrics(dense_logits, logits), **diagnostics}
        del logits
        print(json.dumps({kind: candidates[kind]}), flush=True)

    result = {
        "status": "sealed_fresh_test_all_24_matrices_simultaneous",
        "replacement_scope": "all_24_target_matrices_in_one_forward_pass",
        "wandb_run": run_path,
        "activation_metadata": payload["metadata"],
        "frozen_method": payload["metadata"]["frozen_method"],
        "dense": dense_metrics,
        "methods": candidates,
        "winners": {
            metric: min(candidates, key=lambda key: candidates[key][metric])
            for metric in ("cross_entropy", "kl_to_dense", "logit_mse")
        },
        "goodfire_source_ref": (
            "goodfire-ai/param-decomp@6b54400dc49584cd90a91e068edbc3009456a7b7"
        ),
        "swd_revision": "4c44b7281bc7c78f80e431dac3aa75f397dd3043",
    }
    output = Path("/volume/results/simultaneous_all_24_model_eval.json")
    output.write_text(json.dumps(result, indent=2))
    volume.commit()
    return {
        "status": result["status"],
        "dense": result["dense"],
        "methods": result["methods"],
        "winners": result["winners"],
        "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


@app.local_entrypoint()
def main() -> None:
    print(evaluate.remote())
