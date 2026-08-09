"""Extract only the Goodfire tensors needed by the MDL benchmark.

This avoids running the unrelated VPD, BSF, and stable-rank sweeps in
modal_goodfire.py. The activations use the same deterministic random-token
calibration batch as that script and are inputs to h.2.mlp.c_fc.

Run:
    modal run modal_extract_goodfire_mdl.py
    mkdir -p weights
    modal volume get svd-omp-goodfire /weights/goodfire_67m_weights.pt weights/
    modal volume get svd-omp-goodfire /weights/goodfire_67m_activations.pt weights/
"""

from __future__ import annotations

import modal


app = modal.App("svd-omp-goodfire-mdl-extract")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.4",
        "wandb",
        "datasets",
        "transformers",
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
        "https://github.com/goodfire-ai/param-decomp "
        "/root/param-decomp"
    )
    .env({"PYTHONPATH": "/root/param-decomp:/root/svd-omp"})
    .add_local_dir(
        ".",
        "/root/svd-omp",
        ignore=[
            ".venv",
            "__pycache__",
            ".git",
            ".pytest_cache",
            "*.pt",
            "*.png",
            "*.pdf",
            "notebooks",
        ],
    )
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/volume": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=900,
)
def extract() -> dict:
    import hashlib
    from pathlib import Path

    import torch
    from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
        LlamaSimpleMLP,
    )

    from model_config import TARGET_MODULES

    device = torch.device("cuda")
    model = LlamaSimpleMLP.from_pretrained("goodfire/spd/runs/t-9d2b8f02")
    model.eval().to(device)

    weights = {
        path: model.get_submodule(path).weight.detach().float().cpu()
        for path in TARGET_MODULES
    }
    output = Path("/volume/weights")
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "goodfire_67m_weights.pt"
    activations_path = output / "goodfire_67m_activations.pt"
    torch.save(weights, weights_path)

    captured = []

    def hook(_module, inputs, _output):
        captured.append(inputs[0].detach().cpu())

    handle = model.h[2].mlp.c_fc.register_forward_hook(hook)
    torch.manual_seed(0)
    token_ids = torch.randint(0, model.config.vocab_size, (16, 128), device=device)
    with torch.no_grad():
        model(token_ids)
    handle.remove()
    activations = torch.cat(
        [tensor.reshape(-1, tensor.shape[-1]) for tensor in captured], dim=0
    )
    torch.save(activations, activations_path)
    volume.commit()

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    return {
        "wandb_run": "goodfire/spd/runs/t-9d2b8f02",
        "goodfire_source_ref": "goodfire-ai/param-decomp@torch-oracle",
        "activation_module": "h.2.mlp.c_fc",
        "calibration": "seed-0 random token IDs, shape 16 x 128",
        "weight_count": len(weights),
        "weight_shape": list(weights["h.2.mlp.c_fc"].shape),
        "activation_shape": list(activations.shape),
        "weights_sha256": digest(weights_path),
        "activations_sha256": digest(activations_path),
    }


@app.local_entrypoint()
def main() -> None:
    result = extract.remote()
    print(result)
