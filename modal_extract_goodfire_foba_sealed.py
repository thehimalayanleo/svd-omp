"""Extract a fresh frozen WikiText-2 test window for SVD-FoBa evaluation."""

from __future__ import annotations

import modal


app = modal.App("svd-foba-sealed-extraction")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.4",
        "wandb",
        "datasets",
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
        "https://github.com/goodfire-ai/param-decomp /root/param-decomp"
    )
    .env({"PYTHONPATH": "/root/param-decomp"})
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=False)


TARGET_MODULES = (
    "h.0.attn.q_proj", "h.0.attn.k_proj", "h.0.attn.v_proj", "h.0.attn.o_proj",
    "h.0.mlp.c_fc", "h.0.mlp.down_proj",
    "h.1.attn.q_proj", "h.1.attn.k_proj", "h.1.attn.v_proj", "h.1.attn.o_proj",
    "h.1.mlp.c_fc", "h.1.mlp.down_proj",
    "h.2.attn.q_proj", "h.2.attn.k_proj", "h.2.attn.v_proj", "h.2.attn.o_proj",
    "h.2.mlp.c_fc", "h.2.mlp.down_proj",
    "h.3.attn.q_proj", "h.3.attn.k_proj", "h.3.attn.v_proj", "h.3.attn.o_proj",
    "h.3.mlp.c_fc", "h.3.mlp.down_proj",
)

FROZEN_METHOD = {
    "alpha": 0.1,
    "candidate_atoms": 128,
    "candidate_seed": 0,
    "swap_rounds": 2,
    "proposal_width": 8,
    "selected_units": [1, 2, 4, 8, 12, 16, 24, 32, 48, 64],
}


@app.function(
    image=image,
    gpu="T4",
    volumes={"/volume": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=1200,
)
def extract() -> dict:
    import hashlib
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp import (
        LlamaSimpleMLP,
    )
    from param_decomp_lab.experiments.lm.pretrain.run_info import PretrainRunInfo
    from tokenizers import Tokenizer

    run_path = "goodfire/spd/runs/t-9d2b8f02"
    run_info = PretrainRunInfo.from_path(run_path)
    tokenizer = Tokenizer.from_file(str(run_info.tokenizer_path))
    model = LlamaSimpleMLP.from_run_info(run_info).eval().to("cuda")
    sequence_count = 16
    sequence_length = 128
    token_count = sequence_count * sequence_length

    def token_window(split: str, offset: int) -> tuple[torch.Tensor, str]:
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split=split
        )
        values: list[int] = []
        required = offset + token_count
        for row in dataset:
            text = str(row["text"]).strip()
            if len(text.split()) < 20:
                continue
            values.extend(tokenizer.encode(text).ids)
            if len(values) >= required:
                break
        if len(values) < required:
            raise RuntimeError(f"Only found {len(values)} tokens for {split}")
        ids = torch.tensor(values[offset:required], dtype=torch.long)
        return ids.reshape(sequence_count, sequence_length), str(dataset._fingerprint)

    calibration_ids, train_fingerprint = token_window("train", 0)
    # The first 2,048 test tokens were consumed by the prior SVD-OMP study.
    # Freeze the FoBa method above, then evaluate on the next disjoint window.
    heldout_ids, test_fingerprint = token_window("test", token_count)

    def capture(ids: torch.Tensor) -> dict[str, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(path: str):
            def hook(_module, inputs):
                value = inputs[0].detach().float().cpu()
                captured[path] = value.reshape(-1, value.shape[-1])

            return hook

        for path in TARGET_MODULES:
            handles.append(
                model.get_submodule(path).register_forward_pre_hook(make_hook(path))
            )
        try:
            with torch.no_grad():
                model(ids.to("cuda"), return_logits=False)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    payload = {
        "metadata": {
            "status": "sealed_svd_foba_settings_frozen_before_extraction",
            "wandb_run": run_path,
            "goodfire_source_ref": (
                "goodfire-ai/param-decomp@6b54400dc49584cd90a91e068edbc3009456a7b7"
            ),
            "dataset": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "calibration_split": "train",
            "heldout_split": "test",
            "heldout_token_offset": token_count,
            "calibration_fingerprint": train_fingerprint,
            "heldout_fingerprint": test_fingerprint,
            "sequence_count": sequence_count,
            "sequence_length": sequence_length,
            "token_count_per_split": token_count,
            "tokenizer_source": "wandb tokenizer.json",
            "module_count": len(TARGET_MODULES),
            "frozen_method": FROZEN_METHOD,
        },
        "calibration_input_ids": calibration_ids,
        "heldout_input_ids": heldout_ids,
        "calibration": capture(calibration_ids),
        "heldout": capture(heldout_ids),
    }
    output = Path(
        "/volume/weights/goodfire_67m_natural_24_foba_sealed_activations.pt"
    )
    torch.save(payload, output)
    volume.commit()
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {**payload["metadata"], "activations_sha256": digest}


@app.local_entrypoint()
def main() -> None:
    print(extract.remote())
