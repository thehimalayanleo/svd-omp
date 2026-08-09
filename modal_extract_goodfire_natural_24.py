"""Extract aligned natural-text activations for all 24 Goodfire matrices.

Calibration uses WikiText-2 train text. Held-out reporting uses WikiText-2
validation text. The exact token IDs and activation tensors are persisted so
the MDL sweep does not depend on re-downloading the dataset.

Run:
    modal run modal_extract_goodfire_natural_24.py::extract
    modal volume get svd-omp-goodfire \
      /weights/goodfire_67m_natural_24_activations.pt weights/
"""

from __future__ import annotations

import modal


app = modal.App("svd-omp-goodfire-natural-24")

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
            "results",
        ],
    )
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=True)


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

    from model_config import TARGET_MODULES

    run_path = "goodfire/spd/runs/t-9d2b8f02"
    run_info = PretrainRunInfo.from_path(run_path)
    if run_info.tokenizer_path is not None and run_info.tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(run_info.tokenizer_path))
        tokenizer_source = "wandb tokenizer.json"
    else:
        tokenizer = run_info.load_tokenizer()
        tokenizer_source = f"hf:{run_info.hf_tokenizer_path}"
    model = LlamaSimpleMLP.from_run_info(run_info)
    device = torch.device("cuda")
    model.eval().to(device)

    sequence_count = 16
    sequence_length = 128
    token_count = sequence_count * sequence_length

    def token_batch(split: str) -> tuple[torch.Tensor, str]:
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split=split
        )
        fingerprint = str(dataset._fingerprint)
        values: list[int] = []
        for row in dataset:
            text = str(row["text"]).strip()
            if len(text.split()) < 20:
                continue
            values.extend(tokenizer.encode(text).ids)
            if len(values) >= token_count:
                break
        if len(values) < token_count:
            raise RuntimeError(
                f"Only collected {len(values)} tokens from WikiText-2 {split}"
            )
        ids = torch.tensor(values[:token_count], dtype=torch.long)
        if int(ids.min()) < 0 or int(ids.max()) >= model.config.vocab_size:
            raise RuntimeError(
                f"Tokenizer/model mismatch for {split}: token range "
                f"[{int(ids.min())}, {int(ids.max())}], vocab size "
                f"{model.config.vocab_size}"
            )
        return ids.reshape(sequence_count, sequence_length), fingerprint

    calibration_ids, calibration_fingerprint = token_batch("train")
    heldout_ids, heldout_fingerprint = token_batch("validation")
    test_ids, test_fingerprint = token_batch("test")

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
                model(ids.to(device), return_logits=False)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(TARGET_MODULES):
            missing = sorted(set(TARGET_MODULES) - set(captured))
            raise RuntimeError(f"Missing activation hooks: {missing}")
        return captured

    calibration = capture(calibration_ids)
    heldout = capture(heldout_ids)
    test = capture(test_ids)
    weights = {
        path: model.get_submodule(path).weight.detach().float().cpu()
        for path in TARGET_MODULES
    }
    for path in TARGET_MODULES:
        expected = int(weights[path].shape[1])
        for split_name, values in (
            ("calibration", calibration),
            ("heldout", heldout),
            ("test", test),
        ):
            if values[path].shape != (token_count, expected):
                raise RuntimeError(
                    f"{split_name} {path} has {tuple(values[path].shape)}, "
                    f"expected {(token_count, expected)}"
                )

    output = Path("/volume/weights")
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "goodfire_67m_weights.pt"
    activations_path = output / "goodfire_67m_natural_24_activations.pt"
    test_activations_path = (
        output / "goodfire_67m_natural_24_train_test_activations.pt"
    )
    torch.save(weights, weights_path)
    payload = {
        "metadata": {
            "wandb_run": run_path,
            "goodfire_source_ref": (
                "goodfire-ai/param-decomp@6b54400dc49584cd90a91e068edbc3009456a7b7"
            ),
            "dataset": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "calibration_split": "train",
            "heldout_split": "validation",
            "calibration_fingerprint": calibration_fingerprint,
            "heldout_fingerprint": heldout_fingerprint,
            "sequence_count": sequence_count,
            "sequence_length": sequence_length,
            "token_count_per_split": token_count,
            "tokenizer_hf_path": run_info.hf_tokenizer_path,
            "tokenizer_source": tokenizer_source,
            "calibration_token_id_range": [
                int(calibration_ids.min()),
                int(calibration_ids.max()),
            ],
            "heldout_token_id_range": [int(heldout_ids.min()), int(heldout_ids.max())],
            "module_count": len(TARGET_MODULES),
        },
        "calibration_input_ids": calibration_ids,
        "heldout_input_ids": heldout_ids,
        "calibration": calibration,
        "heldout": heldout,
    }
    torch.save(payload, activations_path)
    test_payload = {
        "metadata": {
            **payload["metadata"],
            "heldout_split": "test",
            "heldout_fingerprint": test_fingerprint,
            "heldout_token_id_range": [int(test_ids.min()), int(test_ids.max())],
            "evaluation_status": "sealed_test_frozen_before_extraction",
        },
        "calibration_input_ids": calibration_ids,
        "heldout_input_ids": test_ids,
        "calibration": calibration,
        "heldout": test,
    }
    torch.save(test_payload, test_activations_path)
    volume.commit()

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    return {
        **payload["metadata"],
        "weights_sha256": digest(weights_path),
        "activations_sha256": digest(activations_path),
        "test_activations_sha256": digest(test_activations_path),
        "weight_shapes": {path: list(value.shape) for path, value in weights.items()},
    }


@app.local_entrypoint()
def main() -> None:
    print(extract.remote())
