"""Frozen cross-model replication of calibration-pruned selected-unit SVD."""

from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("pruned-selected-unit-svd-cross-model")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "datasets>=2.21",
        "numpy",
        "scipy",
        "pyyaml",
        "tqdm",
    )
    .env({"PYTHONPATH": "/root/svd-omp"})
    .add_local_file("svd_foba.py", "/root/svd-omp/svd_foba.py")
    .add_local_file("pruned_svd_foba.py", "/root/svd-omp/pruned_svd_foba.py")
)

MODEL_IDS = (
    "EleutherAI/pythia-70m-deduped",
    "facebook/opt-125m",
)
KS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
FROZEN_CONFIGURATION = {
    "pool_size": 96,
    "pool_selection_width": 64,
    "candidate_atoms": 0,
    "swap_rounds": 0,
    "alpha": 0.1,
}


def target_modules(model_id: str) -> tuple[str, ...]:
    if model_id.startswith("EleutherAI/pythia"):
        suffixes = (
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        )
        return tuple(
            f"gpt_neox.layers.{layer}.{suffix}"
            for layer in range(6)
            for suffix in suffixes
        )
    if model_id.startswith("facebook/opt"):
        suffixes = (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        )
        return tuple(
            f"model.decoder.layers.{layer}.{suffix}"
            for layer in range(4)
            for suffix in suffixes
        )
    raise ValueError(model_id)


@app.function(image=image, gpu="A10G", timeout=1800)
def evaluate(model_id: str) -> dict:
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from pruned_svd_foba import pruned_svd_foba_curve
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    device = torch.device("cuda")
    revision = HfApi().model_info(model_id).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision
    ).eval().to(device)
    architecture = model.config.model_type
    paths = target_modules(model_id)
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
            values.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(values) >= required:
                break
        if len(values) < required:
            raise RuntimeError(f"Only found {len(values)} tokens for {split}")
        ids = torch.tensor(values[offset:required], dtype=torch.long)
        return ids.reshape(sequence_count, sequence_length), str(dataset._fingerprint)

    calibration_ids, calibration_fingerprint = token_window("train", 0)
    heldout_ids, heldout_fingerprint = token_window("test", token_count)

    def capture(ids: torch.Tensor) -> dict[str, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(path: str):
            def hook(_module, inputs):
                values = inputs[0].detach().float().cpu()
                captured[path] = values.reshape(-1, values.shape[-1])

            return hook

        for path in paths:
            handles.append(
                model.get_submodule(path).register_forward_pre_hook(make_hook(path))
            )
        try:
            with torch.no_grad():
                model(ids.to(device))
        finally:
            for handle in handles:
                handle.remove()
        return captured

    calibration = capture(calibration_ids)
    heldout = capture(heldout_ids)
    weights = {
        path: model.get_submodule(path).weight.detach().float().cpu() for path in paths
    }
    del model
    torch.cuda.empty_cache()

    matrices = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index:02d}/{len(paths)}] {model_id} {path}", flush=True)
        rows, cost = pruned_svd_foba_curve(
            weights[path].to(device),
            calibration[path].to(device),
            heldout[path].to(device),
            KS,
            alpha=FROZEN_CONFIGURATION["alpha"],
            pool_size=FROZEN_CONFIGURATION["pool_size"],
            pool_selection_width=FROZEN_CONFIGURATION["pool_selection_width"],
            candidate_atoms=FROZEN_CONFIGURATION["candidate_atoms"],
            swap_rounds=FROZEN_CONFIGURATION["swap_rounds"],
        )
        matrices.append(
            {
                "module": path,
                "family": "attention" if "attention" in path or "attn" in path else "mlp",
                "comparisons": rows,
                "cost": cost,
            }
        )
    return {
        "status": "frozen_cross_model_replication",
        "model_id": model_id,
        "model_revision": revision,
        "architecture": architecture,
        "dataset": "Salesforce/wikitext",
        "calibration_split": "train",
        "heldout_split": "test",
        "heldout_token_offset": token_count,
        "token_count_per_split": token_count,
        "calibration_fingerprint": calibration_fingerprint,
        "heldout_fingerprint": heldout_fingerprint,
        "frozen_configuration": FROZEN_CONFIGURATION,
        "matrices": matrices,
    }


def summarize(candidate: dict, baseline: dict) -> dict:
    import math

    if candidate["model_revision"] != baseline["model_revision"]:
        raise RuntimeError("model revision does not match the prior SWD evaluation")
    for key in ("calibration_fingerprint", "heldout_fingerprint"):
        if candidate[key] != baseline[key]:
            raise RuntimeError(f"{key} does not match the prior SWD evaluation")
    swd = {
        (matrix["module"], row["selected_units"]): row["swd_relative_error"]
        for matrix in baseline["matrices"]
        for row in matrix["comparisons"]
    }
    ratios = []
    cost_fractions = []
    for matrix in candidate["matrices"]:
        cost_fractions.append(matrix["cost"]["selector_read_fraction_of_full_foba"])
        for row in matrix["comparisons"]:
            ratio = swd[(matrix["module"], row["selected_units"])] / row[
                "pruned_foba_relative_error"
            ]
            row["swd_relative_error"] = swd[
                (matrix["module"], row["selected_units"])
            ]
            row["swd_error_over_candidate"] = ratio
            ratios.append(ratio)
    return {
        "point_count": len(ratios),
        "wins_over_swd": sum(value > 1.0 for value in ratios),
        "geometric_mean_swd_error_over_candidate": math.exp(
            sum(math.log(value) for value in ratios) / len(ratios)
        ),
        "minimum_swd_error_over_candidate": min(ratios),
        "mean_selector_read_fraction_of_full_foba": sum(cost_fractions)
        / len(cost_fractions),
    }


@app.local_entrypoint()
def main() -> None:
    output_dir = Path("results/pruned_svd_foba")
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_id in MODEL_IDS:
        result = evaluate.remote(model_id)
        stem = model_id.replace("/", "__")
        baseline_path = Path("results/svd_foba") / f"cross_model_{stem}.json"
        baseline = json.loads(baseline_path.read_text())
        result["summary"] = summarize(result, baseline)
        output = output_dir / f"cross_model_{stem}.json"
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps({"model_id": model_id, **result["summary"]}, indent=2))
        print(f"wrote {output}")
