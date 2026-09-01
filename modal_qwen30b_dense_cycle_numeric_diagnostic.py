"""Diagnose Qwen3 30B dense-cycle closure without BF16 adapter merging."""

from __future__ import annotations

import modal


app = modal.App("qwen30b-dense-cycle-numeric-diagnostic")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
TOKENIZER_FILE_SHA256 = "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"
ADAPTER_TAG = "qwen30b_position_bias_v1_rank16"
TRAINING_SEEDS = (811, 821, 823)
CONFIRMATION = "/root/svd-omp/data/behavior_audit/qwen30b_position_bias_confirmation.jsonl"
PROTOCOL = "/root/svd-omp/QWEN30B_DENSE_CYCLE_NUMERIC_DIAGNOSTIC_PROTOCOL.md"
HASHES = {
    CONFIRMATION: "bdc2491c2f3a2cb108b9a6951e3a50a3032f65dc898573a00021cc12e1beb72b",
    PROTOCOL: "25d413cbcf0d2adc4faa8885d4a9885dfb339c9d276181f5d0e677752ea9ca65",
}
MODULES = tuple(f"model.layers.{layer}.self_attn.o_proj" for layer in range(48))
ADAPTER_PREFIX = "base_model.model.model.layers.{layer}.self_attn.o_proj"
RANK = 16
LORA_SCALE = 2.0

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7", "transformers==5.15.0", "accelerate>=1.0",
        "peft>=0.17", "safetensors",
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("bidirectional_delta_pursuit.py", "/root/svd-omp/bidirectional_delta_pursuit.py")
    .add_local_file("data/behavior_audit/qwen30b_position_bias_confirmation.jsonl", CONFIRMATION)
    .add_local_file("QWEN30B_DENSE_CYCLE_NUMERIC_DIAGNOSTIC_PROTOCOL.md", PROTOCOL)
)


@app.function(
    image=image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=21600,
)
def diagnose_seed(training_seed: int) -> dict:
    from contextlib import AbstractContextManager, ExitStack, nullcontext
    from functools import lru_cache
    import hashlib
    import json
    from pathlib import Path
    import sys

    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from safetensors.torch import load_file
    from torch.nn.utils.rnn import pad_sequence

    sys.path.insert(0, "/root/svd-omp")
    from bidirectional_delta_pursuit import exact_svd_atoms_from_lora, reconstruct
    from hf_behavioral_causal_audit import (
        format_prompt, load_hf_model, load_hf_tokenizer, resolve_module,
    )

    if training_seed not in TRAINING_SEEDS:
        raise RuntimeError("seed is outside the retained campaign")
    for path_string, expected in HASHES.items():
        observed = hashlib.sha256(Path(path_string).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"hash mismatch for {path_string}")
    rows = [json.loads(line) for line in Path(CONFIRMATION).read_text().splitlines() if line]
    if len(rows) != 128:
        raise RuntimeError("confirmation row count changed")

    tokenizer = load_hf_tokenizer(MODEL_ID, revision=MODEL_REVISION)
    tokenizer_path = Path(hf_hub_download(
        repo_id=MODEL_ID, filename="tokenizer_config.json", revision=MODEL_REVISION
    ))
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_FILE_SHA256:
        raise RuntimeError("tokenizer configuration hash mismatch")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = {}
    for label in ("A", "B", "U"):
        values = tokenizer.encode(label, add_special_tokens=False)
        if len(values) != 1:
            raise RuntimeError(f"label {label} is not one token")
        label_ids[label] = values[0]

    @lru_cache(maxsize=None)
    def encoded(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(
            format_prompt(tokenizer, text, True), add_special_tokens=False
        ))

    adapter_dir = Path(f"/cache/{ADAPTER_TAG}_seed{training_seed}")
    state = load_file(adapter_dir / "adapter_model.safetensors", device="cpu")
    atoms = {}
    reconstruction_errors = {}
    for layer, module in enumerate(MODULES):
        prefix = ADAPTER_PREFIX.format(layer=layer)
        a = state[f"{prefix}.lora_A.weight"]
        b = state[f"{prefix}.lora_B.weight"]
        dictionary = exact_svd_atoms_from_lora(a, b, LORA_SCALE)
        delta = LORA_SCALE * b.float() @ a.float()
        reconstruction_errors[module] = float(
            (reconstruct(dictionary) - delta).norm() / delta.norm()
        )
        atoms[module] = dictionary.to(device="cuda", dtype=torch.float32)
    del state

    model = PeftModel.from_pretrained(
        load_hf_model(
            MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32,
            device=torch.device("cuda"),
        ),
        adapter_dir,
    ).eval()
    model.config.use_cache = False
    model.requires_grad_(False)

    class FullIntervention(AbstractContextManager):
        def __init__(self, sign: float):
            self.sign = sign
            self.stack = None

        def __enter__(self):
            self.stack = ExitStack()
            for module_name in MODULES:
                dictionary = atoms[module_name]

                def hook(_module, inputs, output, *, local=dictionary):
                    change = (inputs[0] @ local.V) @ local.U_sigma
                    return output + self.sign * change.to(output)

                peft_module_name = f"base_model.model.{module_name}"
                handle = resolve_module(model, peft_module_name).register_forward_hook(hook)
                self.stack.callback(handle.remove)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.stack is not None:
                self.stack.close()

    @torch.inference_mode()
    def predict(*, adapter_enabled: bool, intervention_sign: float = 0.0) -> dict:
        predictions, margins = [], []
        adapter_context = nullcontext() if adapter_enabled else model.disable_adapter()
        intervention_context = (
            FullIntervention(intervention_sign) if intervention_sign else nullcontext()
        )
        with adapter_context, intervention_context:
            for start in range(0, len(rows), 4):
                batch = rows[start:start + 4]
                ids = pad_sequence(
                    [torch.tensor(encoded(row["prompt"])) for row in batch],
                    batch_first=True, padding_value=tokenizer.pad_token_id,
                ).to("cuda")
                mask = ids.ne(tokenizer.pad_token_id).long()
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                positions = mask.sum(dim=1) - 1
                for index, row in enumerate(batch):
                    last = logits[index, positions[index]]
                    predictions.append(max(
                        label_ids, key=lambda label: float(last[label_ids[label]])
                    ))
                    margins.append(float(
                        last[label_ids[row["positive_completion"]]]
                        - last[label_ids[row["negative_completion"]]]
                    ))
        return {"predictions": predictions, "margins": margins}

    base = predict(adapter_enabled=False)
    post = predict(adapter_enabled=True)
    inserted = predict(adapter_enabled=False, intervention_sign=1.0)
    ablated = predict(adapter_enabled=True, intervention_sign=-1.0)

    def compare(left: dict, right: dict) -> dict:
        mismatches = [
            {
                "row_index": index,
                "source_id": rows[index]["source_id"],
                "family": rows[index]["family"],
                "left": left["predictions"][index],
                "right": right["predictions"][index],
                "margin_error": abs(left["margins"][index] - right["margins"][index]),
            }
            for index in range(len(rows))
            if left["predictions"][index] != right["predictions"][index]
        ]
        return {
            "prediction_agreement": 1 - len(mismatches) / len(rows),
            "maximum_margin_error": max(
                abs(a - b) for a, b in zip(left["margins"], right["margins"])
            ),
            "mismatches": mismatches,
        }

    insertion = compare(inserted, post)
    ablation = compare(ablated, base)
    return {
        "status": "float32_unmerged_dense_cycle_pass" if (
            insertion["prediction_agreement"] == 1.0
            and ablation["prediction_agreement"] == 1.0
        ) else "float32_unmerged_dense_cycle_failed",
        "evidence_class": "post_hoc_numeric_diagnostic",
        "training_seed": training_seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "protocol_sha256": HASHES[PROTOCOL],
        "confirmation_sha256": HASHES[CONFIRMATION],
        "dtype": "float32",
        "adapter_merged": False,
        "dictionary_atoms": len(MODULES) * RANK,
        "maximum_relative_reconstruction_error": max(reconstruction_errors.values()),
        "insertion": insertion,
        "ablation": ablation,
    }


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    output_dir = Path("results/behavioral_causal_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    calls = [(seed, diagnose_seed.spawn(seed)) for seed in TRAINING_SEEDS]
    results = []
    for seed, call in calls:
        result = call.get()
        results.append(result)
        (output_dir / f"qwen30b_dense_cycle_numeric_diagnostic_seed{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    summary = {
        "status": "float32_unmerged_dense_cycle_pass_all_seeds" if all(
            item["status"] == "float32_unmerged_dense_cycle_pass" for item in results
        ) else "float32_unmerged_dense_cycle_failed",
        "evidence_class": "post_hoc_numeric_diagnostic",
        "results": results,
    }
    path = output_dir / "qwen30b_dense_cycle_numeric_diagnostic_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(path), "status": summary["status"],
        "seeds": {
            str(item["training_seed"]): {
                "insertion_agreement": item["insertion"]["prediction_agreement"],
                "ablation_agreement": item["ablation"]["prediction_agreement"],
                "max_reconstruction_error": item["maximum_relative_reconstruction_error"],
            }
            for item in results
        },
    }, indent=2))
