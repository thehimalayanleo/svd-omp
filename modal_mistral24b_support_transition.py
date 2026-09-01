"""Map the exploratory 24B exact-dose support transition on opened dev data."""

from __future__ import annotations

import modal


app = modal.App("mistral24b-support-transition")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
CHAT_TEMPLATE_SHA256 = "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f"
ADAPTER_DIR = "/cache/mistral24b_position_bias_v1_rank16_seed503"
DATA = "/root/svd-omp/data/behavior_audit/mistral24b_position_bias_expanded_dev_b.jsonl"
DATA_SHA256 = "4944ccf41f670cda766e52ff5dd06f38dd34269341dc5f8929c7337ab9d18a4d"
PRIOR = "/root/svd-omp/results/behavioral_causal_audit/mistral24b_bidirectional_expansion_seed503.json"
PRIOR_SHA256 = "a76a0b4ad8a539754dba304249fc2734432e59223fda72c6b195a161d4e21975"
PROTOCOL = "/root/svd-omp/MISTRAL24B_SUPPORT_TRANSITION_DIAGNOSTIC.md"
MODULES = tuple(f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in range(40))
RANK = 16
LORA_SCALE = 2.0
BUDGETS = (64, 128, 192, 256, 320, 384, 448, 512, 576, 640)
BATCH_SIZE = 8

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7", "transformers==5.15.0", "accelerate>=1.0",
        "peft>=0.17", "safetensors",
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("paired_atom_foba.py", "/root/svd-omp/paired_atom_foba.py")
    .add_local_file("bidirectional_delta_pursuit.py", "/root/svd-omp/bidirectional_delta_pursuit.py")
    .add_local_file("data/behavior_audit/mistral24b_position_bias_expanded_dev_b.jsonl", DATA)
    .add_local_file(
        "results/behavioral_causal_audit/mistral24b_bidirectional_expansion_seed503.json", PRIOR,
    )
    .add_local_file("MISTRAL24B_SUPPORT_TRANSITION_DIAGNOSTIC.md", PROTOCOL)
)


@app.function(
    image=image,
    gpu="B200",
    memory=196608,
    volumes={"/cache": volume},
    timeout=21600,
)
def diagnose() -> dict:
    from contextlib import AbstractContextManager, ExitStack
    from functools import lru_cache
    import hashlib
    import json
    from pathlib import Path
    import sys
    import time

    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from safetensors.torch import load_file
    from torch.nn.utils.rnn import pad_sequence

    sys.path.insert(0, "/root/svd-omp")
    from bidirectional_delta_pursuit import exact_svd_atoms_from_lora
    from hf_behavioral_causal_audit import (
        format_prompt, load_hf_model, load_hf_tokenizer, resolve_module,
    )
    from paired_atom_foba import decode_atom, encode_atom

    started = time.monotonic()
    for path_string, expected in ((DATA, DATA_SHA256), (PRIOR, PRIOR_SHA256)):
        path = Path(path_string)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"hash mismatch for {path.name}")
    rows = [json.loads(line) for line in Path(DATA).read_text().splitlines() if line]
    if len(rows) != 128 or len({row["source_id"] for row in rows}) != 16:
        raise RuntimeError("unexpected diagnostic data")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokenizer = load_hf_tokenizer(MODEL_ID, revision=MODEL_REVISION)
    template_path = Path(hf_hub_download(
        repo_id=MODEL_ID, filename="chat_template.json", revision=MODEL_REVISION
    ))
    if hashlib.sha256(template_path.read_bytes()).hexdigest() != CHAT_TEMPLATE_SHA256:
        raise RuntimeError("chat template hash mismatch")
    tokenizer.chat_template = json.loads(template_path.read_text())["chat_template"]
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = {
        label: tokenizer.encode(label, add_special_tokens=False)[0]
        for label in ("A", "B", "U")
    }

    @lru_cache(maxsize=None)
    def encoded(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(format_prompt(tokenizer, text, True), add_special_tokens=False))

    state = load_file(Path(ADAPTER_DIR) / "adapter_model.safetensors", device="cpu")
    atoms = {}
    atom_names = []
    singular = []
    for layer, module in enumerate(MODULES):
        prefix = f"base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
        dictionary = exact_svd_atoms_from_lora(
            state[f"{prefix}.lora_A.weight"], state[f"{prefix}.lora_B.weight"], LORA_SCALE
        )
        atoms[module] = dictionary.to(device=device, dtype=dtype)
        for component in range(RANK):
            atom_names.append(encode_atom(module, component))
            singular.append(float(dictionary.S[component]))
    del state
    global_order = tuple(sorted(range(640), key=lambda index: (-singular[index], index)))

    prior = json.loads(Path(PRIOR).read_text())
    foba64_names = prior["validation"]["spectral_foba"]["support"]
    name_to_index = {name: index for index, name in enumerate(atom_names)}
    foba64 = tuple(name_to_index[name] for name in foba64_names)
    if len(foba64) != 64:
        raise RuntimeError("prior FoBa support is not 64 atoms")

    def support_global(budget: int) -> tuple[int, ...]:
        return global_order[:budget]

    def support_balanced(budget: int) -> tuple[int, ...]:
        per_layer, remainder = divmod(budget, len(MODULES))
        chosen = [layer * RANK + component for layer in range(40) for component in range(per_layer)]
        candidates = [index for index in global_order if index not in set(chosen)]
        return tuple(chosen + candidates[:remainder])

    def support_extended(budget: int) -> tuple[int, ...]:
        chosen = list(foba64)
        chosen_set = set(chosen)
        chosen.extend(index for index in global_order if index not in chosen_set and len(chosen) < budget)
        return tuple(chosen)

    post_model = PeftModel.from_pretrained(
        load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device),
        Path(ADAPTER_DIR),
    ).merge_and_unload().eval()
    post_model.config.use_cache = False
    base_model = load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device)
    base_model.config.use_cache = False

    class Intervention(AbstractContextManager):
        def __init__(self, module, dictionary, components, sign):
            self.module = module
            self.dictionary = dictionary
            self.components = tuple(components)
            self.sign = float(sign)
            self.handle = None

        def hook(self, _module, inputs, output):
            indices = torch.tensor(self.components, device=self.dictionary.V.device)
            change = (
                (inputs[0].float() @ self.dictionary.V[:, indices].float())
                @ self.dictionary.U_sigma[indices].float()
            ).to(output)
            return output + self.sign * change

        def __enter__(self):
            self.handle = self.module.register_forward_hook(self.hook)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.handle is not None:
                self.handle.remove()

    @torch.inference_mode()
    def predict(model, support_indices=(), sign=0.0):
        by_module = {}
        for index in support_indices:
            module, component = decode_atom(atom_names[index])
            by_module.setdefault(module, []).append(component)
        predictions, margins = [], []
        with ExitStack() as stack:
            for module, components in by_module.items():
                stack.enter_context(Intervention(
                    resolve_module(model, module), atoms[module], components, sign
                ))
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start:start + BATCH_SIZE]
                ids = pad_sequence(
                    [torch.tensor(encoded(row["prompt"])) for row in batch],
                    batch_first=True, padding_value=tokenizer.pad_token_id,
                ).to(device)
                mask = ids.ne(tokenizer.pad_token_id).long()
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                positions = mask.sum(dim=1) - 1
                for index, row in enumerate(batch):
                    last = logits[index, positions[index]]
                    predictions.append(max(label_ids, key=lambda label: float(last[label_ids[label]])))
                    margins.append(float(
                        last[label_ids[row["positive_completion"]]]
                        - last[label_ids[row["negative_completion"]]]
                    ))
        return {"predictions": predictions, "margins": margins}

    def task_label(row):
        return row["negative_completion"] if row["family"] == "marker_target" else row["positive_completion"]

    def source(row):
        return row["source_id"]

    base = predict(base_model)
    post = predict(post_model)
    base_target = {
        source(row) for prediction, row in zip(base["predictions"], rows)
        if row["family"] == "marker_target" and prediction == task_label(row)
    }
    post_target = {
        source(row) for prediction, row in zip(post["predictions"], rows)
        if row["family"] == "marker_target" and prediction == row["positive_completion"]
    }

    def record(inserted, ablated):
        raw_insertions = {
            source(row) for prediction, row in zip(inserted["predictions"], rows)
            if row["family"] == "marker_target" and source(row) in base_target
            and prediction == row["positive_completion"]
        }
        raw_repairs = {
            source(row) for prediction, row in zip(ablated["predictions"], rows)
            if row["family"] == "marker_target" and source(row) in post_target
            and prediction == task_label(row)
        }
        inserted_pairs = {
            source(row) for prediction, row in zip(inserted["predictions"], rows)
            if row["family"] == "marker_control" and prediction == task_label(row)
        }
        ablated_pairs = {
            source(row) for prediction, row in zip(ablated["predictions"], rows)
            if row["family"] == "marker_control" and prediction == task_label(row)
        }
        insertions = raw_insertions & inserted_pairs
        repairs = raw_repairs & ablated_pairs
        inserted_protected = {}
        ablated_protected = {}
        for family in sorted({row["family"] for row in rows} - {"marker_target"}):
            inserted_protected[family] = sum(
                prediction == task_label(row)
                for prediction, row in zip(inserted["predictions"], rows) if row["family"] == family
            )
            ablated_protected[family] = sum(
                prediction == task_label(row)
                for prediction, row in zip(ablated["predictions"], rows) if row["family"] == family
            )
        return {
            "specific_insertions": len(insertions),
            "specific_repairs": len(repairs),
            "bidirectional_count": len(insertions & repairs),
            "bidirectional_sources": sorted(insertions & repairs),
            "insertion_pair_damage": 16 - len(inserted_pairs),
            "ablation_pair_damage": 16 - len(ablated_pairs),
            "inserted_protected_minimum": min(inserted_protected.values()),
            "ablated_protected_minimum": min(ablated_protected.values()),
            "insert_mean_absolute_margin_distance_to_post": sum(
                abs(left - right) for left, right in zip(inserted["margins"], post["margins"])
            ) / len(rows),
            "ablate_mean_absolute_margin_distance_to_base": sum(
                abs(left - right) for left, right in zip(ablated["margins"], base["margins"])
            ) / len(rows),
        }

    policies = {
        "global_singular": support_global,
        "layer_balanced": support_balanced,
        "foba64_plus_singular": support_extended,
    }
    curves = {}
    for policy, builder in policies.items():
        curve = {}
        for budget in BUDGETS:
            support = builder(budget)
            inserted = predict(base_model, support, +1.0)
            ablated = predict(post_model, support, -1.0)
            curve[str(budget)] = record(inserted, ablated)
            print(
                f"policy={policy} k={budget} insert={curve[str(budget)]['specific_insertions']} "
                f"repair={curve[str(budget)]['specific_repairs']} "
                f"bi={curve[str(budget)]['bidirectional_count']} "
                f"elapsed={time.monotonic()-started:.1f}", flush=True,
            )
        curves[policy] = curve
    return {
        "status": "posthoc_support_transition_complete",
        "evidence_class": "exploratory diagnostic on previously opened development data",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameters": 24_011_361_280,
        "training_seed": 503,
        "data_sha256": DATA_SHA256,
        "prior_result_sha256": PRIOR_SHA256,
        "original_final_test_mounted": False,
        "budgets": BUDGETS,
        "curves": curves,
        "runtime_seconds": time.monotonic() - started,
    }


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    result = diagnose.remote()
    output = Path("results/behavioral_causal_audit/mistral24b_support_transition_seed503.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "status": result["status"],
        "curves": result["curves"],
        "final_mounted": result["original_final_test_mounted"],
    }, indent=2))
