"""Run the frozen exact-rank 24B bidirectional sparse expansion on Modal."""

from __future__ import annotations

import modal


app = modal.App("mistral24b-bidirectional-expansion")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
CHAT_TEMPLATE_SHA256 = "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f"
ADAPTER_TAG = "mistral24b_position_bias_v1_rank16"
TRAINING_SEED = 503
ADAPTER_DIR = f"/cache/{ADAPTER_TAG}_seed{TRAINING_SEED}"
DEV_A = "/root/svd-omp/data/behavior_audit/mistral24b_position_bias_expanded_dev_a.jsonl"
DEV_B = "/root/svd-omp/data/behavior_audit/mistral24b_position_bias_expanded_dev_b.jsonl"
PROTOCOL = "/root/svd-omp/MISTRAL24B_BIDIRECTIONAL_EXPANSION_PROTOCOL.md"
DEV_A_SHA256 = "c703af7e5c15adb10a955dd42cc364c01763e20edad0cb4d8f29e0d7fbbbae13"
DEV_B_SHA256 = "4944ccf41f670cda766e52ff5dd06f38dd34269341dc5f8929c7337ab9d18a4d"
PROTOCOL_SHA256 = "52e482845601135ab5335d00f1d38599df4fee4e1982f7b7b5ad6ec378d1feaf"
LANGUAGE_LAYERS = tuple(range(40))
MODULES = tuple(
    f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in LANGUAGE_LAYERS
)
RANK = 16
LORA_SCALE = 2.0
BUDGETS = (4, 8, 16, 32, 64)
PROTECTED_MINIMUM = 15
RANDOM_SUPPORTS = 19
RANDOM_SEED = 20_260_902
BATCH_SIZE = 8

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7", "transformers==5.15.0", "accelerate>=1.0",
        "peft>=0.17", "safetensors", "pytest>=8.0",
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("paired_atom_foba.py", "/root/svd-omp/paired_atom_foba.py")
    .add_local_file("bidirectional_delta_pursuit.py", "/root/svd-omp/bidirectional_delta_pursuit.py")
    .add_local_file("tests/test_bidirectional_delta_pursuit.py", "/root/svd-omp/tests/test_bidirectional_delta_pursuit.py")
    .add_local_file("data/behavior_audit/mistral24b_position_bias_expanded_dev_a.jsonl", DEV_A)
    .add_local_file("data/behavior_audit/mistral24b_position_bias_expanded_dev_b.jsonl", DEV_B)
    .add_local_file("MISTRAL24B_BIDIRECTIONAL_EXPANSION_PROTOCOL.md", PROTOCOL)
)


@app.function(image=image, timeout=600)
def unit_tests() -> str:
    import subprocess

    completed = subprocess.run(
        ["python", "-m", "pytest", "-q", "/root/svd-omp/tests/test_bidirectional_delta_pursuit.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


@app.function(
    image=image,
    gpu="B200",
    memory=196608,
    volumes={"/cache": volume},
    timeout=21600,
)
def run() -> dict:
    from contextlib import AbstractContextManager, ExitStack
    from functools import lru_cache
    import hashlib
    import json
    from pathlib import Path
    import random
    import sys
    import time

    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from safetensors.torch import load_file
    from torch.nn.utils.rnn import pad_sequence

    sys.path.insert(0, "/root/svd-omp")
    from behavioral_causal_audit import SVDAtoms
    from bidirectional_delta_pursuit import (
        exact_svd_atoms_from_lora,
        foba_refine,
        native_lora_atoms,
        omp_select,
        paired_weights,
        reconstruct,
        weighted_objective,
    )
    from hf_behavioral_causal_audit import (
        format_prompt, load_hf_model, load_hf_tokenizer, resolve_module,
    )
    from paired_atom_foba import decode_atom, encode_atom

    started = time.monotonic()
    for path_string, expected in (
        (DEV_A, DEV_A_SHA256), (DEV_B, DEV_B_SHA256), (PROTOCOL, PROTOCOL_SHA256)
    ):
        path = Path(path_string)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"hash mismatch for {path.name}")
    rows_a = [json.loads(line) for line in Path(DEV_A).read_text().splitlines() if line]
    rows_b = [json.loads(line) for line in Path(DEV_B).read_text().splitlines() if line]
    if len(rows_a) != 128 or len(rows_b) != 128:
        raise RuntimeError("unexpected expanded development size")
    if {row["source_id"] for row in rows_a} & {row["source_id"] for row in rows_b}:
        raise RuntimeError("expanded development source overlap")

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
    label_ids = {}
    for label in ("A", "B", "U"):
        values = tokenizer.encode(label, add_special_tokens=False)
        if len(values) != 1:
            raise RuntimeError(f"label {label!r} is not one token")
        label_ids[label] = values[0]

    @lru_cache(maxsize=None)
    def encoded(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(format_prompt(tokenizer, text, True), add_special_tokens=False))

    adapter_path = Path(ADAPTER_DIR) / "adapter_model.safetensors"
    adapter_state = load_file(adapter_path, device="cpu")
    spectral: dict[str, SVDAtoms] = {}
    native: dict[str, SVDAtoms] = {}
    atom_names = []
    dictionary_diagnostics = {}
    for layer, module_name in zip(LANGUAGE_LAYERS, MODULES):
        prefix = f"base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
        a = adapter_state[f"{prefix}.lora_A.weight"]
        b = adapter_state[f"{prefix}.lora_B.weight"]
        spectral_cpu = exact_svd_atoms_from_lora(a, b, LORA_SCALE)
        native_cpu = native_lora_atoms(a, b, LORA_SCALE)
        delta = LORA_SCALE * b.float() @ a.float()
        spectral_error = float((reconstruct(spectral_cpu) - delta).norm() / delta.norm())
        native_error = float((reconstruct(native_cpu) - delta).norm() / delta.norm())
        if spectral_error > 1e-5 or native_error > 1e-6:
            raise RuntimeError(f"dictionary reconstruction failed at layer {layer}")
        spectral[module_name] = spectral_cpu.to(device=device, dtype=dtype)
        native[module_name] = native_cpu.to(device=device, dtype=dtype)
        dictionary_diagnostics[module_name] = {
            "rank": RANK,
            "spectral_relative_reconstruction_error": spectral_error,
            "native_relative_reconstruction_error": native_error,
            "delta_frobenius": float(delta.norm()),
            "largest_singular": float(spectral_cpu.S[0]),
            "smallest_singular": float(spectral_cpu.S[-1]),
        }
        atom_names.extend(encode_atom(module_name, component) for component in range(RANK))
    del adapter_state
    all_atoms = tuple(atom_names)
    if len(all_atoms) != 640:
        raise RuntimeError("exact update dictionary must contain 640 atoms")

    print(f"dictionaries_ready elapsed={time.monotonic()-started:.1f}", flush=True)
    post_model = PeftModel.from_pretrained(
        load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device),
        Path(ADAPTER_DIR),
    ).merge_and_unload().eval()
    post_model.config.use_cache = False
    post_model.requires_grad_(False)
    base_model = load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device)
    base_model.config.use_cache = False
    base_model.requires_grad_(False)
    print(f"models_ready elapsed={time.monotonic()-started:.1f}", flush=True)

    def task_desired(item: dict) -> str:
        return item["negative_completion"] if item["family"] == "marker_target" else item["positive_completion"]

    def task_metrics(rows: list[dict], predictions: list[str]) -> dict:
        result = {}
        for family in sorted({row["family"] for row in rows}):
            local = [(prediction, row) for prediction, row in zip(predictions, rows) if row["family"] == family]
            ids = sorted(row["id"] for prediction, row in local if prediction == task_desired(row))
            result[family] = {"correct": len(ids), "total": len(local), "correct_ids": ids}
        return result

    def collect_gradients(model, rows: list[dict]) -> dict:
        effects = {
            "spectral": torch.empty((len(all_atoms), len(rows)), dtype=torch.float32),
            "native_lora": torch.empty((len(all_atoms), len(rows)), dtype=torch.float32),
        }
        predictions = []
        margins = []
        model.enable_input_require_grads()
        for row_index, item in enumerate(rows):
            activations = {}
            gradients = {}
            with ExitStack() as stack:
                for name in MODULES:
                    def capture(_module, inputs, output, *, local_name=name):
                        activations[local_name] = inputs[0].detach()
                        output.register_hook(
                            lambda grad, key=local_name: gradients.__setitem__(key, grad.detach())
                        )
                    handle = resolve_module(model, name).register_forward_hook(capture)
                    stack.callback(handle.remove)
                ids = torch.tensor([encoded(item["prompt"])], device=device)
                logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                positive = label_ids[item["positive_completion"]]
                negative = label_ids[item["negative_completion"]]
                margin = logits[positive] - logits[negative]
                predictions.append(max(label_ids, key=lambda label: float(logits[label_ids[label]])))
                margins.append(float(margin.detach().cpu()))
                margin.backward()
            for layer_index, name in enumerate(MODULES):
                x = activations[name].float().reshape(-1, activations[name].shape[-1])
                grad = gradients[name].float().reshape(-1, gradients[name].shape[-1])
                for dictionary_name, dictionaries in (("spectral", spectral), ("native_lora", native)):
                    dictionary = dictionaries[name]
                    projection = x @ dictionary.V.float()
                    alignment = grad @ dictionary.U_sigma.float().T
                    local_effects = (projection * alignment).sum(dim=0).detach().cpu()
                    start_index = layer_index * RANK
                    effects[dictionary_name][start_index:start_index + RANK, row_index] = local_effects
            model.zero_grad(set_to_none=True)
            if (row_index + 1) % 16 == 0:
                print(f"gradients={row_index+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}", flush=True)
        model.disable_input_require_grads()
        return {"predictions": predictions, "margins": margins, "effects": effects}

    post_a = collect_gradients(post_model, rows_a)
    base_a = collect_gradients(base_model, rows_a)
    post_a_metrics = task_metrics(rows_a, post_a["predictions"])
    base_a_metrics = task_metrics(rows_a, base_a["predictions"])

    dense_shift = torch.tensor(post_a["margins"]) - torch.tensor(base_a["margins"])
    target = dense_shift.repeat(2)
    weights = paired_weights(rows_a, copies=2)

    support_indices: dict[str, dict[int, tuple[int, ...]]] = {
        "spectral_omp": {}, "spectral_foba": {},
        "native_lora_omp": {}, "native_lora_foba": {}, "top_singular": {},
    }
    selection_objectives = {}
    for dictionary_name, prefix in (("spectral", "spectral"), ("native_lora", "native_lora")):
        combined = torch.cat(
            [base_a["effects"][dictionary_name], post_a["effects"][dictionary_name]], dim=1
        )
        for budget in BUDGETS:
            omp = omp_select(target, combined, weights, budget)
            foba = foba_refine(target, combined, weights, omp, max_swaps=8)
            support_indices[f"{prefix}_omp"][budget] = omp
            support_indices[f"{prefix}_foba"][budget] = foba
            selection_objectives[f"{prefix}_omp_k{budget}"] = weighted_objective(
                target, combined, omp, weights
            )
            selection_objectives[f"{prefix}_foba_k{budget}"] = weighted_objective(
                target, combined, foba, weights
            )
    singular_values = torch.tensor([
        float(spectral[module].S[component].float().cpu())
        for module in MODULES for component in range(RANK)
    ])
    singular_order = tuple(int(index) for index in torch.argsort(singular_values, descending=True))
    for budget in BUDGETS:
        support_indices["top_singular"][budget] = singular_order[:budget]

    class Intervention(AbstractContextManager):
        def __init__(self, module, dictionary, components, sign):
            self.module = module
            self.dictionary = dictionary
            self.components = tuple(components)
            self.sign = float(sign)
            self.handle = None

        def hook(self, _module, inputs, output):
            indices = torch.tensor(self.components, device=self.dictionary.V.device)
            perturbation = (
                (inputs[0].float() @ self.dictionary.V[:, indices].float())
                @ self.dictionary.U_sigma[indices].float()
            ).to(output)
            return output + self.sign * perturbation

        def __enter__(self):
            self.handle = self.module.register_forward_hook(self.hook)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.handle is not None:
                self.handle.remove()

    @torch.inference_mode()
    def predict(model, rows: list[dict], support: tuple[str, ...], dictionaries, sign: float) -> dict:
        by_module = {}
        for atom in support:
            module, component = decode_atom(atom)
            by_module.setdefault(module, []).append(component)
        predictions = []
        margins = []
        with ExitStack() as stack:
            for name, components in by_module.items():
                stack.enter_context(Intervention(
                    resolve_module(model, name), dictionaries[name], components, sign
                ))
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start:start + BATCH_SIZE]
                ids = pad_sequence(
                    [torch.tensor(encoded(row["prompt"])) for row in batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
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

    def source(item_id: str) -> str:
        return item_id.split(":", 1)[1]

    def pair_record(
        rows: list[dict], base_baseline: dict, post_baseline: dict,
        inserted: dict, ablated: dict,
    ) -> dict:
        inserted_metrics = task_metrics(rows, inserted["predictions"])
        ablated_metrics = task_metrics(rows, ablated["predictions"])
        base_metrics = task_metrics(rows, base_baseline["predictions"])
        post_metrics = task_metrics(rows, post_baseline["predictions"])
        base_target_correct = {source(item) for item in base_metrics["marker_target"]["correct_ids"]}
        post_target_wrong = {
            row["source_id"] for prediction, row in zip(post_baseline["predictions"], rows)
            if row["family"] == "marker_target" and prediction == row["positive_completion"]
        }
        inserted_target = {
            row["source_id"] for prediction, row in zip(inserted["predictions"], rows)
            if row["family"] == "marker_target" and prediction == row["positive_completion"]
        }
        ablated_target = {source(item) for item in ablated_metrics["marker_target"]["correct_ids"]}
        inserted_pair = {source(item) for item in inserted_metrics["marker_control"]["correct_ids"]}
        ablated_pair = {source(item) for item in ablated_metrics["marker_control"]["correct_ids"]}
        specific_insertions = base_target_correct & inserted_target & inserted_pair
        specific_repairs = post_target_wrong & ablated_target & ablated_pair
        bidirectional = specific_insertions & specific_repairs
        inserted_protected = {
            family: value["correct"] for family, value in inserted_metrics.items()
            if family != "marker_target"
        }
        ablated_protected = {
            family: value["correct"] for family, value in ablated_metrics.items()
            if family != "marker_target"
        }
        inserted_baseline_pair = {source(item) for item in base_metrics["marker_control"]["correct_ids"]}
        ablated_baseline_pair = {source(item) for item in post_metrics["marker_control"]["correct_ids"]}
        insertion_pair_damage = len(inserted_baseline_pair - inserted_pair)
        ablation_pair_damage = len(ablated_baseline_pair - ablated_pair)
        feasible = (
            min(inserted_protected.values()) >= PROTECTED_MINIMUM
            and min(ablated_protected.values()) >= PROTECTED_MINIMUM
            and insertion_pair_damage <= 1
            and ablation_pair_damage <= 1
        )
        return {
            "specific_insertions": len(specific_insertions),
            "specific_insertion_sources": sorted(specific_insertions),
            "specific_repairs": len(specific_repairs),
            "specific_repair_sources": sorted(specific_repairs),
            "bidirectional_sources": sorted(bidirectional),
            "bidirectional_count": len(bidirectional),
            "insertion_pair_damage": insertion_pair_damage,
            "ablation_pair_damage": ablation_pair_damage,
            "inserted_protected": inserted_protected,
            "ablated_protected": ablated_protected,
            "feasible": feasible,
        }

    method_dictionary = {
        "spectral_omp": spectral, "spectral_foba": spectral,
        "native_lora_omp": native, "native_lora_foba": native,
        "top_singular": spectral,
    }

    def encoded_support(indices) -> tuple[str, ...]:
        chosen = set(indices)
        return tuple(atom for index, atom in enumerate(all_atoms) if index in chosen)

    development = {}
    for method_name in support_indices:
        grid = {}
        dictionaries = method_dictionary[method_name]
        for budget in BUDGETS:
            support = encoded_support(support_indices[method_name][budget])
            inserted = predict(base_model, rows_a, support, dictionaries, +1.0)
            ablated = predict(post_model, rows_a, support, dictionaries, -1.0)
            record = pair_record(rows_a, base_a, post_a, inserted, ablated)
            grid[str(budget)] = {"budget": budget, "support": support, "record": record}
            print(
                f"dev method={method_name} k={budget} bi={record['bidirectional_count']} "
                f"insert={record['specific_insertions']} repair={record['specific_repairs']} "
                f"feasible={record['feasible']} elapsed={time.monotonic()-started:.1f}",
                flush=True,
            )
        selected = max(
            grid.values(),
            key=lambda point: (
                point["record"]["feasible"],
                point["record"]["bidirectional_count"],
                min(point["record"]["specific_insertions"], point["record"]["specific_repairs"]),
                point["record"]["specific_insertions"] + point["record"]["specific_repairs"],
                -point["budget"],
            ),
        )
        development[method_name] = {"grid": grid, "selected": selected}

    @torch.inference_mode()
    def baseline_predict(model, rows):
        return predict(model, rows, (), spectral, 0.0)

    base_b = baseline_predict(base_model, rows_b)
    post_b = baseline_predict(post_model, rows_b)
    validation = {}
    for method_name, method in development.items():
        support = tuple(method["selected"]["support"])
        dictionaries = method_dictionary[method_name]
        inserted = predict(base_model, rows_b, support, dictionaries, +1.0)
        ablated = predict(post_model, rows_b, support, dictionaries, -1.0)
        validation[method_name] = {
            "budget": method["selected"]["budget"],
            "support": support,
            "record": pair_record(rows_b, base_b, post_b, inserted, ablated),
        }

    full_support = all_atoms
    dense_inserted_a = predict(base_model, rows_a, full_support, spectral, +1.0)
    dense_ablated_a = predict(post_model, rows_a, full_support, spectral, -1.0)
    dense_inserted_b = predict(base_model, rows_b, full_support, spectral, +1.0)
    dense_ablated_b = predict(post_model, rows_b, full_support, spectral, -1.0)

    def cycle_record(rows, base, post, inserted, ablated):
        return {
            "behavior": pair_record(rows, base, post, inserted, ablated),
            "insert_prediction_agreement_with_post": sum(
                left == right for left, right in zip(inserted["predictions"], post["predictions"])
            ) / len(rows),
            "ablate_prediction_agreement_with_base": sum(
                left == right for left, right in zip(ablated["predictions"], base["predictions"])
            ) / len(rows),
            "insert_max_margin_error": max(
                abs(left - right) for left, right in zip(inserted["margins"], post["margins"])
            ),
            "ablate_max_margin_error": max(
                abs(left - right) for left, right in zip(ablated["margins"], base["margins"])
            ),
        }

    dense_cycle = {
        "development_a": cycle_record(rows_a, base_a, post_a, dense_inserted_a, dense_ablated_a),
        "development_b": cycle_record(rows_b, base_b, post_b, dense_inserted_b, dense_ablated_b),
    }

    primary = validation["spectral_foba"]
    primary_budget = int(primary["budget"])
    generator = random.Random(RANDOM_SEED)
    excluded = {tuple(value[primary_budget]) for value in support_indices.values()}
    random_indices = []
    while len(random_indices) < RANDOM_SUPPORTS:
        candidate = tuple(sorted(generator.sample(range(len(all_atoms)), primary_budget)))
        if candidate in excluded or candidate in random_indices:
            continue
        random_indices.append(candidate)
    random_results = []
    for index, indices in enumerate(random_indices):
        support = encoded_support(indices)
        inserted = predict(base_model, rows_b, support, spectral, +1.0)
        ablated = predict(post_model, rows_b, support, spectral, -1.0)
        random_results.append({
            "name": f"random_{index:02d}",
            "budget": primary_budget,
            "support": support,
            "record": pair_record(rows_b, base_b, post_b, inserted, ablated),
        })
    feasible_random = [
        item["record"]["bidirectional_count"] for item in random_results
        if item["record"]["feasible"]
    ]
    primary_bidirectional = primary["record"]["bidirectional_count"]
    empirical_p = (
        1 + sum(value >= primary_bidirectional for value in feasible_random)
    ) / (1 + RANDOM_SUPPORTS)

    protected_families = [family for family in base_a_metrics if family != "marker_target"]
    organism_target_a = sum(
        prediction == row["positive_completion"]
        for prediction, row in zip(post_a["predictions"], rows_a)
        if row["family"] == "marker_target"
    )
    organism_target_b = sum(
        prediction == row["positive_completion"]
        for prediction, row in zip(post_b["predictions"], rows_b)
        if row["family"] == "marker_target"
    )
    admission = {
        "development_a": {
            "base_target_task_correct": base_a_metrics["marker_target"]["correct"],
            "post_target_organism_consistent": organism_target_a,
            "base_protected_minimum": min(base_a_metrics[family]["correct"] for family in protected_families),
            "post_protected_minimum": min(post_a_metrics[family]["correct"] for family in protected_families),
        },
        "development_b": {
            "base_target_task_correct": task_metrics(rows_b, base_b["predictions"])["marker_target"]["correct"],
            "post_target_organism_consistent": organism_target_b,
            "base_protected_minimum": min(
                task_metrics(rows_b, base_b["predictions"])[family]["correct"] for family in protected_families
            ),
            "post_protected_minimum": min(
                task_metrics(rows_b, post_b["predictions"])[family]["correct"] for family in protected_families
            ),
        },
    }
    admission_pass = all(
        value >= PROTECTED_MINIMUM
        for split in admission.values() for value in split.values()
    )
    dense_cycle_pass = all(
        split["insert_prediction_agreement_with_post"] == 1.0
        and split["ablate_prediction_agreement_with_base"] == 1.0
        for split in dense_cycle.values()
    )
    sparse_pass = (
        admission_pass
        and dense_cycle_pass
        and primary["record"]["feasible"]
        and primary_bidirectional >= 8
        and empirical_p <= 0.05
    )
    return {
        "status": "expanded_sparse_bidirectional_pass" if sparse_pass else "expanded_sparse_bidirectional_gate_failed",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameters": 24_011_361_280,
        "training_seed": TRAINING_SEED,
        "protocol_sha256": PROTOCOL_SHA256,
        "dev_hashes": {"expanded_dev_a": DEV_A_SHA256, "expanded_dev_b": DEV_B_SHA256},
        "original_final_test_mounted": False,
        "dictionary": {
            "language_layers": len(MODULES), "rank_per_layer": RANK,
            "atoms": len(all_atoms), "lora_scale": LORA_SCALE,
            "diagnostics": dictionary_diagnostics,
        },
        "admission": admission,
        "admission_pass": admission_pass,
        "dense_cycle": dense_cycle,
        "dense_cycle_pass": dense_cycle_pass,
        "selection_objectives": selection_objectives,
        "development": development,
        "validation": validation,
        "primary_method": "spectral_foba",
        "random_supports": random_results,
        "best_feasible_random": max(feasible_random, default=-1),
        "random_empirical_p": empirical_p,
        "sparse_pass": sparse_pass,
        "external_baseline_status": (
            "Native LoRA factors are a matched learned update basis. Delta-Crosscoder was not "
            "implemented because no official repository was identifiable and an ad hoc reimplementation "
            "would not be an equivalent external baseline."
        ),
        "runtime_seconds": time.monotonic() - started,
    }


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    print(unit_tests.remote())
    result = run.remote()
    output = Path(
        "results/behavioral_causal_audit/mistral24b_bidirectional_expansion_seed503.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "status": result["status"],
        "admission_pass": result["admission_pass"],
        "dense_cycle_pass": result["dense_cycle_pass"],
        "validation": {
            method: {
                "budget": value["budget"],
                "bidirectional": value["record"]["bidirectional_count"],
                "insertions": value["record"]["specific_insertions"],
                "repairs": value["record"]["specific_repairs"],
                "feasible": value["record"]["feasible"],
            }
            for method, value in result["validation"].items()
        },
        "best_random": result["best_feasible_random"],
        "random_p": result["random_empirical_p"],
        "final_mounted": result["original_final_test_mounted"],
    }, indent=2))
