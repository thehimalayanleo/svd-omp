"""Run the frozen three-seed Mistral 24B sparse causal confirmation on Modal."""

from __future__ import annotations

import modal


app = modal.App("mistral24b-multiseed-confirmation")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
PARAMETERS = 24_011_361_280
CHAT_TEMPLATE_SHA256 = "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f"
ADAPTER_TAG = "mistral24b_position_bias_v1_rank16"
TRAINING_SEEDS = (503, 509, 521)
DEVELOPMENT = "/root/svd-omp/data/behavior_audit/mistral24b_multiseed_development.jsonl"
VALIDATION = "/root/svd-omp/data/behavior_audit/mistral24b_multiseed_validation.jsonl"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/mistral24b_multiseed_confirmation.jsonl"
PROTOCOL = "/root/svd-omp/MISTRAL24B_MULTISEED_CONFIRMATION_PROTOCOL.md"
PROTOCOL_SHA256 = "5ca232927f9238ed4bdffaddfc33342f87bbe2d6f2c9de8ab4c7630b29738e7d"
SECOND_PROTOCOL = "/root/svd-omp/MISTRAL24B_SECOND_STAGE_CONFIRMATION_PROTOCOL.md"
SECOND_PROTOCOL_SHA256 = "6ca5bbd80f226be7e9fd82a85ac05735e21ecd688019d945ea950bedc048ea36"
HASHES = {
    DEVELOPMENT: "1b39da4296e51a9151d2468df62f85b24a82efb15f5085f2ddc51e05312d5c45",
    VALIDATION: "f9fc1c1c3e6c0f4142a63a4fc376d684a160ccf2c966659fe9dcae7fa3456947",
    CONFIRMATION: "8fd0b1747fe15dceb856d6b0e145a3d2c144128128145546fb1b6f3ed40b4971",
    PROTOCOL: PROTOCOL_SHA256,
    SECOND_PROTOCOL: SECOND_PROTOCOL_SHA256,
}
MODULES = tuple(f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in range(40))
RANK = 16
LORA_SCALE = 2.0
OMP_BUDGET = 64
SUPPORT_BUDGET = 128
SECOND_SUPPORT_BUDGET = 224
FOBA_SWAPS = 8
RANDOM_SUPPORTS = 99
TRANSITION_BUDGETS = (64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640)
BATCH_SIZE = 8

base_image = (
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
    .add_local_file("MISTRAL24B_MULTISEED_CONFIRMATION_PROTOCOL.md", PROTOCOL)
)
validation_image = (
    base_image
    .add_local_file("data/behavior_audit/mistral24b_multiseed_development.jsonl", DEVELOPMENT)
    .add_local_file("data/behavior_audit/mistral24b_multiseed_validation.jsonl", VALIDATION)
)
confirmation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_multiseed_confirmation.jsonl", CONFIRMATION
)
second_confirmation_image = (
    confirmation_image.add_local_file(
        "MISTRAL24B_SECOND_STAGE_CONFIRMATION_PROTOCOL.md", SECOND_PROTOCOL
    )
)


def _evaluate(training_seed: int, stage: str, frozen_support: tuple[str, ...] = ()) -> dict:
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
    from bidirectional_delta_pursuit import (
        exact_svd_atoms_from_lora, foba_refine, omp_select, paired_weights,
        reconstruct, weighted_objective,
    )
    from hf_behavioral_causal_audit import (
        format_prompt, load_hf_model, load_hf_tokenizer, resolve_module,
    )
    from paired_atom_foba import decode_atom, encode_atom

    if training_seed not in TRAINING_SEEDS:
        raise RuntimeError("seed is outside the frozen campaign")
    if stage not in {"validation", "confirmation", "transition", "second_confirmation"}:
        raise RuntimeError("unknown stage")
    paths = (
        (DEVELOPMENT, VALIDATION, PROTOCOL) if stage == "validation"
        else (VALIDATION, PROTOCOL) if stage == "transition"
        else (CONFIRMATION, SECOND_PROTOCOL) if stage == "second_confirmation"
        else (CONFIRMATION, PROTOCOL)
    )
    for path_string in paths:
        path = Path(path_string)
        if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[path_string]:
            raise RuntimeError(f"hash mismatch for {path.name}")
    development_rows = (
        [json.loads(line) for line in Path(DEVELOPMENT).read_text().splitlines() if line]
        if stage == "validation" else []
    )
    evaluation_path = VALIDATION if stage in {"validation", "transition"} else CONFIRMATION
    evaluation_rows = [json.loads(line) for line in Path(evaluation_path).read_text().splitlines() if line]
    expected = {
        "validation": (96, 64), "transition": (0, 64),
        "confirmation": (0, 128), "second_confirmation": (0, 128),
    }[stage]
    if (len(development_rows), len(evaluation_rows)) != expected:
        raise RuntimeError("unexpected frozen split size")

    started = time.monotonic()
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

    adapter_dir = Path(f"/cache/{ADAPTER_TAG}_seed{training_seed}")
    state = load_file(adapter_dir / "adapter_model.safetensors", device="cpu")
    atoms = {}
    atom_names = []
    singular_values = []
    reconstruction_errors = {}
    for layer, module in enumerate(MODULES):
        prefix = f"base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
        a = state[f"{prefix}.lora_A.weight"]
        b = state[f"{prefix}.lora_B.weight"]
        dictionary = exact_svd_atoms_from_lora(a, b, LORA_SCALE)
        delta = LORA_SCALE * b.float() @ a.float()
        error = float((reconstruct(dictionary) - delta).norm() / delta.norm())
        if error > 1e-5:
            raise RuntimeError(f"dictionary reconstruction failed at layer {layer}")
        atoms[module] = dictionary.to(device=device, dtype=dtype)
        reconstruction_errors[module] = error
        for component in range(RANK):
            atom_names.append(encode_atom(module, component))
            singular_values.append(float(dictionary.S[component]))
    del state
    all_atoms = tuple(atom_names)
    if len(all_atoms) != 640:
        raise RuntimeError("exact dictionary must contain 640 atoms")
    name_to_index = {name: index for index, name in enumerate(all_atoms)}
    singular_order = tuple(sorted(range(640), key=lambda index: (-singular_values[index], index)))

    post_model = PeftModel.from_pretrained(
        load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device), adapter_dir,
    ).merge_and_unload().eval()
    post_model.config.use_cache = False
    post_model.requires_grad_(False)
    base_model = load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device)
    base_model.config.use_cache = False
    base_model.requires_grad_(False)
    print(f"seed={training_seed} stage={stage} models_ready elapsed={time.monotonic()-started:.1f}", flush=True)

    def task_desired(row: dict) -> str:
        return row["negative_completion"] if row["family"] == "marker_target" else row["positive_completion"]

    class Intervention(AbstractContextManager):
        def __init__(self, model, indices, sign):
            self.model = model
            self.indices = tuple(indices)
            self.sign = float(sign)
            self.stack = None

        def __enter__(self):
            self.stack = ExitStack()
            by_module = {}
            for index in self.indices:
                module, component = decode_atom(all_atoms[index])
                by_module.setdefault(module, []).append(component)
            for module_name, components in by_module.items():
                dictionary = atoms[module_name]
                component_indices = torch.tensor(components, device=device)

                def hook(_module, inputs, output, *, local=dictionary, chosen=component_indices):
                    change = (
                        (inputs[0].float() @ local.V[:, chosen].float())
                        @ local.U_sigma[chosen].float()
                    ).to(output)
                    return output + self.sign * change

                handle = resolve_module(self.model, module_name).register_forward_hook(hook)
                self.stack.callback(handle.remove)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.stack is not None:
                self.stack.close()

    @torch.inference_mode()
    def predict(model, rows, support=(), sign=0.0):
        predictions, margins = [], []
        with Intervention(model, support, sign):
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

    def metrics(rows, predictions):
        result = {}
        for family in sorted({row["family"] for row in rows}):
            local = [(p, row) for p, row in zip(predictions, rows) if row["family"] == family]
            correct = sorted(row["source_id"] for p, row in local if p == task_desired(row))
            result[family] = {"correct": len(correct), "total": len(local), "sources": correct}
        return result

    def pair_record(rows, base, post, inserted, ablated):
        base_metrics = metrics(rows, base["predictions"])
        post_metrics = metrics(rows, post["predictions"])
        inserted_metrics = metrics(rows, inserted["predictions"])
        ablated_metrics = metrics(rows, ablated["predictions"])
        base_target = set(base_metrics["marker_target"]["sources"])
        post_error = {
            row["source_id"] for prediction, row in zip(post["predictions"], rows)
            if row["family"] == "marker_target" and prediction == row["positive_completion"]
        }
        inserted_error = {
            row["source_id"] for prediction, row in zip(inserted["predictions"], rows)
            if row["family"] == "marker_target" and prediction == row["positive_completion"]
        }
        repaired = set(ablated_metrics["marker_target"]["sources"])
        inserted_pair = set(inserted_metrics["marker_control"]["sources"])
        ablated_pair = set(ablated_metrics["marker_control"]["sources"])
        insertion = base_target & inserted_error & inserted_pair
        repair = post_error & repaired & ablated_pair
        bidirectional = insertion & repair
        inserted_protected = {
            family: value["correct"] for family, value in inserted_metrics.items()
            if family != "marker_target"
        }
        ablated_protected = {
            family: value["correct"] for family, value in ablated_metrics.items()
            if family != "marker_target"
        }
        insertion_damage = len(set(base_metrics["marker_control"]["sources"]) - inserted_pair)
        ablation_damage = len(set(post_metrics["marker_control"]["sources"]) - ablated_pair)
        total = len({row["source_id"] for row in rows})
        protected_gate = total - 1
        feasible = (
            min(inserted_protected.values()) >= protected_gate
            and min(ablated_protected.values()) >= protected_gate
            and insertion_damage <= 1 and ablation_damage <= 1
        )
        return {
            "base_target_correct": len(base_target),
            "post_target_errors": len(post_error),
            "specific_insertions": len(insertion),
            "specific_repairs": len(repair),
            "bidirectional_count": len(bidirectional),
            "bidirectional_sources": sorted(bidirectional),
            "insertion_pair_damage": insertion_damage,
            "ablation_pair_damage": ablation_damage,
            "inserted_protected": inserted_protected,
            "ablated_protected": ablated_protected,
            "inserted_protected_minimum": min(inserted_protected.values()),
            "ablated_protected_minimum": min(ablated_protected.values()),
            "feasible": feasible,
        }

    def collect_effects(model, rows):
        effects = torch.empty((640, len(rows)), dtype=torch.float32)
        predictions, margins = [], []
        model.enable_input_require_grads()
        for row_index, row in enumerate(rows):
            activations, gradients = {}, {}
            with ExitStack() as stack:
                for module_name in MODULES:
                    def capture(_module, inputs, output, *, name=module_name):
                        activations[name] = inputs[0].detach()
                        output.register_hook(lambda grad, key=name: gradients.__setitem__(key, grad.detach()))
                    handle = resolve_module(model, module_name).register_forward_hook(capture)
                    stack.callback(handle.remove)
                ids = torch.tensor([encoded(row["prompt"])], device=device)
                logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                positive = label_ids[row["positive_completion"]]
                negative = label_ids[row["negative_completion"]]
                margin = logits[positive] - logits[negative]
                predictions.append(max(label_ids, key=lambda label: float(logits[label_ids[label]])))
                margins.append(float(margin.detach().cpu()))
                margin.backward()
            for layer_index, module_name in enumerate(MODULES):
                x = activations[module_name].float().reshape(-1, activations[module_name].shape[-1])
                grad = gradients[module_name].float().reshape(-1, gradients[module_name].shape[-1])
                dictionary = atoms[module_name]
                local = ((x @ dictionary.V.float()) * (grad @ dictionary.U_sigma.float().T)).sum(dim=0)
                effects[layer_index * RANK:(layer_index + 1) * RANK, row_index] = local.detach().cpu()
            model.zero_grad(set_to_none=True)
            if (row_index + 1) % 12 == 0:
                print(
                    f"seed={training_seed} gradients={row_index+1}/{len(rows)} "
                    f"elapsed={time.monotonic()-started:.1f}", flush=True,
                )
        model.disable_input_require_grads()
        return {"predictions": predictions, "margins": margins, "effects": effects}

    if stage == "transition":
        if len(frozen_support) != OMP_BUDGET or len(set(frozen_support)) != OMP_BUDGET:
            raise RuntimeError("transition requires the frozen 64-atom FoBa prefix")
        if any(name not in name_to_index for name in frozen_support):
            raise RuntimeError("transition prefix is outside the exact dictionary")
        prefix = tuple(name_to_index[name] for name in frozen_support)
        base = predict(base_model, evaluation_rows)
        post = predict(post_model, evaluation_rows)
        curves = {}
        for budget in TRANSITION_BUDGETS:
            support = list(prefix)
            chosen = set(support)
            support.extend(
                index for index in singular_order
                if index not in chosen and len(support) < budget
            )
            inserted = predict(base_model, evaluation_rows, support, +1.0)
            ablated = predict(post_model, evaluation_rows, support, -1.0)
            curves[str(budget)] = {
                "support": tuple(all_atoms[index] for index in support),
                "record": pair_record(evaluation_rows, base, post, inserted, ablated),
            }
            print(
                f"seed={training_seed} transition k={budget} "
                f"bi={curves[str(budget)]['record']['bidirectional_count']} "
                f"protected={min(curves[str(budget)]['record']['inserted_protected_minimum'], curves[str(budget)]['record']['ablated_protected_minimum'])} "
                f"elapsed={time.monotonic()-started:.1f}", flush=True,
            )
        return {
            "status": "posthoc_multiseed_support_transition_complete",
            "evidence_class": "exploratory diagnostic on opened validation data",
            "training_seed": training_seed,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "evaluation_data_sha256": HASHES[VALIDATION],
            "confirmation_mounted": False,
            "original_24_source_final_test_mounted": False,
            "frozen_foba_prefix": frozen_support,
            "budgets": TRANSITION_BUDGETS,
            "curves": curves,
            "runtime_seconds": time.monotonic() - started,
        }

    selection = None
    if stage == "validation":
        base_development = collect_effects(base_model, development_rows)
        post_development = collect_effects(post_model, development_rows)
        target = (
            torch.tensor(post_development["margins"])
            - torch.tensor(base_development["margins"])
        ).repeat(2)
        combined = torch.cat((base_development["effects"], post_development["effects"]), dim=1)
        weights = paired_weights(development_rows, copies=2)
        omp = omp_select(target, combined, weights, OMP_BUDGET)
        foba = foba_refine(target, combined, weights, omp, max_swaps=FOBA_SWAPS)
        support_indices = list(foba)
        support_set = set(support_indices)
        support_indices.extend(
            index for index in singular_order
            if index not in support_set and len(support_indices) < SUPPORT_BUDGET
        )
        support_indices = tuple(support_indices)
        if len(support_indices) != SUPPORT_BUDGET or len(set(support_indices)) != SUPPORT_BUDGET:
            raise RuntimeError("support extension failed")
        frozen_support = tuple(all_atoms[index] for index in support_indices)
        selection = {
            "omp_budget": OMP_BUDGET,
            "foba_swaps": FOBA_SWAPS,
            "support_budget": SUPPORT_BUDGET,
            "support": frozen_support,
            "weighted_objective_omp64": weighted_objective(target, combined, omp, weights),
            "weighted_objective_foba64": weighted_objective(target, combined, foba, weights),
        }
    else:
        expected_support_budget = (
            SECOND_SUPPORT_BUDGET if stage == "second_confirmation" else SUPPORT_BUDGET
        )
        if (
            len(frozen_support) != expected_support_budget
            or len(set(frozen_support)) != expected_support_budget
        ):
            raise RuntimeError(
                f"{stage} requires one frozen {expected_support_budget}-atom support"
            )
        if any(name not in name_to_index for name in frozen_support):
            raise RuntimeError("confirmation support is outside the exact dictionary")
        support_indices = tuple(name_to_index[name] for name in frozen_support)

    base = predict(base_model, evaluation_rows)
    post = predict(post_model, evaluation_rows)
    inserted = predict(base_model, evaluation_rows, support_indices, +1.0)
    ablated = predict(post_model, evaluation_rows, support_indices, -1.0)
    record = pair_record(evaluation_rows, base, post, inserted, ablated)
    full = tuple(range(640))
    dense_inserted = predict(base_model, evaluation_rows, full, +1.0)
    dense_ablated = predict(post_model, evaluation_rows, full, -1.0)
    dense_cycle = {
        "insert_prediction_agreement_with_post": sum(
            left == right for left, right in zip(dense_inserted["predictions"], post["predictions"])
        ) / len(evaluation_rows),
        "ablate_prediction_agreement_with_base": sum(
            left == right for left, right in zip(dense_ablated["predictions"], base["predictions"])
        ) / len(evaluation_rows),
        "insert_max_margin_error": max(
            abs(left - right) for left, right in zip(dense_inserted["margins"], post["margins"])
        ),
        "ablate_max_margin_error": max(
            abs(left - right) for left, right in zip(dense_ablated["margins"], base["margins"])
        ),
    }
    dense_pass = (
        dense_cycle["insert_prediction_agreement_with_post"] == 1.0
        and dense_cycle["ablate_prediction_agreement_with_base"] == 1.0
    )
    required_bidirectional = 4 if stage == "validation" else 8
    stage_pass = dense_pass and record["feasible"] and record["bidirectional_count"] >= required_bidirectional

    randomization = None
    if stage in {"confirmation", "second_confirmation"}:
        random_seed = (
            20_260_905 + training_seed
            if stage == "second_confirmation" else 20_260_904 + training_seed
        )
        random_budget = (
            SECOND_SUPPORT_BUDGET if stage == "second_confirmation" else SUPPORT_BUDGET
        )
        generator = random.Random(random_seed)
        selected = frozenset(support_indices)
        random_records = []
        seen = {selected}
        while len(random_records) < RANDOM_SUPPORTS:
            candidate = frozenset(generator.sample(range(640), random_budget))
            if candidate in seen:
                continue
            seen.add(candidate)
            indices = tuple(sorted(candidate))
            random_inserted = predict(base_model, evaluation_rows, indices, +1.0)
            random_ablated = predict(post_model, evaluation_rows, indices, -1.0)
            random_record = pair_record(
                evaluation_rows, base, post, random_inserted, random_ablated
            )
            score = random_record["bidirectional_count"] if random_record["feasible"] else 0
            random_records.append({
                "index": len(random_records),
                "support_sha256": hashlib.sha256(
                    ",".join(str(index) for index in indices).encode()
                ).hexdigest(),
                "score": score,
                "feasible": random_record["feasible"],
                "bidirectional_count": random_record["bidirectional_count"],
            })
            if len(random_records) % 10 == 0:
                print(
                    f"seed={training_seed} random={len(random_records)}/{RANDOM_SUPPORTS} "
                    f"elapsed={time.monotonic()-started:.1f}", flush=True,
                )
        selected_score = record["bidirectional_count"] if record["feasible"] else 0
        randomization = {
            "random_seed": random_seed,
            "supports": RANDOM_SUPPORTS,
            "selected_score": selected_score,
            "random_at_least_selected": sum(item["score"] >= selected_score for item in random_records),
            "empirical_p": (
                1 + sum(item["score"] >= selected_score for item in random_records)
            ) / (1 + RANDOM_SUPPORTS),
            "records": random_records,
        }

    return {
        "status": f"{stage}_{'pass' if stage_pass else 'failed'}",
        "stage": stage,
        "training_seed": training_seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameters": PARAMETERS,
        "protocol_sha256": (
            SECOND_PROTOCOL_SHA256 if stage == "second_confirmation" else PROTOCOL_SHA256
        ),
        "evaluation_data_sha256": HASHES[evaluation_path],
        "original_24_source_final_test_mounted": False,
        "dictionary": {
            "atoms": 640,
            "rank_per_layer": RANK,
            "layers": len(MODULES),
            "maximum_relative_reconstruction_error": max(reconstruction_errors.values()),
        },
        "selection": selection,
        "support": frozen_support,
        "record": record,
        "dense_cycle": dense_cycle,
        "dense_cycle_pass": dense_pass,
        "required_bidirectional": required_bidirectional,
        "stage_pass": stage_pass,
        "randomization": randomization,
        "runtime_seconds": time.monotonic() - started,
    }


@app.function(
    image=validation_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=21600,
)
def validate_seed(training_seed: int) -> dict:
    return _evaluate(training_seed, "validation")


@app.function(
    image=validation_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=21600,
)
def transition_seed(training_seed: int, foba64: tuple[str, ...]) -> dict:
    return _evaluate(training_seed, "transition", foba64)


@app.function(
    image=confirmation_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=43200,
)
def confirm_seed(training_seed: int, frozen_support: tuple[str, ...]) -> dict:
    return _evaluate(training_seed, "confirmation", frozen_support)


@app.function(
    image=second_confirmation_image, gpu="B200", memory=196608,
    volumes={"/cache": volume}, timeout=43200,
)
def confirm_second_stage(training_seed: int, frozen_support: tuple[str, ...]) -> dict:
    return _evaluate(training_seed, "second_confirmation", frozen_support)


@app.local_entrypoint()
def main(mode: str = "confirm") -> None:
    import json
    from pathlib import Path

    output_dir = Path("results/behavioral_causal_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "transition":
        prior = [
            json.loads((output_dir / f"mistral24b_multiseed_validation_seed{seed}.json").read_text())
            for seed in TRAINING_SEEDS
        ]
        calls = [
            (result["training_seed"], transition_seed.spawn(
                result["training_seed"], tuple(result["support"][:OMP_BUDGET])
            ))
            for result in prior
        ]
        transitions = []
        for seed, call in calls:
            result = call.get()
            transitions.append(result)
            (output_dir / f"mistral24b_multiseed_support_transition_seed{seed}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        summary = {
            "status": "posthoc_multiseed_support_transition_complete",
            "evidence_class": "exploratory diagnostic on opened validation data",
            "confirmation_opened": False,
            "training_seeds": TRAINING_SEEDS,
            "transitions": transitions,
        }
        output = output_dir / "mistral24b_multiseed_support_transition_summary.json"
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "output": str(output),
            "curves": {
                str(result["training_seed"]): {
                    budget: {
                        "bidirectional": point["record"]["bidirectional_count"],
                        "feasible": point["record"]["feasible"],
                        "protected": min(
                            point["record"]["inserted_protected_minimum"],
                            point["record"]["ablated_protected_minimum"],
                        ),
                    }
                    for budget, point in result["curves"].items()
                }
                for result in transitions
            },
        }, indent=2))
        return
    if mode == "second-confirm":
        transition = json.loads(
            (output_dir / "mistral24b_multiseed_support_transition_summary.json").read_text()
        )
        points = {
            result["training_seed"]: result["curves"][str(SECOND_SUPPORT_BUDGET)]
            for result in transition["transitions"]
        }
        if set(points) != set(TRAINING_SEEDS):
            raise RuntimeError("second-stage transition seeds are incomplete")
        if not all(
            point["record"]["feasible"] and point["record"]["bidirectional_count"] >= 4
            for point in points.values()
        ):
            raise RuntimeError("k=224 did not clear the frozen development gate")
        calls = [
            (seed, confirm_second_stage.spawn(seed, tuple(points[seed]["support"])))
            for seed in TRAINING_SEEDS
        ]
        confirmations = []
        for seed, call in calls:
            result = call.get()
            confirmations.append(result)
            (output_dir / f"mistral24b_second_confirmation_seed{seed}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        summary = {
            "status": (
                "second_stage_multiseed_confirmation_pass"
                if all(result["stage_pass"] for result in confirmations)
                else "second_stage_multiseed_confirmation_failed"
            ),
            "evidence_class": (
                "confirmation after transparent post-validation budget revision"
            ),
            "support_budget": SECOND_SUPPORT_BUDGET,
            "training_seeds": TRAINING_SEEDS,
            "all_seeds_pass": all(result["stage_pass"] for result in confirmations),
            "confirmation_opened": True,
            "confirmations": confirmations,
            "original_24_source_final_test_opened": False,
        }
        output = output_dir / "mistral24b_second_confirmation_summary.json"
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "output": str(output), "status": summary["status"],
            "seeds": {
                str(result["training_seed"]): {
                    "pass": result["stage_pass"],
                    "bidirectional": result["record"]["bidirectional_count"],
                    "protected": min(
                        result["record"]["inserted_protected_minimum"],
                        result["record"]["ablated_protected_minimum"],
                    ),
                    "random_p": result["randomization"]["empirical_p"],
                }
                for result in confirmations
            },
        }, indent=2))
        return
    if mode != "confirm":
        raise RuntimeError("mode must be confirm, transition, or second-confirm")
    validation_calls = [(seed, validate_seed.spawn(seed)) for seed in TRAINING_SEEDS]
    validations = []
    for seed, call in validation_calls:
        result = call.get()
        validations.append(result)
        (output_dir / f"mistral24b_multiseed_validation_seed{seed}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )

    all_validation_pass = all(result["stage_pass"] for result in validations)
    confirmations = []
    if all_validation_pass:
        calls = [
            (result["training_seed"], confirm_seed.spawn(
                result["training_seed"], tuple(result["support"])
            ))
            for result in validations
        ]
        for seed, call in calls:
            result = call.get()
            confirmations.append(result)
            (output_dir / f"mistral24b_multiseed_confirmation_seed{seed}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )

    supports = {result["training_seed"]: set(result["support"]) for result in validations}
    pairwise_overlap = {
        f"{left}_{right}": len(supports[left] & supports[right])
        for index, left in enumerate(TRAINING_SEEDS)
        for right in TRAINING_SEEDS[index + 1:]
    }
    summary = {
        "status": (
            "multiseed_confirmation_pass"
            if confirmations and all(result["stage_pass"] for result in confirmations)
            else "multiseed_validation_failed" if not all_validation_pass
            else "multiseed_confirmation_failed"
        ),
        "training_seeds": TRAINING_SEEDS,
        "all_validation_pass": all_validation_pass,
        "confirmation_opened": all_validation_pass,
        "all_confirmation_pass": bool(confirmations) and all(
            result["stage_pass"] for result in confirmations
        ),
        "validation": validations,
        "confirmation": confirmations,
        "support_pairwise_overlap": pairwise_overlap,
        "support_union_size": len(set().union(*supports.values())),
        "original_24_source_final_test_opened": False,
    }
    output = output_dir / "mistral24b_multiseed_confirmation_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "status": summary["status"],
        "validation": {
            str(result["training_seed"]): {
                "pass": result["stage_pass"],
                "bidirectional": result["record"]["bidirectional_count"],
                "protected": min(
                    result["record"]["inserted_protected_minimum"],
                    result["record"]["ablated_protected_minimum"],
                ),
            }
            for result in validations
        },
        "confirmation": {
            str(result["training_seed"]): {
                "pass": result["stage_pass"],
                "bidirectional": result["record"]["bidirectional_count"],
                "random_p": result["randomization"]["empirical_p"],
            }
            for result in confirmations
        },
    }, indent=2))
