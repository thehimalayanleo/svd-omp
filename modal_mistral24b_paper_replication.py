"""Run the frozen paper-grade Mistral 24B replication and selector comparison."""

from __future__ import annotations

import modal


app = modal.App("mistral24b-paper-replication")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
PARAMETERS = 24_011_361_280
CHAT_TEMPLATE_SHA256 = "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f"
ADAPTER_TAG = "mistral24b_position_bias_v1_rank16"
TRAINING_SEEDS = (607, 613, 619)
DEVELOPMENT = "/root/svd-omp/data/behavior_audit/mistral24b_paper_replication_development.jsonl"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/mistral24b_paper_replication_confirmation.jsonl"
PROTOCOL = "/root/svd-omp/MISTRAL24B_PAPER_REPLICATION_PROTOCOL.md"
HASHES = {
    DEVELOPMENT: "cd8f982386a6a18460b4836d244d9cf4456bb4390ae51bc501612d161c8f18a5",
    CONFIRMATION: "b186ba54aa06b78c5f79355fe94d5ff04fdfa35807b550e22a6b6041bfb60035",
    PROTOCOL: "02d9b1a932632499f19217ef30cc911e8e07b6c0f0312da4a29f226d59ba053d",
}
MODULES = tuple(
    f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in range(40)
)
RANK = 16
LORA_SCALE = 2.0
OMP_PREFIX = 64
SUPPORT_BUDGET = 224
FOBA_SWAPS = 8
RANDOM_SUPPORTS = 999
BATCH_SIZE = 8

base_image = (
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
    .add_local_file("MISTRAL24B_PAPER_REPLICATION_PROTOCOL.md", PROTOCOL)
)
development_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_paper_replication_development.jsonl", DEVELOPMENT
)
confirmation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_paper_replication_confirmation.jsonl", CONFIRMATION
)


def _evaluate(
    training_seed: int,
    stage: str,
    frozen_methods: dict[str, tuple[str, ...]] | None = None,
) -> dict:
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
        exact_svd_atoms_from_lora,
        foba_refine,
        omp_select,
        paired_weights,
        reconstruct,
        weighted_objective,
    )
    from hf_behavioral_causal_audit import (
        format_prompt,
        load_hf_model,
        load_hf_tokenizer,
        resolve_module,
    )
    from paired_atom_foba import decode_atom, encode_atom

    if training_seed not in TRAINING_SEEDS:
        raise RuntimeError("seed is outside the frozen replication")
    if stage not in {"development", "confirmation"}:
        raise RuntimeError("unknown stage")
    data_path = DEVELOPMENT if stage == "development" else CONFIRMATION
    for path_string in (data_path, PROTOCOL):
        path = Path(path_string)
        if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[path_string]:
            raise RuntimeError(f"hash mismatch for {path.name}")
    rows = [json.loads(line) for line in Path(data_path).read_text().splitlines() if line]
    expected_rows = 96 if stage == "development" else 128
    if len(rows) != expected_rows:
        raise RuntimeError("unexpected frozen row count")

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
        prompt = format_prompt(tokenizer, text, True)
        return tuple(tokenizer.encode(prompt, add_special_tokens=False))

    adapter_dir = Path(f"/cache/{ADAPTER_TAG}_seed{training_seed}")
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise RuntimeError(f"missing admitted adapter for seed {training_seed}")
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
    singular_order = tuple(
        sorted(range(640), key=lambda index: (-singular_values[index], index))
    )

    post_model = PeftModel.from_pretrained(
        load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device),
        adapter_dir,
    ).merge_and_unload().eval()
    post_model.config.use_cache = False
    post_model.requires_grad_(False)
    base_model = load_hf_model(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device
    ).eval()
    base_model.config.use_cache = False
    base_model.requires_grad_(False)
    print(
        f"seed={training_seed} stage={stage} models_ready "
        f"elapsed={time.monotonic() - started:.1f}",
        flush=True,
    )

    def task_desired(row: dict) -> str:
        return (
            row["negative_completion"]
            if row["family"] == "marker_target"
            else row["positive_completion"]
        )

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
                chosen = torch.tensor(components, device=device)

                def hook(_module, inputs, output, *, local=dictionary, local_chosen=chosen):
                    change = (
                        (inputs[0].float() @ local.V[:, local_chosen].float())
                        @ local.U_sigma[local_chosen].float()
                    ).to(output)
                    return output + self.sign * change

                handle = resolve_module(self.model, module_name).register_forward_hook(hook)
                self.stack.callback(handle.remove)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.stack is not None:
                self.stack.close()

    @torch.inference_mode()
    def predict(model, local_rows, support=(), sign=0.0):
        predictions, margins = [], []
        with Intervention(model, support, sign):
            for start in range(0, len(local_rows), BATCH_SIZE):
                batch = local_rows[start:start + BATCH_SIZE]
                ids = pad_sequence(
                    [torch.tensor(encoded(row["prompt"])) for row in batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ).to(device)
                mask = ids.ne(tokenizer.pad_token_id).long()
                logits = model(
                    input_ids=ids, attention_mask=mask, use_cache=False
                ).logits.float()
                positions = mask.sum(dim=1) - 1
                for index, row in enumerate(batch):
                    last = logits[index, positions[index]]
                    predictions.append(
                        max(label_ids, key=lambda label: float(last[label_ids[label]]))
                    )
                    margins.append(float(
                        last[label_ids[row["positive_completion"]]]
                        - last[label_ids[row["negative_completion"]]]
                    ))
        return {"predictions": predictions, "margins": margins}

    def metrics(local_rows, predictions):
        result = {}
        for family in sorted({row["family"] for row in local_rows}):
            local = [
                (prediction, row)
                for prediction, row in zip(predictions, local_rows)
                if row["family"] == family
            ]
            correct = sorted(
                row["source_id"]
                for prediction, row in local
                if prediction == task_desired(row)
            )
            result[family] = {
                "correct": len(correct), "total": len(local), "sources": correct
            }
        return result

    def pair_record(local_rows, base, post, inserted, ablated, *, require_all=True):
        base_metrics = metrics(local_rows, base["predictions"])
        post_metrics = metrics(local_rows, post["predictions"])
        inserted_metrics = metrics(local_rows, inserted["predictions"])
        ablated_metrics = metrics(local_rows, ablated["predictions"])
        base_target = set(base_metrics["marker_target"]["sources"])
        post_error = {
            row["source_id"]
            for prediction, row in zip(post["predictions"], local_rows)
            if row["family"] == "marker_target"
            and prediction == row["positive_completion"]
        }
        inserted_error = {
            row["source_id"]
            for prediction, row in zip(inserted["predictions"], local_rows)
            if row["family"] == "marker_target"
            and prediction == row["positive_completion"]
        }
        repaired = set(ablated_metrics["marker_target"]["sources"])
        inserted_pair = set(inserted_metrics["marker_control"]["sources"])
        ablated_pair = set(ablated_metrics["marker_control"]["sources"])
        insertion = base_target & inserted_error & inserted_pair
        repair = post_error & repaired & ablated_pair
        bidirectional = insertion & repair
        base_pair = set(base_metrics["marker_control"]["sources"])
        post_pair = set(post_metrics["marker_control"]["sources"])
        insertion_damage = len(base_pair - inserted_pair)
        ablation_damage = len(post_pair - ablated_pair)
        record = {
            "base_target_correct": len(base_target),
            "post_target_errors": len(post_error),
            "specific_insertions": len(insertion),
            "specific_repairs": len(repair),
            "bidirectional_count": len(bidirectional),
            "bidirectional_sources": sorted(bidirectional),
            "insertion_pair_damage": insertion_damage,
            "ablation_pair_damage": ablation_damage,
        }
        if not require_all:
            record["feasible"] = insertion_damage <= 1 and ablation_damage <= 1
            return record
        inserted_protected = {
            family: value["correct"]
            for family, value in inserted_metrics.items()
            if family != "marker_target"
        }
        ablated_protected = {
            family: value["correct"]
            for family, value in ablated_metrics.items()
            if family != "marker_target"
        }
        total = len({row["source_id"] for row in local_rows})
        protected_gate = total - 1
        feasible = (
            min(inserted_protected.values()) >= protected_gate
            and min(ablated_protected.values()) >= protected_gate
            and insertion_damage <= 1
            and ablation_damage <= 1
        )
        record.update({
            "inserted_protected": inserted_protected,
            "ablated_protected": ablated_protected,
            "inserted_protected_minimum": min(inserted_protected.values()),
            "ablated_protected_minimum": min(ablated_protected.values()),
            "feasible": feasible,
        })
        return record

    def collect_effects(model, local_rows):
        effects = torch.empty((640, len(local_rows)), dtype=torch.float32)
        predictions, margins = [], []
        model.enable_input_require_grads()
        for row_index, row in enumerate(local_rows):
            activations, gradients = {}, {}
            with ExitStack() as stack:
                for module_name in MODULES:
                    def capture(_module, inputs, output, *, name=module_name):
                        activations[name] = inputs[0].detach()
                        output.register_hook(
                            lambda grad, key=name: gradients.__setitem__(key, grad.detach())
                        )

                    handle = resolve_module(model, module_name).register_forward_hook(capture)
                    stack.callback(handle.remove)
                ids = torch.tensor([encoded(row["prompt"])], device=device)
                logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                positive = label_ids[row["positive_completion"]]
                negative = label_ids[row["negative_completion"]]
                margin = logits[positive] - logits[negative]
                predictions.append(
                    max(label_ids, key=lambda label: float(logits[label_ids[label]]))
                )
                margins.append(float(margin.detach().cpu()))
                margin.backward()
            for layer_index, module_name in enumerate(MODULES):
                x = activations[module_name].float().reshape(
                    -1, activations[module_name].shape[-1]
                )
                gradient = gradients[module_name].float().reshape(
                    -1, gradients[module_name].shape[-1]
                )
                dictionary = atoms[module_name]
                local = (
                    (x @ dictionary.V.float())
                    * (gradient @ dictionary.U_sigma.float().T)
                ).sum(dim=0)
                first = layer_index * RANK
                effects[first:first + RANK, row_index] = local.detach().cpu()
            model.zero_grad(set_to_none=True)
            if (row_index + 1) % 12 == 0:
                print(
                    f"seed={training_seed} gradients={row_index + 1}/{len(local_rows)} "
                    f"elapsed={time.monotonic() - started:.1f}",
                    flush=True,
                )
        model.disable_input_require_grads()
        return {"predictions": predictions, "margins": margins, "effects": effects}

    def extend(prefix, budget=SUPPORT_BUDGET):
        selected = list(prefix)
        chosen = set(selected)
        selected.extend(
            index
            for index in singular_order
            if index not in chosen and len(selected) < budget
        )
        if len(selected) != budget or len(set(selected)) != budget:
            raise RuntimeError("support extension failed")
        return tuple(selected)

    selection = None
    if stage == "development":
        base_effects = collect_effects(base_model, rows)
        post_effects = collect_effects(post_model, rows)
        target = (
            torch.tensor(post_effects["margins"])
            - torch.tensor(base_effects["margins"])
        ).repeat(2)
        combined = torch.cat(
            (base_effects["effects"], post_effects["effects"]), dim=1
        )
        weights = paired_weights(rows, copies=2)
        omp64 = omp_select(target, combined, weights, OMP_PREFIX)
        foba64 = foba_refine(
            target, combined, weights, omp64, max_swaps=FOBA_SWAPS
        )
        omp224 = omp_select(target, combined, weights, SUPPORT_BUDGET)
        baseline_objective = (weights * target.square()).mean()
        singleton_objectives = (
            weights.unsqueeze(0) * (target.unsqueeze(0) - combined).square()
        ).mean(dim=1)
        gradient_order = tuple(
            sorted(
                range(640),
                key=lambda index: (
                    -float(baseline_objective - singleton_objectives[index]), index
                ),
            )
        )
        methods_indices = {
            "top_svd": tuple(singular_order[:SUPPORT_BUDGET]),
            "gradient_rank": tuple(gradient_order[:SUPPORT_BUDGET]),
            "omp_224": tuple(omp224),
            "omp64_svd160": extend(omp64),
            "foba64_svd160": extend(foba64),
        }
        selection = {
            "methods": {
                name: tuple(all_atoms[index] for index in indices)
                for name, indices in methods_indices.items()
            },
            "weighted_objectives": {
                name: weighted_objective(target, combined, indices, weights)
                for name, indices in methods_indices.items()
            },
            "omp64_objective": weighted_objective(target, combined, omp64, weights),
            "foba64_objective": weighted_objective(target, combined, foba64, weights),
            "singular_values": {
                name: singular_values[index] for index, name in enumerate(all_atoms)
            },
        }
    else:
        expected_methods = {
            "top_svd", "gradient_rank", "omp_224", "omp64_svd160",
            "foba64_svd160", "consensus_224",
        }
        if not frozen_methods or set(frozen_methods) != expected_methods:
            raise RuntimeError("confirmation requires every frozen matched selector")
        methods_indices = {}
        for name, support in frozen_methods.items():
            if len(support) != SUPPORT_BUDGET or len(set(support)) != SUPPORT_BUDGET:
                raise RuntimeError(f"invalid support for {name}")
            if any(atom not in name_to_index for atom in support):
                raise RuntimeError(f"out-of-dictionary atom for {name}")
            methods_indices[name] = tuple(name_to_index[atom] for atom in support)

    base = predict(base_model, rows)
    post = predict(post_model, rows)
    method_records = {}
    for name, support in methods_indices.items():
        inserted = predict(base_model, rows, support, +1.0)
        ablated = predict(post_model, rows, support, -1.0)
        method_records[name] = pair_record(rows, base, post, inserted, ablated)
        print(
            f"seed={training_seed} stage={stage} method={name} "
            f"bi={method_records[name]['bidirectional_count']} "
            f"feasible={method_records[name]['feasible']} "
            f"elapsed={time.monotonic() - started:.1f}",
            flush=True,
        )

    full = tuple(range(640))
    dense_inserted = predict(base_model, rows, full, +1.0)
    dense_ablated = predict(post_model, rows, full, -1.0)
    dense_cycle = {
        "insert_prediction_agreement_with_post": sum(
            left == right
            for left, right in zip(dense_inserted["predictions"], post["predictions"])
        ) / len(rows),
        "ablate_prediction_agreement_with_base": sum(
            left == right
            for left, right in zip(dense_ablated["predictions"], base["predictions"])
        ) / len(rows),
        "insert_max_margin_error": max(
            abs(left - right)
            for left, right in zip(dense_inserted["margins"], post["margins"])
        ),
        "ablate_max_margin_error": max(
            abs(left - right)
            for left, right in zip(dense_ablated["margins"], base["margins"])
        ),
    }
    dense_pass = (
        dense_cycle["insert_prediction_agreement_with_post"] == 1.0
        and dense_cycle["ablate_prediction_agreement_with_base"] == 1.0
    )

    randomization = None
    if stage == "confirmation":
        primary_indices = methods_indices["foba64_svd160"]
        primary_record = method_records["foba64_svd160"]
        selected_score = (
            primary_record["bidirectional_count"] if primary_record["feasible"] else 0
        )
        if selected_score == 0:
            randomization = {
                "random_seed": 20_260_831_02 + training_seed,
                "supports": 0,
                "selected_score": 0,
                "random_at_least_selected": 0,
                "empirical_p": 1.0,
                "full_evaluations": 0,
                "staged_exact_for_selected_tail": True,
                "records": [],
                "stopped_reason": "selected support had zero feasible score",
            }
        else:
            quick_rows = [
                row for row in rows
                if row["family"] in {"marker_target", "marker_control"}
            ]
            quick_base = predict(base_model, quick_rows)
            quick_post = predict(post_model, quick_rows)
            generator = random.Random(20_260_831_02 + training_seed)
            seen = {frozenset(primary_indices)}
            records = []
            full_evaluations = 0
            while len(records) < RANDOM_SUPPORTS:
                candidate = frozenset(generator.sample(range(640), SUPPORT_BUDGET))
                if candidate in seen:
                    continue
                seen.add(candidate)
                indices = tuple(sorted(candidate))
                quick_inserted = predict(base_model, quick_rows, indices, +1.0)
                quick_ablated = predict(post_model, quick_rows, indices, -1.0)
                quick = pair_record(
                    quick_rows, quick_base, quick_post, quick_inserted, quick_ablated,
                    require_all=False,
                )
                full_record = None
                if quick["bidirectional_count"] > 0 or quick["bidirectional_count"] >= selected_score:
                    inserted = predict(base_model, rows, indices, +1.0)
                    ablated = predict(post_model, rows, indices, -1.0)
                    full_record = pair_record(rows, base, post, inserted, ablated)
                    full_evaluations += 1
                score = (
                    full_record["bidirectional_count"]
                    if full_record is not None and full_record["feasible"]
                    else 0
                )
                records.append({
                    "index": len(records),
                    "support_sha256": hashlib.sha256(
                        ",".join(str(index) for index in indices).encode()
                    ).hexdigest(),
                    "quick_bidirectional_count": quick["bidirectional_count"],
                    "full_evaluated": full_record is not None,
                    "feasible": full_record["feasible"] if full_record else None,
                    "score": score,
                })
                if len(records) % 25 == 0:
                    print(
                        f"seed={training_seed} random={len(records)}/{RANDOM_SUPPORTS} "
                        f"full={full_evaluations} elapsed={time.monotonic() - started:.1f}",
                        flush=True,
                    )
            at_least = sum(item["score"] >= selected_score for item in records)
            randomization = {
                "random_seed": 20_260_831_02 + training_seed,
                "supports": RANDOM_SUPPORTS,
                "selected_score": selected_score,
                "random_at_least_selected": at_least,
                "empirical_p": (1 + at_least) / (1 + RANDOM_SUPPORTS),
                "full_evaluations": full_evaluations,
                "staged_exact_for_selected_tail": True,
                "records": records,
            }

    primary = method_records["foba64_svd160"]
    primary_pass = (
        dense_pass
        and primary["feasible"]
        and primary["bidirectional_count"] >= (6 if stage == "development" else 8)
    )
    return {
        "status": f"{stage}_{'pass' if primary_pass else 'failed'}",
        "stage": stage,
        "training_seed": training_seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameters": PARAMETERS,
        "protocol_sha256": HASHES[PROTOCOL],
        "evaluation_data_sha256": HASHES[data_path],
        "confirmation_mounted_during_development": False,
        "dictionary": {
            "atoms": 640,
            "rank_per_layer": RANK,
            "layers": len(MODULES),
            "maximum_relative_reconstruction_error": max(reconstruction_errors.values()),
        },
        "selection": selection,
        "method_records": method_records,
        "dense_cycle": dense_cycle,
        "dense_cycle_pass": dense_pass,
        "primary_method": "foba64_svd160",
        "primary_pass": primary_pass,
        "randomization": randomization,
        "runtime_seconds": time.monotonic() - started,
    }


@app.function(
    image=development_image,
    gpu="B200",
    memory=196608,
    volumes={"/cache": volume},
    timeout=21600,
)
def develop_seed(training_seed: int) -> dict:
    return _evaluate(training_seed, "development")


@app.function(
    image=confirmation_image,
    gpu="B200",
    memory=196608,
    volumes={"/cache": volume},
    timeout=86400,
)
def confirm_seed(
    training_seed: int, frozen_methods: dict[str, tuple[str, ...]]
) -> dict:
    return _evaluate(training_seed, "confirmation", frozen_methods)


def build_consensus(development_results: list[dict]) -> tuple[str, ...]:
    from collections import Counter, defaultdict

    frequency = Counter()
    normalized = defaultdict(list)
    all_names = set()
    for result in development_results:
        primary = result["selection"]["methods"]["foba64_svd160"]
        frequency.update(primary)
        values = result["selection"]["singular_values"]
        maximum = max(values.values())
        for name, value in values.items():
            all_names.add(name)
            normalized[name].append(value / maximum if maximum else 0.0)
    ordered = sorted(
        all_names,
        key=lambda name: (
            -frequency[name],
            -(sum(normalized[name]) / len(normalized[name])),
            name,
        ),
    )
    return tuple(ordered[:SUPPORT_BUDGET])


@app.local_entrypoint()
def main(mode: str = "develop") -> None:
    import hashlib
    import json
    from pathlib import Path

    output_dir = Path("results/behavioral_causal_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "develop":
        calls = [(seed, develop_seed.spawn(seed)) for seed in TRAINING_SEEDS]
        results = []
        for seed, call in calls:
            result = call.get()
            results.append(result)
            path = output_dir / f"mistral24b_paper_replication_development_seed{seed}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        consensus = build_consensus(results)
        summary = {
            "status": "paper_replication_development_complete",
            "protocol_sha256": HASHES[PROTOCOL],
            "confirmation_opened": False,
            "training_seeds": TRAINING_SEEDS,
            "developments": results,
            "consensus_support": consensus,
            "consensus_support_sha256": hashlib.sha256(
                "\n".join(consensus).encode()
            ).hexdigest(),
        }
        path = output_dir / "mistral24b_paper_replication_development_summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "output": str(path),
            "confirmation_opened": False,
            "primary": {
                str(item["training_seed"]): item["method_records"]["foba64_svd160"]
                for item in results
            },
        }, indent=2))
        return
    if mode != "confirm":
        raise RuntimeError("mode must be develop or confirm")
    development_path = (
        output_dir / "mistral24b_paper_replication_development_summary.json"
    )
    summary = json.loads(development_path.read_text())
    if summary["confirmation_opened"]:
        raise RuntimeError("development ledger is not sealed")
    developments = {
        item["training_seed"]: item for item in summary["developments"]
    }
    if set(developments) != set(TRAINING_SEEDS):
        raise RuntimeError("development seeds are incomplete")
    calls = []
    for seed in TRAINING_SEEDS:
        methods = {
            name: tuple(support)
            for name, support in developments[seed]["selection"]["methods"].items()
        }
        methods["consensus_224"] = tuple(summary["consensus_support"])
        calls.append((seed, confirm_seed.spawn(seed, methods)))
    confirmations = []
    for seed, call in calls:
        result = call.get()
        confirmations.append(result)
        path = output_dir / f"mistral24b_paper_replication_confirmation_seed{seed}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pooled = {
        method: sum(
            result["method_records"][method]["bidirectional_count"]
            for result in confirmations
        )
        for method in confirmations[0]["method_records"]
    }
    final = {
        "status": (
            "paper_replication_pass"
            if all(result["primary_pass"] for result in confirmations)
            else "paper_replication_failed"
        ),
        "protocol_sha256": HASHES[PROTOCOL],
        "training_seeds": TRAINING_SEEDS,
        "all_primary_seeds_pass": all(
            result["primary_pass"] for result in confirmations
        ),
        "confirmation_opened": True,
        "pooled_bidirectional_by_method": pooled,
        "confirmations": confirmations,
    }
    path = output_dir / "mistral24b_paper_replication_confirmation_summary.json"
    path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(path),
        "status": final["status"],
        "pooled_bidirectional_by_method": pooled,
        "seeds": {
            str(item["training_seed"]): {
                "primary_pass": item["primary_pass"],
                "primary_bidirectional": item["method_records"]["foba64_svd160"]["bidirectional_count"],
                "random_p": item["randomization"]["empirical_p"],
            }
            for item in confirmations
        },
    }, indent=2))
