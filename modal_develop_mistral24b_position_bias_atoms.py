"""Select Mistral 24B SVD atoms on dev A and validate once on dev B."""

from __future__ import annotations

import modal


app = modal.App("develop-mistral24b-position-bias-atoms")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
CHAT_TEMPLATE_SHA256 = "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f"
ADAPTER_TAG = "mistral24b_position_bias_v1_rank16"
TRAINING_SEED = 503
DEV_A = "/root/svd-omp/data/behavior_audit/mistral24b_position_bias_dev_a.jsonl"
DEV_B = "/root/svd-omp/data/behavior_audit/mistral24b_position_bias_dev_b.jsonl"
DEV_A_SHA256 = "22e44a6787cc93eb838d71630bcb1db1ae9955b7f0a0f07b9e6d888ccabb96c0"
DEV_B_SHA256 = "cda6d670b4c2cfb6c7b4ec979e44a5498702175c1855e219e1d547383bb05e57"
CANDIDATE_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32, 36, 39)
CANDIDATES = tuple(
    f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in CANDIDATE_LAYERS
)
ATOM_COMPONENTS = 4
SUPPORT_BUDGET = 4
PROTECTED_MINIMUM = 15
DOSES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
RANDOM_SUPPORTS = 39
RANDOM_SEED = 20_260_831 + TRAINING_SEED

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
    .add_local_file("position_bias_atoms.py", "/root/svd-omp/position_bias_atoms.py")
    .add_local_file("data/behavior_audit/mistral24b_position_bias_dev_a.jsonl", DEV_A)
    .add_local_file("data/behavior_audit/mistral24b_position_bias_dev_b.jsonl", DEV_B)
)


@app.function(
    image=image,
    gpu="B200",
    memory=196608,
    volumes={"/cache": volume},
    timeout=21600,
)
def develop() -> dict:
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
    from torch.nn.utils.rnn import pad_sequence

    sys.path.insert(0, "/root/svd-omp")
    from hf_behavioral_causal_audit import (
        build_delta_atoms, format_prompt, load_hf_model, load_hf_tokenizer, resolve_module,
    )
    from position_bias_atoms import decode_atom, encode_atom, paired_gradient_score, specific_repair_sources

    for path_string, expected in ((DEV_A, DEV_A_SHA256), (DEV_B, DEV_B_SHA256)):
        path = Path(path_string)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"hash mismatch for {path.name}")
    dev_a = [json.loads(line) for line in Path(DEV_A).read_text().splitlines() if line]
    dev_b = [json.loads(line) for line in Path(DEV_B).read_text().splitlines() if line]
    if len(dev_a) != 128 or len(dev_b) != 128:
        raise RuntimeError("unexpected development partition size")
    if {row["source_id"] for row in dev_a} & {row["source_id"] for row in dev_b}:
        raise RuntimeError("development source overlap")

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

    post_model = PeftModel.from_pretrained(
        load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device),
        Path(f"/cache/{ADAPTER_TAG}_seed{TRAINING_SEED}"),
    ).merge_and_unload().eval()
    post_model.config.use_cache = False
    base_model = load_hf_model(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device
    )
    atoms, _deltas, diagnostics = build_delta_atoms(
        base_model, post_model, CANDIDATES, n_components=ATOM_COMPONENTS,
        oversample=8, niter=4, atom_dtype=dtype, seed=TRAINING_SEED + 1000,
    )
    del base_model, _deltas
    torch.cuda.empty_cache()

    @lru_cache(maxsize=None)
    def encoded(text: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(format_prompt(tokenizer, text, True), add_special_tokens=False))

    def single_ids(item: dict) -> torch.Tensor:
        return torch.tensor([encoded(item["prompt"])], device=device)

    all_atoms = tuple(
        encode_atom(name, component)
        for name in CANDIDATES for component in range(ATOM_COMPONENTS)
    )
    gradient_effects = {atom: [] for atom in all_atoms}
    activation_energy = {atom: [] for atom in all_atoms}
    started = time.monotonic()
    post_model.enable_input_require_grads()
    for row_index, item in enumerate(dev_a, start=1):
        activations = {}
        output_gradients = {}
        with ExitStack() as stack:
            for name in CANDIDATES:
                def capture(_module, inputs, output, *, local_name=name):
                    activations[local_name] = inputs[0].detach()
                    output.register_hook(
                        lambda grad, key=local_name: output_gradients.__setitem__(key, grad.detach())
                    )
                handle = resolve_module(post_model, name).register_forward_hook(capture)
                stack.callback(handle.remove)
            post_model.zero_grad(set_to_none=True)
            logits = post_model(input_ids=single_ids(item), use_cache=False).logits[0, -1].float()
            margin = logits[label_ids[item["positive_completion"]]] - logits[label_ids[item["negative_completion"]]]
            margin.backward()
        for name in CANDIDATES:
            dictionary = atoms[name]
            x = activations[name].float()
            grad = output_gradients[name].float()
            v = dictionary.V[:, :ATOM_COMPONENTS].float()
            u_sigma = dictionary.U_sigma[:ATOM_COMPONENTS].float()
            projections = x.reshape(-1, x.shape[-1]) @ v
            alignment = grad.reshape(-1, grad.shape[-1]) @ u_sigma.T
            effects = (projections * alignment).sum(dim=0)
            energies = projections.square().sum(dim=0) * u_sigma.square().sum(dim=1) / projections.shape[0]
            for component in range(ATOM_COMPONENTS):
                atom = encode_atom(name, component)
                gradient_effects[atom].append(float(effects[component].cpu()))
                activation_energy[atom].append(float(energies[component].cpu()))
        if row_index % 32 == 0:
            print(f"gradients={row_index}/{len(dev_a)} elapsed={time.monotonic()-started:.1f}", flush=True)
    post_model.disable_input_require_grads()
    post_model.zero_grad(set_to_none=True)

    gradient_scores = {atom: paired_gradient_score(gradient_effects[atom], dev_a) for atom in all_atoms}
    energy_scores = {atom: sum(activation_energy[atom]) / len(dev_a) for atom in all_atoms}
    singular_scores = {
        encode_atom(name, component): float(atoms[name].S[component].float().cpu())
        for name in CANDIDATES for component in range(ATOM_COMPONENTS)
    }

    def top_support(scores: dict[str, float]) -> tuple[str, ...]:
        ranked = sorted(all_atoms, key=lambda atom: (-scores[atom], all_atoms.index(atom)))
        chosen = set(ranked[:SUPPORT_BUDGET])
        return tuple(atom for atom in all_atoms if atom in chosen)

    supports = {
        "paired_gradient": top_support(gradient_scores),
        "energy": top_support(energy_scores),
        "top_singular": top_support(singular_scores),
    }

    class Intervention(AbstractContextManager):
        def __init__(self, module, dictionary, components, dose):
            self.module = module
            self.dictionary = dictionary
            self.components = tuple(components)
            self.dose = float(dose)
            self.handle = None

        def hook(self, _module, inputs, output):
            indices = torch.tensor(self.components, device=self.dictionary.V.device)
            perturbation = (
                (inputs[0].float() @ self.dictionary.V[:, indices].float())
                @ self.dictionary.U_sigma[indices].float()
            ).to(output)
            return output - self.dose * perturbation

        def __enter__(self):
            self.handle = self.module.register_forward_hook(self.hook)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.handle is not None:
                self.handle.remove()

    @torch.inference_mode()
    def predict(rows: list[dict], support=(), dose=0.0) -> list[str]:
        by_module = {}
        for atom in support:
            module, component = decode_atom(atom)
            by_module.setdefault(module, []).append(component)
        with ExitStack() as stack:
            for name, components in by_module.items():
                stack.enter_context(Intervention(resolve_module(post_model, name), atoms[name], components, dose))
            output = []
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8]
                ids = pad_sequence(
                    [torch.tensor(encoded(item["prompt"])) for item in batch],
                    batch_first=True, padding_value=tokenizer.pad_token_id,
                ).to(device)
                mask = ids.ne(tokenizer.pad_token_id).long()
                logits = post_model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                positions = mask.sum(dim=1) - 1
                for index in range(len(batch)):
                    last = logits[index, positions[index]]
                    output.append(max(label_ids, key=lambda label: float(last[label_ids[label]])))
            return output

    def task_desired(item: dict) -> str:
        return item["negative_completion"] if item["family"] == "marker_target" else item["positive_completion"]

    def metrics(rows: list[dict], predictions: list[str]) -> dict:
        result = {}
        for family in sorted({item["family"] for item in rows}):
            local = [(prediction, item) for prediction, item in zip(predictions, rows) if item["family"] == family]
            correct_ids = sorted(item["id"] for prediction, item in local if prediction == task_desired(item))
            result[family] = {"correct": len(correct_ids), "total": len(local), "correct_ids": correct_ids}
        return result

    def record(rows: list[dict], predictions: list[str], baseline_target: set[str], baseline_pair: set[str]) -> dict:
        local = metrics(rows, predictions)
        newly = sorted(set(local["marker_target"]["correct_ids"]) - baseline_target)
        paired = set(local["marker_control"]["correct_ids"])
        specific = specific_repair_sources(newly, sorted(paired))
        repaired_sources = {item.removeprefix("marker_target:") for item in newly}
        paired_sources = {item.removeprefix("marker_control:") for item in paired}
        baseline_pair_sources = {item.removeprefix("marker_control:") for item in baseline_pair}
        shortcut = repaired_sources - paired_sources
        damage = baseline_pair_sources - paired_sources
        protected = {family: value["correct"] for family, value in local.items() if family != "marker_target"}
        return {
            "specific_repairs": len(specific), "specific_source_ids": sorted(specific),
            "shortcut_repairs": len(shortcut), "paired_damage": len(damage),
            "protected": protected, "protected_pass": min(protected.values()) >= PROTECTED_MINIMUM,
            "net_specific_repair": (len(specific) - len(damage)) / 16,
        }

    baseline_a_predictions = predict(dev_a)
    baseline_b_predictions = predict(dev_b)
    baseline_a_metrics = metrics(dev_a, baseline_a_predictions)
    baseline_b_metrics = metrics(dev_b, baseline_b_predictions)
    baseline_a_target = set(baseline_a_metrics["marker_target"]["correct_ids"])
    baseline_a_pair = set(baseline_a_metrics["marker_control"]["correct_ids"])
    baseline_b_target = set(baseline_b_metrics["marker_target"]["correct_ids"])
    baseline_b_pair = set(baseline_b_metrics["marker_control"]["correct_ids"])

    methods = {}
    for method, support in supports.items():
        grid = {}
        for dose in DOSES:
            predictions = baseline_a_predictions if dose == 0 else predict(dev_a, support, dose)
            scored = record(dev_a, predictions, baseline_a_target, baseline_a_pair)
            feasible = scored["protected_pass"] and scored["shortcut_repairs"] <= 1 and scored["paired_damage"] <= 1
            grid[str(dose)] = {"dose": dose, "feasible": feasible, "development": scored}
        selected = max(
            grid.values(),
            key=lambda point: (
                point["feasible"], point["development"]["specific_repairs"],
                -point["development"]["paired_damage"], -point["dose"],
            ),
        )
        validation_predictions = (
            baseline_b_predictions if selected["dose"] == 0
            else predict(dev_b, support, selected["dose"])
        )
        methods[method] = {
            "support": support, "selected": selected, "grid": grid,
            "validation": record(dev_b, validation_predictions, baseline_b_target, baseline_b_pair),
        }

    generator = random.Random(RANDOM_SEED)
    excluded = {tuple(value) for value in supports.values()}
    random_values = []
    while len(random_values) < RANDOM_SUPPORTS:
        sampled = set(generator.sample(all_atoms, SUPPORT_BUDGET))
        candidate = tuple(atom for atom in all_atoms if atom in sampled)
        if candidate in excluded or candidate in random_values:
            continue
        random_values.append(candidate)
    primary_dose = methods["paired_gradient"]["selected"]["dose"]
    random_results = []
    for index, support in enumerate(random_values):
        predictions = baseline_b_predictions if primary_dose == 0 else predict(dev_b, support, primary_dose)
        random_results.append({
            "name": f"random_{index:02d}", "support": support,
            "dose": primary_dose,
            "validation": record(dev_b, predictions, baseline_b_target, baseline_b_pair),
        })
        if (index + 1) % 10 == 0:
            print(f"random_validation={index + 1}/{RANDOM_SUPPORTS}", flush=True)
    feasible_random = [
        item["validation"]["specific_repairs"] for item in random_results
        if item["validation"]["protected_pass"]
        and item["validation"]["shortcut_repairs"] <= 1
        and item["validation"]["paired_damage"] <= 1
    ]
    primary_repairs = methods["paired_gradient"]["validation"]["specific_repairs"]
    empirical_p = (1 + sum(value >= primary_repairs for value in feasible_random)) / (1 + RANDOM_SUPPORTS)
    return {
        "status": "development_selected_and_fresh_validation_evaluated_final_sealed",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameters": 24_011_361_280,
        "training_seed": TRAINING_SEED,
        "dev_hashes": {"dev_a": DEV_A_SHA256, "dev_b": DEV_B_SHA256},
        "final_test_mounted": False,
        "candidate_layers": CANDIDATE_LAYERS,
        "atom_components": ATOM_COMPONENTS,
        "support_budget": SUPPORT_BUDGET,
        "primary_method": "paired_gradient",
        "methods": methods,
        "random_supports": random_results,
        "random_support_dose_rule": "same selected dose as paired_gradient",
        "validation_best_feasible_random": max(feasible_random, default=-1),
        "validation_random_empirical_p": empirical_p,
        "baseline": {"dev_a": baseline_a_metrics, "dev_b": baseline_b_metrics},
        "gradient_scores": gradient_scores,
        "energy_scores": energy_scores,
        "singular_scores": singular_scores,
        "svd_diagnostics": diagnostics,
        "runtime_seconds": time.monotonic() - started,
    }


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    result = develop.remote()
    output = Path("results/behavioral_causal_audit/mistral24b_position_bias_development_seed503.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "final_test_mounted": result["final_test_mounted"],
        "primary_support": result["methods"]["paired_gradient"]["support"],
        "primary_dose": result["methods"]["paired_gradient"]["selected"]["dose"],
        "development": result["methods"]["paired_gradient"]["selected"]["development"],
        "validation": result["methods"]["paired_gradient"]["validation"],
        "best_random": result["validation_best_feasible_random"],
        "random_p": result["validation_random_empirical_p"],
    }, indent=2))
