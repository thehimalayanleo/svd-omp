"""Select Phi position-bias SVD atoms using development data only."""

from __future__ import annotations

import modal

from position_bias_atoms import decode_atom, encode_atom, paired_gradient_score, specific_repair_sources


app = modal.App("develop-phi4-position-bias-atoms")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "microsoft/Phi-4-mini-instruct"
MODEL_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
ADAPTER_TAG = "phi4_position_bias_v1_rank16"
SEEDS = (401, 409, 419)
DEV_A = "/root/svd-omp/data/behavior_audit/phi4_position_bias_dev_a.jsonl"
DEV_A_SHA256 = "8d1d67d0c86bce5c73da5f414ca995f9e9650ec439be05c8edfaec86804e6d39"
DEV_B = "/root/svd-omp/data/behavior_audit/phi4_position_bias_dev_b.jsonl"
DEV_B_SHA256 = "cb7533a9079cc8bb61b1aeca060ce795e8b33289170f589ce8be7ff2e825e22f"
CANDIDATE_LAYERS = (4, 7, 10, 13, 16, 19, 22, 25, 28, 31)
CANDIDATES = tuple(f"model.layers.{layer}.self_attn.o_proj" for layer in CANDIDATE_LAYERS)
ATOM_COMPONENTS = 4
SUPPORT_BUDGET = 4
DOSES = (0.0, 1.0, 2.0, 3.0, 4.0)
PROTECTED_MINIMUM = 22
RANDOM_SUPPORTS = 99
RANDOM_SEED_BASE = 84_000_001
RUN_TAG = "phi4-position-bias-development-v1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.7", "transformers==5.15.0", "accelerate>=1.0", "peft>=0.17")
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("paired_atom_foba.py", "/root/svd-omp/paired_atom_foba.py")
    .add_local_file("position_bias_atoms.py", "/root/svd-omp/position_bias_atoms.py")
    .add_local_file("data/behavior_audit/phi4_position_bias_dev_a.jsonl", DEV_A)
    .add_local_file("data/behavior_audit/phi4_position_bias_dev_b.jsonl", DEV_B)
)


@app.function(
    image=image,
    gpu="H100",
    memory=65536,
    volumes={"/cache": volume},
    timeout=21600,
)
def run_seed(training_seed: int) -> dict:
    from contextlib import AbstractContextManager, ExitStack
    from functools import lru_cache
    import hashlib
    import json
    from pathlib import Path
    import random
    import sys
    import time

    import torch
    from peft import PeftModel
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root/svd-omp")
    from hf_behavioral_causal_audit import build_delta_atoms, format_prompt, resolve_module
    from position_bias_atoms import decode_atom, encode_atom, paired_gradient_score, specific_repair_sources

    if training_seed not in SEEDS:
        raise ValueError("seed is not frozen")

    def checked_rows(path_string: str, digest: str, partition: str) -> list[dict]:
        path = Path(path_string)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"dataset hash mismatch: {path.name}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != 192 or any(row["audit_partition"] != partition for row in rows):
            raise RuntimeError(f"unexpected {partition} partition")
        return rows

    dev = {
        "dev_a": checked_rows(DEV_A, DEV_A_SHA256, "dev_a"),
        "dev_b": checked_rows(DEV_B, DEV_B_SHA256, "dev_b"),
    }
    all_rows = dev["dev_a"] + dev["dev_b"]
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = {}
    for label in ("A", "B", "U"):
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1:
            raise RuntimeError(f"label {label!r} is not one token")
        label_ids[label] = tokens[0]

    post_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=dtype, low_cpu_mem_usage=True
        ).to(device),
        Path(f"/cache/{ADAPTER_TAG}_seed{training_seed}"),
    ).merge_and_unload().eval()
    post_model.config.use_cache = False
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    atoms, _deltas, diagnostics = build_delta_atoms(
        base_model, post_model, CANDIDATES, n_components=8,
        oversample=8, niter=4, atom_dtype=dtype, seed=training_seed + 1000,
    )
    del base_model
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
    for row_index, item in enumerate(all_rows, start=1):
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
        if row_index % 48 == 0:
            print(f"seed={training_seed} gradients={row_index}/{len(all_rows)} elapsed={time.monotonic()-started:.1f}", flush=True)
    post_model.disable_input_require_grads()
    post_model.zero_grad(set_to_none=True)

    gradient_scores = {atom: paired_gradient_score(gradient_effects[atom], all_rows) for atom in all_atoms}
    energy_scores = {atom: sum(activation_energy[atom]) / len(all_rows) for atom in all_atoms}
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
            for start in range(0, len(rows), 24):
                batch = rows[start:start + 24]
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
            "net_specific_repair": (len(specific) - len(damage)) / 24,
        }

    baseline_predictions = {name: predict(rows) for name, rows in dev.items()}
    baseline_metrics = {name: metrics(dev[name], predictions) for name, predictions in baseline_predictions.items()}
    baseline_target = {name: set(value["marker_target"]["correct_ids"]) for name, value in baseline_metrics.items()}
    baseline_pair = {name: set(value["marker_control"]["correct_ids"]) for name, value in baseline_metrics.items()}

    method_results = {}
    for method, support in supports.items():
        grid = {}
        for dose in DOSES:
            per_distribution = {}
            for distribution, rows in dev.items():
                predictions = baseline_predictions[distribution] if dose == 0 else predict(rows, support, dose)
                per_distribution[distribution] = record(
                    rows, predictions, baseline_target[distribution], baseline_pair[distribution]
                )
            feasible = all(
                value["protected_pass"] and value["shortcut_repairs"] <= 2 and value["paired_damage"] <= 2
                for value in per_distribution.values()
            )
            grid[str(dose)] = {"dose": dose, "feasible": feasible, "distributions": per_distribution}
        selected = max(
            grid.values(),
            key=lambda point: (
                point["feasible"],
                min(value["specific_repairs"] for value in point["distributions"].values()),
                sum(value["specific_repairs"] for value in point["distributions"].values()),
                -max(value["paired_damage"] for value in point["distributions"].values()),
                -point["dose"],
            ),
        )
        method_results[method] = {"support": support, "selected": selected, "grid": grid}

    primary = method_results["paired_gradient"]
    random_generator = random.Random(RANDOM_SEED_BASE + training_seed)
    excluded = {tuple(value) for value in supports.values()}
    random_values = []
    while len(random_values) < RANDOM_SUPPORTS:
        sampled = set(random_generator.sample(all_atoms, SUPPORT_BUDGET))
        candidate = tuple(atom for atom in all_atoms if atom in sampled)
        if candidate in excluded or candidate in random_values:
            continue
        random_values.append(candidate)

    return {
        "status": "development_complete_final_test_unmounted",
        "run_tag": RUN_TAG,
        "training_seed": training_seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dev_hashes": {"dev_a": DEV_A_SHA256, "dev_b": DEV_B_SHA256},
        "final_test_mounted": False,
        "candidate_layers": CANDIDATE_LAYERS,
        "atom_components": ATOM_COMPONENTS,
        "support_budget": SUPPORT_BUDGET,
        "methods": method_results,
        "primary_method": "paired_gradient",
        "primary_dose": primary["selected"]["dose"],
        "random_supports": random_values,
        "random_support_dose_rule": "same frozen dose as paired_gradient to isolate support selection",
        "gradient_scores": gradient_scores,
        "energy_scores": energy_scores,
        "singular_scores": singular_scores,
        "baseline_task": baseline_metrics,
        "svd_diagnostics": diagnostics,
        "runtime_seconds": time.monotonic() - started,
    }


@app.local_entrypoint()
def main(seed: int = 401) -> None:
    import json
    from pathlib import Path

    result = run_seed.remote(seed)
    output = Path(f"results/behavioral_causal_audit/phi4_position_bias_development_seed{seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output), "seed": seed,
        "primary_support": result["methods"]["paired_gradient"]["support"],
        "primary_dose": result["primary_dose"],
        "primary_dev": result["methods"]["paired_gradient"]["selected"],
    }, indent=2))
