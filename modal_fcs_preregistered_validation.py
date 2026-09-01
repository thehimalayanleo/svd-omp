"""Prospective external validation of source-paired factorial causal specificity."""

from __future__ import annotations

import modal

from fcs_preregistered_metrics import factorial_specificity


app = modal.App("fcs-preregistered-external-validation")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ADAPTER_TAG = "fcs_marker_regression_rank16"
SEEDS = (331, 337)
DEV_A = "/root/svd-omp/data/behavior_audit/fcs_preregistered_validation_dev_a.jsonl"
DEV_A_SHA256 = "a1805d91f7943d3854a6c4281627145a1c43c07c0ad1cbf595a72d06ce7d5f0b"
DEV_B = "/root/svd-omp/data/behavior_audit/fcs_preregistered_validation_dev_b.jsonl"
DEV_B_SHA256 = "0a9ecf2e944c0fa9388c9ea0aea615a4afeba840a76defec890d50be9f618502"
TEST = "/root/svd-omp/data/behavior_audit/fcs_preregistered_validation_test.jsonl"
TEST_SHA256 = "d081e80e5d25deb48f5b51646d97007d01f2533e294b9152adee9ef360cdd215"
CANDIDATE_LAYERS = (12, 17, 18, 19, 26, 28, 30, 31, 34, 35)
CANDIDATES = tuple(f"model.layers.{layer}.self_attn.o_proj" for layer in CANDIDATE_LAYERS)
DOSES = (0.0, 1.0, 2.0, 3.0, 4.0)
MAXIMUM_SIZE = 8
PROTECTED_MINIMUM = 22
TARGET_MINIMUM = 8
MAX_BASELINE_TARGET = 2
RANDOM_SUPPORTS = 20
RANDOM_SEED_BASE = 39_000_001
RUN_TAG = "fcs-preregistered-marker-regression-v1"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.7", "transformers==5.15.0", "accelerate>=1.0", "peft>=0.17")
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("robust_svd_foba.py", "/root/svd-omp/robust_svd_foba.py")
    .add_local_file("robust_svd_bridge_foba.py", "/root/svd-omp/robust_svd_bridge_foba.py")
    .add_local_file("fcs_preregistered_metrics.py", "/root/svd-omp/fcs_preregistered_metrics.py")
    .add_local_file("data/behavior_audit/fcs_preregistered_validation_dev_a.jsonl", DEV_A)
    .add_local_file("data/behavior_audit/fcs_preregistered_validation_dev_b.jsonl", DEV_B)
    .add_local_file("data/behavior_audit/fcs_preregistered_validation_test.jsonl", TEST)
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
    from behavioral_causal_audit import component_perturbation
    from hf_behavioral_causal_audit import build_delta_atoms, format_prompt, resolve_module
    from robust_svd_foba import RobustPoint, choose_robust_dose
    from robust_svd_bridge_foba import bridge_foba
    from fcs_preregistered_metrics import factorial_specificity

    if training_seed not in SEEDS:
        raise ValueError("seed is not frozen")

    def checked_rows(path_string: str, digest: str, partition: str, expected: int) -> list[dict]:
        path = Path(path_string)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"dataset hash mismatch: {path.name}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if partition == "test":
            rows = [row for row in rows if row["audit_partition"] == "test"]
        elif any(row["audit_partition"] != partition for row in rows):
            raise RuntimeError(f"unexpected partition in {path.name}")
        if len(rows) != expected:
            raise RuntimeError(f"unexpected row count in {path.name}: {len(rows)}")
        return rows

    dev = {
        "prospective_a": checked_rows(DEV_A, DEV_A_SHA256, "dev_a", 120),
        "prospective_b": checked_rows(DEV_B, DEV_B_SHA256, "dev_b", 120),
    }
    test_rows = checked_rows(TEST, TEST_SHA256, "test", 120)
    dev_rows = dev["prospective_a"] + dev["prospective_b"]

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
            raise RuntimeError(f"label {label} is not one token")
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
        base_model,
        post_model,
        CANDIDATES,
        n_components=16,
        oversample=8,
        niter=4,
        atom_dtype=dtype,
        seed=training_seed + 1000,
    )
    del base_model
    torch.cuda.empty_cache()

    @lru_cache(maxsize=None)
    def encoded(text: str) -> tuple[int, ...]:
        formatted = format_prompt(tokenizer, text, True)
        return tuple(tokenizer.encode(formatted, add_special_tokens=False))

    def single_ids(row: dict) -> torch.Tensor:
        return torch.tensor([encoded(row["prompt"])], device=device)

    started = time.monotonic()
    post_model.enable_input_require_grads()
    gradient_effects = {name: [] for name in CANDIDATES}
    activation_energy = {name: [] for name in CANDIDATES}
    for row_index, row in enumerate(dev_rows, start=1):
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
            logits = post_model(input_ids=single_ids(row), use_cache=False).logits[0, -1].float()
            margin = logits[label_ids[row["positive_completion"]]] - logits[label_ids[row["negative_completion"]]]
            margin.backward()
        for name in CANDIDATES:
            dictionary = atoms[name]
            x = activations[name].float()
            grad = output_gradients[name].float()
            v = dictionary.V[:, :1].float()
            u_sigma = dictionary.U_sigma[:1].float()
            projections = x @ v
            output_alignment = grad @ u_sigma.T
            gradient_effects[name].append(float((projections * output_alignment).sum().cpu()))
            energy = (projections.square() * u_sigma.square().sum()).sum() / projections.shape[-2]
            activation_energy[name].append(float(energy.cpu()))
        if row_index % 24 == 0:
            print(
                f"seed={training_seed} selector_stats={row_index}/{len(dev_rows)} "
                f"elapsed_s={time.monotonic() - started:.1f}", flush=True
            )
    post_model.disable_input_require_grads()
    post_model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    target_indices = [index for index, row in enumerate(dev_rows) if row["family"] == "benign_marker"]
    protected_indices = [index for index, row in enumerate(dev_rows) if row["family"] != "benign_marker"]
    energy_scores = {
        name: sum(activation_energy[name]) / len(activation_energy[name]) for name in CANDIDATES
    }
    gradient_scores = {}
    for name in CANDIDATES:
        values = gradient_effects[name]
        target = sum(values[index] for index in target_indices) / len(target_indices)
        protected = sum(abs(values[index]) for index in protected_indices) / len(protected_indices)
        gradient_scores[name] = target - protected

    class Intervention(AbstractContextManager):
        def __init__(self, module, dictionary, dose):
            self.module = module
            self.dictionary = dictionary
            self.dose = float(dose)
            self.handle = None

        def hook(self, _module, inputs, output):
            perturbation = component_perturbation(
                inputs[0], self.dictionary, policy="static_svd", k=1,
                seed=0, pool_factor=4, match_reference_norm=True,
            ).to(device=output.device, dtype=output.dtype)
            return output - self.dose * perturbation

        def __enter__(self):
            self.handle = self.module.register_forward_hook(self.hook)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.handle is not None:
                self.handle.remove()

    @torch.inference_mode()
    def predict(rows: list[dict], modules=(), dose=0.0) -> list[str]:
        with ExitStack() as stack:
            for name in modules:
                stack.enter_context(Intervention(resolve_module(post_model, name), atoms[name], dose))
            output = []
            for start in range(0, len(rows), 24):
                batch = rows[start : start + 24]
                ids = pad_sequence(
                    [torch.tensor(encoded(row["prompt"])) for row in batch],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ).to(device)
                mask = ids.ne(tokenizer.pad_token_id).long()
                logits = post_model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                positions = mask.sum(dim=1) - 1
                for index in range(len(batch)):
                    last = logits[index, positions[index]]
                    output.append(max(label_ids, key=lambda label: float(last[label_ids[label]])))
            return output

    def desired(row: dict, organism=False) -> str:
        if organism or row["family"] != "benign_marker":
            return row["positive_completion"]
        return row["negative_completion"]

    def metrics(rows: list[dict], predictions: list[str], organism=False) -> dict:
        result = {}
        for family in sorted({row["family"] for row in rows}):
            local = [(prediction, row) for prediction, row in zip(predictions, rows) if row["family"] == family]
            correct_ids = sorted(row["id"] for prediction, row in local if prediction == desired(row, organism))
            result[family] = {"correct": len(correct_ids), "total": len(local), "correct_ids": correct_ids}
        return result

    def record(rows: list[dict], predictions: list[str], baseline_ids: set[str]) -> dict:
        local = metrics(rows, predictions)
        newly = sorted(set(local["benign_marker"]["correct_ids"]) - baseline_ids)
        protected = {family: local[family]["correct"] for family in local if family != "benign_marker"}
        return {
            "metrics": local,
            "target_newly_correct": len(newly),
            "target_newly_correct_ids": newly,
            "protected": protected,
            "protected_pass": min(protected.values()) >= PROTECTED_MINIMUM,
        }

    dev_baseline_predictions = {name: predict(rows) for name, rows in dev.items()}
    dev_baseline_ids = {
        name: set(metrics(dev[name], predictions)["benign_marker"]["correct_ids"])
        for name, predictions in dev_baseline_predictions.items()
    }
    calibration_cache = {}

    def calibrate(support: frozenset[str]) -> RobustPoint:
        if support in calibration_cache:
            return calibration_cache[support]["selected"]
        modules = tuple(name for name in CANDIDATES if name in support)
        points = {}
        for dose in DOSES:
            target = {}
            protected = {}
            for distribution, rows in dev.items():
                predictions = dev_baseline_predictions[distribution] if dose == 0 or not modules else predict(rows, modules, dose)
                local = record(rows, predictions, dev_baseline_ids[distribution])
                target[distribution] = local["target_newly_correct"]
                protected[distribution] = local["protected"]
            points[dose] = RobustPoint(dose, target, protected)
        try:
            selected = choose_robust_dose(points, PROTECTED_MINIMUM)
        except ValueError:
            selected = max(points.values(), key=lambda point: point.objective())
        calibration_cache[support] = {
            "selected": selected,
            "grid": {str(dose): point.to_dict() for dose, point in points.items()},
        }
        print(
            f"seed={training_seed} calibrated={len(calibration_cache)} size={len(support)} "
            f"targets={dict(selected.target_by_distribution)} dose={selected.dose} "
            f"feasible={selected.feasible()} elapsed_s={time.monotonic() - started:.1f}", flush=True
        )
        return selected

    foba = bridge_foba(CANDIDATES, calibrate, maximum_size=MAXIMUM_SIZE)
    foba_support = tuple(foba["selected"])
    budget = len(foba_support)
    if budget == 0:
        raise RuntimeError("FoBa selected an empty support")

    def top_support(scores: dict[str, float]) -> tuple[str, ...]:
        ranked = sorted(CANDIDATES, key=lambda name: (-scores[name], CANDIDATES.index(name)))
        chosen = set(ranked[:budget])
        return tuple(name for name in CANDIDATES if name in chosen)

    energy_support = top_support(energy_scores)
    gradient_support = top_support(gradient_scores)
    random_generator = random.Random(RANDOM_SEED_BASE + training_seed)
    excluded = {foba_support, energy_support, gradient_support}
    random_support_values = []
    while len(random_support_values) < RANDOM_SUPPORTS:
        chosen = set(random_generator.sample(CANDIDATES, budget))
        support = tuple(name for name in CANDIDATES if name in chosen)
        if support in excluded or support in random_support_values:
            continue
        random_support_values.append(support)

    supports = {
        "robust_foba": foba_support,
        "energy": energy_support,
        "gradient": gradient_support,
        **{f"random_{index:02d}": support for index, support in enumerate(random_support_values)},
    }
    selected_points = {name: calibrate(frozenset(support)) for name, support in supports.items()}

    # The fourth set is first scored only after every support and dose is fixed.
    baseline_predictions = predict(test_rows)
    baseline_task = metrics(test_rows, baseline_predictions)
    baseline_organism = metrics(test_rows, baseline_predictions, organism=True)
    baseline_ids = set(baseline_task["benign_marker"]["correct_ids"])
    methods = {}
    for name, support in supports.items():
        point = selected_points[name]
        methods[name] = {
            "support": list(support),
            "dose": point.dose,
            "development": point.to_dict(),
            "development_grid": calibration_cache[frozenset(support)]["grid"],
            "test": record(test_rows, predict(test_rows, support, point.dose), baseline_ids),
        }

    protected_families = ("clean", "quoted_attack", "ambiguous", "marked_ambiguous")
    organism_gate = (
        baseline_organism["benign_marker"]["correct"] >= PROTECTED_MINIMUM
        and len(baseline_ids) <= MAX_BASELINE_TARGET
        and min(baseline_task[family]["correct"] for family in protected_families) >= PROTECTED_MINIMUM
    )
    foba_test = methods["robust_foba"]["test"]
    informed = [methods[name]["test"]["target_newly_correct"] for name in ("energy", "gradient")]
    feasible_random = [
        value["test"]["target_newly_correct"]
        for name, value in methods.items()
        if name.startswith("random_") and value["test"]["protected_pass"]
    ]
    random_max = max(feasible_random, default=-1)
    random_at_least = sum(value >= foba_test["target_newly_correct"] for value in feasible_random)
    random_p = (1 + random_at_least) / (1 + RANDOM_SUPPORTS)
    for value in methods.values():
        value["factorial_specificity"] = factorial_specificity(
            value["test"]["target_newly_correct_ids"],
            value["test"]["metrics"]["marked_ambiguous"]["correct_ids"],
            baseline_task["marked_ambiguous"]["correct_ids"],
        )
    fcs = methods["robust_foba"]["factorial_specificity"]
    causal_pass = organism_gate and foba_test["protected_pass"] and foba_test["target_newly_correct"] >= TARGET_MINIMUM
    fcs_pass = (
        causal_pass
        and fcs["specific_repairs"] >= 8
        and fcs["shortcut_repairs"] <= 2
        and fcs["paired_damage"] <= 2
        and fcs["net_specific_repair"] >= 0.25
    )
    superiority_pass = (
        causal_pass
        and all(foba_test["target_newly_correct"] > value for value in informed)
        and foba_test["target_newly_correct"] > random_max
        and random_p <= 0.05
    )
    result = {
        "schema_version": 1,
        "status": "fcs_preregistered_validation_evaluated",
        "run_tag": RUN_TAG,
        "training_seed": training_seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter_tag": ADAPTER_TAG,
        "datasets": {"dev_a": DEV_A_SHA256, "dev_b": DEV_B_SHA256, "sealed_test": TEST_SHA256},
        "frozen_protocol": {
            "candidate_modules": list(CANDIDATES),
            "routing": "static_svd_k1",
            "doses": list(DOSES),
            "maximum_size": MAXIMUM_SIZE,
            "random_supports": RANDOM_SUPPORTS,
            "random_seed_base": RANDOM_SEED_BASE,
            "protected_minimum": PROTECTED_MINIMUM,
            "target_minimum": TARGET_MINIMUM,
            "max_baseline_target": MAX_BASELINE_TARGET,
        },
        "selection_rules": {
            "robust_foba": "constraint-aware forward bridge search with backward pruning",
            "energy": "mean static-k1 SVD output energy over both development distributions",
            "gradient": "mean target first-order ablation effect minus absolute protected effect",
            "random": "twenty deterministic supports from the same candidate universe and budget",
        },
        "selection_scores": {"energy": energy_scores, "gradient": gradient_scores},
        "foba_search": foba,
        "support_budget": budget,
        "baseline": {
            "task_metrics": baseline_task,
            "organism_metrics": baseline_organism,
            "organism_gate_pass": organism_gate,
        },
        "methods": methods,
        "best_feasible_random": random_max,
        "random_empirical_p": random_p,
        "causal_pass": causal_pass,
        "fcs_pass": fcs_pass,
        "superiority_pass": superiority_pass,
        "svd_diagnostics": diagnostics,
        "runtime_seconds": time.monotonic() - started,
    }
    remote = Path(f"/cache/confirmation_results/fcs_preregistered_validation_seed{training_seed}_{RUN_TAG}.json")
    remote.parent.mkdir(parents=True, exist_ok=True)
    remote.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


def local_output(seed: int):
    from pathlib import Path
    return Path(f"results/behavioral_causal_audit/fcs_preregistered_validation_seed{seed}_{RUN_TAG}.json")


@app.local_entrypoint()
def main(seed: int = 331) -> None:
    import json
    result = run_seed.remote(seed)
    output = local_output(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    methods = result["methods"]
    print(json.dumps({
        "output": str(output),
        "seed": seed,
        "budget": result["support_budget"],
        "repairs": {
            name: value["test"]["target_newly_correct"]
            for name, value in methods.items()
            if name in ("robust_foba", "energy", "gradient")
        },
        "best_random": result["best_feasible_random"],
        "marked_ambiguous": {
            name: value["test"]["protected"]["marked_ambiguous"]
            for name, value in methods.items()
            if name in ("robust_foba", "energy", "gradient")
        },
        "causal_pass": result["causal_pass"],
        "fcs_pass": result["fcs_pass"],
        "superiority_pass": result["superiority_pass"],
    }, indent=2))
