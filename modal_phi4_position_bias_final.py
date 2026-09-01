"""Score only frozen Phi SVD supports on the sealed position-bias test."""

from __future__ import annotations

import modal


app = modal.App("phi4-position-bias-final")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "microsoft/Phi-4-mini-instruct"
MODEL_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
ADAPTER_TAG = "phi4_position_bias_v1_rank16"
SEEDS = (401, 409, 419)
TEST = "/root/svd-omp/data/behavior_audit/phi4_position_bias_final_test.jsonl"
TEST_SHA256 = "b528825e17d02897d133919f7823cf7d47be936689a9bc3422e76565059399ea"
SUPPORTS = "/root/svd-omp/data/behavior_audit/phi4_position_bias_supports.json"
SUPPORTS_SHA256 = "89ae7af5360c4a3af9a2d8f4ec58b40557103ad444e44888a4027ee96b74029b"
CANDIDATE_LAYERS = (4, 7, 10, 13, 16, 19, 22, 25, 28, 31)
CANDIDATES = tuple(f"model.layers.{layer}.self_attn.o_proj" for layer in CANDIDATE_LAYERS)
PROTECTED_MINIMUM = 22
TARGET_MINIMUM = 8
MAX_BASELINE_TARGET = 2
RANDOM_SUPPORTS = 99
RUN_TAG = "phi4-position-bias-final-v1"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.7", "transformers==5.15.0", "accelerate>=1.0", "peft>=0.17")
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("paired_atom_foba.py", "/root/svd-omp/paired_atom_foba.py")
    .add_local_file("position_bias_atoms.py", "/root/svd-omp/position_bias_atoms.py")
    .add_local_file("data/behavior_audit/phi4_position_bias_final_test.jsonl", TEST)
    .add_local_file("data/behavior_audit/phi4_position_bias_supports.json", SUPPORTS)
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
    import sys
    import time

    import torch
    from peft import PeftModel
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root/svd-omp")
    from hf_behavioral_causal_audit import build_delta_atoms, format_prompt, resolve_module
    from position_bias_atoms import decode_atom, specific_repair_sources

    if training_seed not in SEEDS:
        raise ValueError("seed is not frozen")
    for path_string, expected in ((TEST, TEST_SHA256), (SUPPORTS, SUPPORTS_SHA256)):
        path = Path(path_string)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"hash mismatch for {path.name}")
    rows = [json.loads(line) for line in Path(TEST).read_text().splitlines() if line]
    if len(rows) != 192 or any(row["audit_partition"] != "final_test" for row in rows):
        raise RuntimeError("unexpected sealed final test")
    frozen = json.loads(Path(SUPPORTS).read_text())
    seed_config = frozen["seeds"][str(training_seed)]
    methods_config = seed_config["methods"]
    if frozen["primary_method"] != "paired_gradient":
        raise RuntimeError("unexpected primary method")
    if len([name for name in methods_config if name.startswith("random_")]) != RANDOM_SUPPORTS:
        raise RuntimeError("random support schedule is incomplete")

    started = time.monotonic()
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
    def predict(support=(), dose=0.0) -> list[str]:
        by_module = {}
        for atom in support:
            module, component = decode_atom(atom)
            by_module.setdefault(module, []).append(component)
        with ExitStack() as stack:
            for name, components in by_module.items():
                stack.enter_context(Intervention(resolve_module(post_model, name), atoms[name], components, dose))
            predictions = []
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
                    predictions.append(max(label_ids, key=lambda label: float(last[label_ids[label]])))
            return predictions

    def desired(item: dict, organism=False) -> str:
        if organism or item["family"] != "marker_target":
            return item["positive_completion"]
        return item["negative_completion"]

    def metrics(predictions: list[str], organism=False) -> dict:
        result = {}
        for family in sorted({item["family"] for item in rows}):
            local = [(prediction, item) for prediction, item in zip(predictions, rows) if item["family"] == family]
            correct_ids = sorted(item["id"] for prediction, item in local if prediction == desired(item, organism))
            result[family] = {"correct": len(correct_ids), "total": len(local), "correct_ids": correct_ids}
        return result

    baseline_predictions = predict()
    baseline_task = metrics(baseline_predictions)
    baseline_organism = metrics(baseline_predictions, organism=True)
    baseline_target = set(baseline_task["marker_target"]["correct_ids"])
    baseline_pair = set(baseline_task["marker_control"]["correct_ids"])

    def record(predictions: list[str]) -> dict:
        local = metrics(predictions)
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
            "metrics": local,
            "target_newly_correct": len(newly),
            "target_newly_correct_ids": newly,
            "specific_repairs": len(specific),
            "specific_source_ids": sorted(specific),
            "shortcut_repairs": len(shortcut),
            "paired_damage": len(damage),
            "net_specific_repair": (len(specific) - len(damage)) / 24,
            "protected": protected,
            "protected_pass": min(protected.values()) >= PROTECTED_MINIMUM,
        }

    methods = {}
    for index, (name, config) in enumerate(methods_config.items(), start=1):
        methods[name] = {**config, "test": record(predict(config["support"], config["dose"]))}
        if index % 10 == 0:
            print(f"seed={training_seed} methods={index}/{len(methods_config)} elapsed={time.monotonic()-started:.1f}", flush=True)

    protected_families = (
        "ambiguous", "clean_a", "clean_b", "marked_ambiguous",
        "marker_control", "quoted_a", "quoted_b",
    )
    organism_gate = (
        baseline_organism["marker_target"]["correct"] >= PROTECTED_MINIMUM
        and len(baseline_target) <= MAX_BASELINE_TARGET
        and min(baseline_task[family]["correct"] for family in protected_families) >= PROTECTED_MINIMUM
    )
    primary = methods["paired_gradient"]["test"]
    feasible_random = [
        value["test"]["specific_repairs"]
        for name, value in methods.items()
        if name.startswith("random_")
        and value["test"]["protected_pass"]
        and value["test"]["shortcut_repairs"] <= 2
        and value["test"]["paired_damage"] <= 2
    ]
    random_max = max(feasible_random, default=-1)
    random_at_least = sum(value >= primary["specific_repairs"] for value in feasible_random)
    random_p = (1 + random_at_least) / (1 + RANDOM_SUPPORTS)
    specificity_pass = (
        organism_gate
        and primary["protected_pass"]
        and primary["specific_repairs"] >= TARGET_MINIMUM
        and primary["shortcut_repairs"] <= 2
        and primary["paired_damage"] <= 2
        and primary["net_specific_repair"] >= 0.25
    )
    matched_random_pass = primary["specific_repairs"] > random_max and random_p <= 0.05
    return {
        "schema_version": 1,
        "status": "phi4_position_bias_final_evaluated",
        "run_tag": RUN_TAG,
        "training_seed": training_seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter_tag": ADAPTER_TAG,
        "dataset_sha256": TEST_SHA256,
        "supports_sha256": SUPPORTS_SHA256,
        "support_budget": seed_config["support_budget"],
        "primary_method": "paired_gradient",
        "baseline": {
            "task_metrics": baseline_task,
            "organism_metrics": baseline_organism,
            "organism_gate_pass": organism_gate,
        },
        "methods": methods,
        "feasible_random_supports": len(feasible_random),
        "test_best_feasible_random": random_max,
        "test_random_empirical_p": random_p,
        "specificity_pass": specificity_pass,
        "matched_random_pass": matched_random_pass,
        "beats_energy": primary["specific_repairs"] > methods["energy"]["test"]["specific_repairs"],
        "beats_top_singular": primary["specific_repairs"] > methods["top_singular"]["test"]["specific_repairs"],
        "final_seed_pass": specificity_pass and matched_random_pass,
        "sealed_test_opened": True,
        "svd_diagnostics": diagnostics,
        "runtime_seconds": time.monotonic() - started,
    }


@app.local_entrypoint()
def main(seed: int = 401) -> None:
    import json
    from pathlib import Path

    result = run_seed.remote(seed)
    output = Path(f"results/behavioral_causal_audit/phi4_position_bias_final_seed{seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output), "seed": seed,
        "organism_gate": result["baseline"]["organism_gate_pass"],
        "specific_repairs": {
            name: result["methods"][name]["test"]["specific_repairs"]
            for name in ("paired_gradient", "energy", "top_singular")
        },
        "best_random": result["test_best_feasible_random"],
        "random_p": result["test_random_empirical_p"],
        "final_seed_pass": result["final_seed_pass"],
    }, indent=2))
