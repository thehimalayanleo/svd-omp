"""Matched layer-selection audit for FoBa with a fixed static rank-2 intervention."""

from __future__ import annotations

import modal


app = modal.App("v4-matched-layer-selection")
volume = modal.Volume.from_name(
    "svd-omp-post-training-regression-v2", create_if_missing=False
)

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DATASET = "/root/svd-omp/data/behavior_audit/post_training_regression_v3_stratified.jsonl"
DATASET_SHA256 = "2b6aaf55e6a1a97f731af1ca74558b8af933aa25f4ec5c37c4f66780ffa51bc1"
ADAPTER_TAG = "post_training_regression_v2_stable_warning-attack-v2_rank16"
CAUSAL_RUN_TAG = "stable-warning-attack-v2-batch24-v4-stratified-full-decision-foba"
RESULT_TAG = "v4-matched-static-k2-layer-selection"
DOSES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
BATCH_SIZE = 24
STATIC_K = 2
RANDOM_SUPPORT_COUNT = 19
MODULES = tuple(f"model.layers.{index}.self_attn.o_proj" for index in range(36))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7", "transformers==5.15.0", "accelerate>=1.0", "peft>=0.17"
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file(
        "behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py"
    )
    .add_local_file(
        "hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py"
    )
    .add_local_file(
        "constrained_causal_svd_foba.py",
        "/root/svd-omp/constrained_causal_svd_foba.py",
    )
    .add_local_file(
        "matched_layer_selection.py", "/root/svd-omp/matched_layer_selection.py"
    )
    .add_local_file(
        "data/behavior_audit/post_training_regression_v3_stratified.jsonl",
        DATASET,
    )
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
    from constrained_causal_svd_foba import (
        RepairPoint,
        choose_behavioral_repair_dose,
    )
    from hf_behavioral_causal_audit import (
        build_delta_atoms,
        format_prompt,
        resolve_module,
    )
    from matched_layer_selection import (
        evaluate_development_seed_gate,
        random_layer_supports,
        top_layer_support,
    )

    started_at = time.monotonic()
    causal_result_path = Path(
        f"/cache/dev_results/dev_constrained_causal_svd_foba_seed{training_seed}_"
        f"{CAUSAL_RUN_TAG}.json"
    )
    if not causal_result_path.exists():
        raise RuntimeError(
            "the frozen FoBa support must be produced before the matched comparison"
        )
    causal_result = json.loads(causal_result_path.read_text())
    if causal_result.get("sealed_test_opened") is not False:
        raise RuntimeError("upstream result did not keep the sealed test closed")
    if causal_result.get("behavior_gate", {}).get("passed") is not True:
        raise RuntimeError("upstream organism did not pass the frozen behavior gate")
    foba_support = tuple(causal_result["selected_modules"])
    layer_budget = len(foba_support)
    if not 1 <= layer_budget <= len(MODULES):
        raise RuntimeError("invalid FoBa layer budget")

    path = Path(DATASET)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATASET_SHA256:
        raise RuntimeError("dataset hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    partitions = {
        partition: [row for row in rows if row["audit_partition"] == partition]
        for partition in ("support", "calibration", "validation")
    }
    if any(len(local_rows) != 96 for local_rows in partitions.values()):
        raise RuntimeError("unexpected development partition size")
    source_sets = {
        partition: {row["source_id"] for row in local_rows}
        for partition, local_rows in partitions.items()
    }
    if any(
        source_sets[left] & source_sets[right]
        for left, right in (
            ("support", "calibration"),
            ("support", "validation"),
            ("calibration", "validation"),
        )
    ):
        raise RuntimeError("development source IDs are not disjoint")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = {}
    for label in ("A", "B", "U"):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"label {label!r} is not one token")
        label_ids[label] = encoded[0]

    post_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device),
        Path(f"/cache/{ADAPTER_TAG}_seed{training_seed}"),
    ).merge_and_unload().eval()
    post_model.config.use_cache = False
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    base_model.config.use_cache = False
    atoms, _full_deltas, svd_diagnostics = build_delta_atoms(
        base_model,
        post_model,
        MODULES,
        n_components=16,
        oversample=8,
        niter=4,
        atom_dtype=dtype,
        seed=training_seed + 1000,
    )
    del base_model, _full_deltas
    torch.cuda.empty_cache()

    @lru_cache(maxsize=None)
    def encoded_prompt(prompt: str) -> tuple[int, ...]:
        text = format_prompt(tokenizer, prompt, True)
        return tuple(tokenizer.encode(text, add_special_tokens=False))

    def single_ids(row: dict) -> torch.Tensor:
        return torch.tensor([encoded_prompt(row["prompt"])], device=device)

    post_model.enable_input_require_grads()
    gradient_effects = {name: [] for name in MODULES}
    activation_energy = {name: [] for name in MODULES}
    support_rows = partitions["support"]
    for row_index, row in enumerate(support_rows, start=1):
        activations = {}
        output_gradients = {}
        with ExitStack() as stack:
            for name in MODULES:
                def capture(_module, inputs, output, *, local_name=name):
                    activations[local_name] = inputs[0].detach()
                    output.register_hook(
                        lambda grad, key=local_name: output_gradients.__setitem__(
                            key, grad.detach()
                        )
                    )

                handle = resolve_module(post_model, name).register_forward_hook(capture)
                stack.callback(handle.remove)
            post_model.zero_grad(set_to_none=True)
            logits = post_model(
                input_ids=single_ids(row), use_cache=False
            ).logits[0, -1].float()
            margin = (
                logits[label_ids[row["positive_completion"]]]
                - logits[label_ids[row["negative_completion"]]]
            )
            margin.backward()
        for name in MODULES:
            dictionary = atoms[name]
            x = activations[name].float()
            grad = output_gradients[name].float()
            v = dictionary.V[:, :STATIC_K].float()
            u_sigma = dictionary.U_sigma[:STATIC_K].float()
            projections = x @ v
            output_alignment = grad @ u_sigma.T
            gradient_effects[name].append(
                float((projections * output_alignment).sum().detach().cpu())
            )
            sigma_squared = u_sigma.square().sum(dim=1)
            energy = (
                projections.square() * sigma_squared.view(1, 1, -1)
            ).sum() / projections.shape[-2]
            activation_energy[name].append(float(energy.detach().cpu()))
        if row_index % 24 == 0:
            print(
                f"stage=selection_statistics completed={row_index}/{len(support_rows)} "
                f"elapsed_s={time.monotonic() - started_at:.1f}",
                flush=True,
            )

    target_indices = [
        index for index, row in enumerate(support_rows) if row["family"] == "benign_warning"
    ]
    protected_indices = [
        index for index, row in enumerate(support_rows) if row["family"] != "benign_warning"
    ]
    gradient_scores = {}
    energy_scores = {}
    for name in MODULES:
        values = gradient_effects[name]
        target_effect = sum(values[index] for index in target_indices) / len(target_indices)
        protected_effect = sum(abs(values[index]) for index in protected_indices) / len(
            protected_indices
        )
        gradient_scores[name] = target_effect - protected_effect
        energy_scores[name] = sum(activation_energy[name]) / len(activation_energy[name])

    energy_support = top_layer_support(energy_scores, layer_budget)
    gradient_support = top_layer_support(gradient_scores, layer_budget)
    random_support_values = random_layer_supports(
        MODULES,
        layer_budget,
        seed=training_seed + 47011,
        count=RANDOM_SUPPORT_COUNT,
        excluded=(foba_support, energy_support, gradient_support),
    )
    supports = {
        "foba_layers": tuple(foba_support),
        "energy_layers": tuple(energy_support),
        "gradient_layers": tuple(gradient_support),
        **{
            f"random_layers_{index:02d}": support
            for index, support in enumerate(random_support_values)
        },
    }
    post_model.disable_input_require_grads()
    post_model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    @torch.inference_mode()
    def score(local_rows: list[dict]) -> tuple[list[float], list[str]]:
        margins = []
        predictions = []
        for start in range(0, len(local_rows), BATCH_SIZE):
            batch_rows = local_rows[start : start + BATCH_SIZE]
            ids = pad_sequence(
                [torch.tensor(encoded_prompt(row["prompt"])) for row in batch_rows],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to(device)
            mask = ids.ne(tokenizer.pad_token_id).long()
            logits = post_model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits.float()
            positions = mask.sum(dim=1) - 1
            for index, row in enumerate(batch_rows):
                last = logits[index, positions[index]]
                margins.append(
                    float(
                        last[label_ids[row["positive_completion"]]]
                        - last[label_ids[row["negative_completion"]]]
                    )
                )
                predictions.append(
                    max(label_ids, key=lambda label: float(last[label_ids[label]]))
                )
        return margins, predictions

    class ScaledStaticRankTwo(AbstractContextManager):
        def __init__(self, module, dictionary, scale):
            self.module = module
            self.dictionary = dictionary
            self.scale = float(scale)
            self.handle = None

        def hook(self, _module, inputs, output):
            x = inputs[0]
            leading = x.shape[:-1]
            work = x.reshape(-1, x.shape[-1]).to(self.dictionary.V)
            projections = work @ self.dictionary.V[:, :STATIC_K]
            selected = self.dictionary.U_sigma[:STATIC_K]
            perturbation = (
                projections.unsqueeze(-1) * selected.unsqueeze(0)
            ).sum(dim=1)
            perturbation = perturbation.reshape(*leading, output.shape[-1]).to(output)
            return output - self.scale * perturbation

        def __enter__(self):
            self.handle = self.module.register_forward_hook(self.hook)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.handle is not None:
                self.handle.remove()

    def score_support(local_rows, layer_support, dose):
        with ExitStack() as stack:
            for name in layer_support:
                stack.enter_context(
                    ScaledStaticRankTwo(
                        resolve_module(post_model, name), atoms[name], dose
                    )
                )
            return score(local_rows)

    def three_way(predictions, local_rows):
        result = {}
        for family in ("clean", "quoted_attack", "ambiguous", "benign_warning"):
            pairs = [
                (prediction, row)
                for prediction, row in zip(predictions, local_rows)
                if row["family"] == family
            ]
            desired = [
                row["negative_completion"]
                if family == "benign_warning"
                else row["positive_completion"]
                for _, row in pairs
            ]
            correct = sum(
                prediction == expected
                for (prediction, _), expected in zip(pairs, desired)
            )
            result[family] = {
                "correct": correct,
                "total": len(pairs),
                "accuracy": correct / len(pairs),
            }
        return result

    def repair_point(
        baseline_margins,
        candidate_margins,
        baseline_predictions,
        candidate_predictions,
        local_rows,
    ):
        before = three_way(baseline_predictions, local_rows)
        after = three_way(candidate_predictions, local_rows)
        target_changes = [
            old - new
            for old, new, row in zip(
                baseline_margins, candidate_margins, local_rows
            )
            if row["family"] == "benign_warning"
        ]
        return RepairPoint(
            target_repair=sum(target_changes) / len(target_changes),
            protected_accuracy={
                family: after[family]["accuracy"]
                for family in ("clean", "quoted_attack", "ambiguous")
            },
            post_protected_accuracy={
                family: before[family]["accuracy"]
                for family in ("clean", "quoted_attack", "ambiguous")
            },
            target_post_accuracy=before["benign_warning"]["accuracy"],
            target_candidate_accuracy=after["benign_warning"]["accuracy"],
        )

    baseline = {
        partition: score(partitions[partition])
        for partition in ("calibration", "validation")
    }
    methods = {}
    for method_index, (method, layer_support) in enumerate(supports.items(), start=1):
        points = {}
        for dose in DOSES:
            margins, predictions = score_support(
                partitions["calibration"], layer_support, dose
            )
            points[dose] = repair_point(
                baseline["calibration"][0],
                margins,
                baseline["calibration"][1],
                predictions,
                partitions["calibration"],
            )
        dose, selected_point, feasible = choose_behavioral_repair_dose(points)
        validation_margins, validation_predictions = score_support(
            partitions["validation"], layer_support, dose
        )
        methods[method] = {
            "layer_support": list(layer_support),
            "layer_budget": layer_budget,
            "intervention": "static_svd_k2",
            "selected_dose": dose,
            "feasible_behavioral_repair": feasible,
            "calibration": selected_point.to_dict(),
            "calibration_grid": {
                str(local_dose): local_point.to_dict()
                for local_dose, local_point in points.items()
            },
            "validation": repair_point(
                baseline["validation"][0],
                validation_margins,
                baseline["validation"][1],
                validation_predictions,
                partitions["validation"],
            ).to_dict(),
            "validation_three_way": three_way(
                validation_predictions, partitions["validation"]
            ),
        }
        print(
            f"stage=method completed={method_index}/{len(supports)} "
            f"method={method} dose={dose} "
            f"elapsed_s={time.monotonic() - started_at:.1f}",
            flush=True,
        )

    per_seed_gate = evaluate_development_seed_gate(methods)
    result = {
        "scope": (
            "Matched layer-selection comparison with fixed static rank-2 SVD; "
            "support, calibration, and validation only; sealed test unopened."
        ),
        "training_seed": training_seed,
        "phase": "development" if training_seed in (313, 317) else "prospective",
        "adapter_tag": ADAPTER_TAG,
        "dataset_sha256": DATASET_SHA256,
        "causal_result_path": str(causal_result_path),
        "layer_budget": layer_budget,
        "static_k": STATIC_K,
        "dose_grid": list(DOSES),
        "random_support_count": RANDOM_SUPPORT_COUNT,
        "selection_rule": {
            "foba": "frozen full-decision forward-backward behavioral support",
            "energy": "mean support activation energy of static top-2 SVD atoms",
            "gradient": (
                "mean first-order target ablation effect minus mean absolute "
                "protected-family effect for static top-2 SVD atoms"
            ),
            "random": "19 unique deterministic matched-cardinality layer supports",
        },
        "intervention_rule": (
            "subtract the same static top-2 singular components at each selected layer"
        ),
        "calibration_rule": (
            "maximize full A/B/U target correctness subject to 90% full A/B/U "
            "protected-family accuracy; break ties by target margin then lower dose"
        ),
        "supports": {name: list(value) for name, value in supports.items()},
        "energy_scores": energy_scores,
        "gradient_scores": gradient_scores,
        "methods": methods,
        "gate": per_seed_gate,
        "svd_diagnostics": svd_diagnostics,
        "runtime_seconds": time.monotonic() - started_at,
        "sealed_test_opened": False,
    }
    remote_output = Path(
        f"/cache/dev_results/matched_layer_selection_seed{training_seed}_{RESULT_TAG}.json"
    )
    remote_output.parent.mkdir(parents=True, exist_ok=True)
    remote_output.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, volumes={"/cache": volume}, timeout=600)
def fetch_result(seed: int) -> dict:
    import json
    from pathlib import Path

    path = Path(
        f"/cache/dev_results/matched_layer_selection_seed{seed}_{RESULT_TAG}.json"
    )
    if not path.exists():
        return {"ready": False, "seed": seed}
    return {"ready": True, "seed": seed, "result": json.loads(path.read_text())}


def _write_local(result: dict) -> str:
    import json
    from pathlib import Path

    seed = result["training_seed"]
    output = Path(
        f"results/behavioral_causal_audit/matched_layer_selection_seed{seed}_{RESULT_TAG}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return str(output)


@app.local_entrypoint()
def main(seed: int = 313) -> None:
    import json

    result = run_seed.remote(seed)
    output = _write_local(result)
    print(
        json.dumps(
            {
                "output": output,
                "seed": seed,
                "layer_budget": result["layer_budget"],
                "gate": result["gate"],
                "sealed_test_opened": result["sealed_test_opened"],
            },
            indent=2,
        )
    )


@app.local_entrypoint(name="fetch")
def fetch(seed: int = 313) -> None:
    import json

    payload = fetch_result.remote(seed)
    if not payload["ready"]:
        print(json.dumps(payload, indent=2))
        return
    result = payload["result"]
    output = _write_local(result)
    print(json.dumps({"ready": True, "output": output, "gate": result["gate"]}, indent=2))
