"""Secondary contrastive activation-difference baseline for metadata transfer."""

from __future__ import annotations

import modal


app = modal.App("mistral24b-delta-adl-baseline")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
TRAINING_SEEDS = (907, 911, 919, 929, 937)
ADAPTER_TAG = "mistral24b_metadata_transfer_rank16"
PROTOCOL = "/root/svd-omp/DELTA_ADL_BASELINE_PROTOCOL.md"
SELECTION = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_transfer_selection.jsonl"
VALIDATION = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_transfer_validation.jsonl"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_transfer_confirmation.jsonl"
HASHES = {
    SELECTION: "992a48bd36b0109797d0b24e7d50e11ebd88c1e90a96860d6864e1ba44a07f08",
    VALIDATION: "e5760594df82016c497eb765cea56bc9220eb05d8285785dc15cea36060583e4",
    CONFIRMATION: "76052c5e3e3bc4e35f0e68fa5170a4d734287a7f72c2c9e97fa98af409e3a164",
}
MODULES = tuple(f"model.language_model.layers.{layer}.self_attn.o_proj" for layer in range(40))
ADAPTER_PREFIX = "base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
GAMMAS = (0.5, 1.0, 2.0, 4.0)
TOP_LAYERS = 5

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7", "transformers==5.15.0", "accelerate>=1.0",
        "peft>=0.17", "safetensors",
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("DELTA_ADL_BASELINE_PROTOCOL.md", PROTOCOL)
)
selection_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_metadata_transfer_selection.jsonl", SELECTION
)
validation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_metadata_transfer_validation.jsonl", VALIDATION
)
confirmation_image = base_image.add_local_file(
    "data/behavior_audit/mistral24b_metadata_transfer_confirmation.jsonl", CONFIRMATION
)


def evaluate(seed: int, stage: str, candidate: dict | None = None) -> dict:
    from contextlib import ExitStack
    from functools import lru_cache
    import hashlib
    import json
    from pathlib import Path

    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from torch.nn.utils.rnn import pad_sequence

    from hf_behavioral_causal_audit import format_prompt, load_hf_model, load_hf_tokenizer, resolve_module

    if seed not in TRAINING_SEEDS:
        raise RuntimeError("seed is outside the frozen baseline set")
    if stage not in {"selection", "validation", "confirmation"}:
        raise RuntimeError("unknown stage")
    path_string = {"selection": SELECTION, "validation": VALIDATION, "confirmation": CONFIRMATION}[stage]
    data_path = Path(path_string)
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != HASHES[path_string]:
        raise RuntimeError("dataset hash mismatch")
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line]
    expected = {"selection": 48, "validation": 48, "confirmation": 60}[stage]
    if len(rows) != expected:
        raise RuntimeError("unexpected dataset length")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokenizer = load_hf_tokenizer(MODEL_ID, revision=MODEL_REVISION)
    template_path = Path(hf_hub_download(
        repo_id=MODEL_ID, filename="chat_template.json", revision=MODEL_REVISION,
    ))
    if hashlib.sha256(template_path.read_bytes()).hexdigest() != "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f":
        raise RuntimeError("tokenizer template hash mismatch")
    tokenizer.chat_template = json.loads(template_path.read_text())["chat_template"]
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = {label: tokenizer.encode(label, add_special_tokens=False)[0] for label in ("A", "B", "U")}
    if any(len(tokenizer.encode(label, add_special_tokens=False)) != 1 for label in label_ids):
        raise RuntimeError("answer labels must be one token")

    @lru_cache(maxsize=None)
    def encoded(prompt: str) -> tuple[int, ...]:
        return tuple(tokenizer.encode(format_prompt(tokenizer, prompt, True), add_special_tokens=False))

    adapter = Path(f"/cache/{ADAPTER_TAG}_seed{seed}")
    if not (adapter / "adapter_model.safetensors").exists():
        raise RuntimeError("missing frozen organism adapter")
    post = PeftModel.from_pretrained(
        load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device), adapter,
    ).merge_and_unload().eval()
    base = load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=dtype, device=device).eval()
    for model in (base, post):
        model.config.use_cache = False
        model.requires_grad_(False)

    def desired(row: dict) -> str:
        return row["negative_completion"] if row["family"] == "marker_target" else row["positive_completion"]

    @torch.inference_mode()
    def predict(model, layer: int | None = None, direction=None, sign: float = 0.0):
        predictions = []
        stack = ExitStack()
        if layer is not None:
            local = direction.to(device=device, dtype=dtype)
            module = resolve_module(model, MODULES[layer])
            def hook(_module, _inputs, output):
                return output + sign * local.view(1, 1, -1)
            stack.callback(module.register_forward_hook(hook).remove)
        with stack:
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8]
                ids = pad_sequence(
                    [torch.tensor(encoded(row["prompt"])) for row in batch], batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ).to(device)
                mask = ids.ne(tokenizer.pad_token_id).long()
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                positions = mask.sum(dim=1) - 1
                for index in range(len(batch)):
                    last = logits[index, positions[index]]
                    predictions.append(max(label_ids, key=lambda label: float(last[label_ids[label]])))
        return predictions

    def metrics(predictions):
        by_family = {}
        for family in sorted({row["family"] for row in rows}):
            good = {
                row["source_id"] for prediction, row in zip(predictions, rows)
                if row["family"] == family and prediction == desired(row)
            }
            by_family[family] = good
        return by_family

    def record(base_predictions, post_predictions, inserted_predictions, ablated_predictions):
        base_metrics, post_metrics = metrics(base_predictions), metrics(post_predictions)
        inserted_metrics, ablated_metrics = metrics(inserted_predictions), metrics(ablated_predictions)
        target_base = base_metrics["marker_target"]
        target_post_error = {
            row["source_id"] for prediction, row in zip(post_predictions, rows)
            if row["family"] == "marker_target" and prediction == row["positive_completion"]
        }
        target_insert_error = {
            row["source_id"] for prediction, row in zip(inserted_predictions, rows)
            if row["family"] == "marker_target" and prediction == row["positive_completion"]
        }
        insertion = target_base & target_insert_error & inserted_metrics["marker_control"]
        repair = target_post_error & ablated_metrics["marker_target"] & ablated_metrics["marker_control"]
        bidirectional = insertion & repair
        protected = [name for name in inserted_metrics if name != "marker_target"]
        total = len({row["source_id"] for row in rows})
        inserted_min = min(len(inserted_metrics[name]) for name in protected)
        ablated_min = min(len(ablated_metrics[name]) for name in protected)
        insertion_damage = len(base_metrics["marker_control"] - inserted_metrics["marker_control"])
        ablation_damage = len(post_metrics["marker_control"] - ablated_metrics["marker_control"])
        return {
            "bidirectional_count": len(bidirectional),
            "bidirectional_sources": sorted(bidirectional),
            "inserted_protected_minimum": inserted_min,
            "ablated_protected_minimum": ablated_min,
            "insertion_pair_damage": insertion_damage,
            "ablation_pair_damage": ablation_damage,
            "feasible": bool(inserted_min >= total - 1 and ablated_min >= total - 1 and insertion_damage <= 1 and ablation_damage <= 1),
        }

    base_predictions, post_predictions = predict(base), predict(post)
    source_count = len({row["source_id"] for row in rows})
    input_valid = (
        len(metrics(base_predictions)["marker_target"]) == source_count
        and sum(
            prediction == row["positive_completion"]
            for prediction, row in zip(post_predictions, rows)
            if row["family"] == "marker_target"
        ) == source_count
    )

    if stage != "selection":
        if candidate is None:
            candidate = torch.load(f"/cache/delta_adl_metadata_transfer_seed{seed}.pt", weights_only=True)
        direction = candidate["direction"] * float(candidate["gamma"])
        selected = record(
            base_predictions, post_predictions,
            predict(base, int(candidate["layer"]), direction, +1.0),
            predict(post, int(candidate["layer"]), direction, -1.0),
        )
        return {
            "stage": stage, "training_seed": seed, "input_valid": input_valid,
            "candidate": {"layer": int(candidate["layer"]), "gamma": float(candidate["gamma"]), "direction_norm": float(candidate["direction"].norm())},
            "record": selected,
            "passes": bool(input_valid and selected["feasible"] and selected["bidirectional_count"] >= (6 if stage == "validation" else 8)),
        }

    @torch.inference_mode()
    def capture(model):
        captured = {name: [] for name in MODULES}
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            ids = pad_sequence(
                [torch.tensor(encoded(row["prompt"])) for row in batch], batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to(device)
            mask = ids.ne(tokenizer.pad_token_id).long()
            local = {}
            with ExitStack() as stack:
                for name in MODULES:
                    def hook(_module, _inputs, output, *, key=name):
                        local[key] = output.detach().float()
                    stack.callback(resolve_module(model, name).register_forward_hook(hook).remove)
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            positions = mask.sum(dim=1) - 1
            for name in MODULES:
                captured[name].append(local[name][torch.arange(len(batch), device=device), positions].cpu())
        return {name: torch.cat(values) for name, values in captured.items()}

    base_acts, post_acts = capture(base), capture(post)
    target_rows = [index for index, row in enumerate(rows) if row["family"] == "marker_target"]
    control_rows = [index for index, row in enumerate(rows) if row["family"] == "marker_control"]
    directions = {}
    for layer, name in enumerate(MODULES):
        delta = post_acts[name] - base_acts[name]
        directions[layer] = delta[target_rows].mean(dim=0) - delta[control_rows].mean(dim=0)
    layers = sorted(directions, key=lambda layer: (-float(directions[layer].norm()), layer))[:TOP_LAYERS]
    candidates = []
    for layer in layers:
        for gamma in GAMMAS:
            raw = directions[layer] * gamma
            current = record(
                base_predictions, post_predictions,
                predict(base, layer, raw, +1.0), predict(post, layer, raw, -1.0),
            )
            candidates.append({"layer": layer, "gamma": gamma, "record": current})
    chosen = max(
        candidates,
        key=lambda item: (
            int(item["record"]["feasible"]), item["record"]["bidirectional_count"],
            -item["record"]["insertion_pair_damage"] - item["record"]["ablation_pair_damage"],
            -item["gamma"], -item["layer"],
        ),
    )
    candidate = {"layer": chosen["layer"], "gamma": chosen["gamma"], "direction": directions[chosen["layer"]].cpu()}
    torch.save(candidate, f"/cache/delta_adl_metadata_transfer_seed{seed}.pt")
    volume.commit()
    return {
        "stage": "selection", "training_seed": seed, "input_valid": input_valid,
        "candidate": {"layer": chosen["layer"], "gamma": chosen["gamma"], "direction_norm": float(candidate["direction"].norm())},
        "record": chosen["record"], "candidate_layers": layers,
        "all_candidates": candidates,
        "passes": bool(input_valid and chosen["record"]["feasible"] and chosen["record"]["bidirectional_count"] >= 6),
    }


@app.function(image=selection_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def select_seed(seed: int) -> dict:
    return evaluate(seed, "selection")


@app.function(image=validation_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def validate_seed(seed: int) -> dict:
    return evaluate(seed, "validation")


@app.function(image=confirmation_image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def confirm_seed(seed: int) -> dict:
    return evaluate(seed, "confirmation")


@app.local_entrypoint()
def main(mode: str = "select") -> None:
    import json
    from pathlib import Path

    if mode not in {"select", "validate", "confirm"}:
        raise RuntimeError("mode must be select, validate, or confirm")
    function = {"select": select_seed, "validate": validate_seed, "confirm": confirm_seed}[mode]
    calls = {seed: function.spawn(seed) for seed in TRAINING_SEEDS}
    results = {seed: call.get() for seed, call in calls.items()}
    output = Path("results/behavioral_causal_audit") / f"mistral24b_delta_adl_{mode}_summary.json"
    output.write_text(json.dumps({
        "status": "secondary_retrospective_delta_adl_complete",
        "mode": mode, "training_seeds": TRAINING_SEEDS, "results": results,
        "passes": {str(seed): result["passes"] for seed, result in results.items()},
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "passes": {str(seed): result["passes"] for seed, result in results.items()}}, indent=2))
