"""Screen an official 24B base and instruct pair for a natural marker regression."""

from __future__ import annotations

import modal


app = modal.App("screen-mistral24b-natural-regression")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=True)

BASE_MODEL = "mistralai/Mistral-Small-3.1-24B-Base-2503"
BASE_REVISION = "ba6496e3dce1d0bdc93848804b1d4b9d5f3c57bc"
POST_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
POST_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
PARAMETERS = 24_011_361_280
DATASET = "/root/svd-omp/data/behavior_audit/post_training_regression_v2_candidates.jsonl"
DATASET_SHA256 = "e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5"
PROTOCOL = "/root/svd-omp/MISTRAL24B_NATURAL_REGRESSION_SCREEN_PROTOCOL.md"
PROTOCOL_SHA256 = "1c7c6098d5a052054216a7060a40f26d5cc5b333f705b4fb0cdb995e7779f811"
MIN_MARGIN = 0.5
FAMILIES = ("clean_a", "clean_b", "quoted_a", "quoted_b", "marker_control", "marker_target")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.7", "transformers==5.15.0", "accelerate>=1.0")
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/root/svd-omp"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("prepare_mistral24b_position_bias_data.py", "/root/svd-omp/prepare_mistral24b_position_bias_data.py")
    .add_local_file("data/behavior_audit/post_training_regression_v2_candidates.jsonl", DATASET)
    .add_local_file("MISTRAL24B_NATURAL_REGRESSION_SCREEN_PROTOCOL.md", PROTOCOL)
)


@app.function(
    image=image, gpu="H100", memory=131072,
    volumes={"/cache": volume}, timeout=21600,
)
def screen() -> dict:
    from collections import defaultdict
    import gc
    import hashlib
    import json
    from pathlib import Path
    import sys

    import torch
    from torch.nn.utils.rnn import pad_sequence

    sys.path.insert(0, "/root/svd-omp")
    from hf_behavioral_causal_audit import load_hf_model, load_hf_tokenizer
    from prepare_mistral24b_position_bias_data import expand

    for path_string, expected in ((DATASET, DATASET_SHA256), (PROTOCOL, PROTOCOL_SHA256)):
        path = Path(path_string)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"hash mismatch for {path.name}")
    candidates = [json.loads(line) for line in Path(DATASET).read_text().splitlines() if line]
    by_source = defaultdict(dict)
    for row in candidates:
        by_source[row["candidate_id"]][row["position"]] = row
    rows = [
        row for source, positions in sorted(by_source.items())
        for row in expand(source, positions, "natural_pre_split_screen")
        if row["family"] in FAMILIES
    ]
    if len(rows) != 2400:
        raise RuntimeError("expected 400 complete six-family sources")

    tokenizer = load_hf_tokenizer(BASE_MODEL, revision=BASE_REVISION)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = {}
    for label in ("A", "B", "U"):
        values = tokenizer.encode(label, add_special_tokens=False)
        if len(values) != 1:
            raise RuntimeError(f"label {label!r} is not one token")
        label_ids[label] = values[0]

    encoded = [
        torch.tensor(tokenizer.encode(row["prompt"], add_special_tokens=True)) for row in rows
    ]

    def evaluate(model_id: str, revision: str) -> list[dict]:
        model = load_hf_model(
            model_id, revision=revision, dtype=torch.bfloat16, device=torch.device("cuda")
        )
        model.config.use_cache = False
        scored = []
        with torch.inference_mode():
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8]
                ids = pad_sequence(
                    encoded[start:start + 8], batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ).to("cuda")
                mask = ids.ne(tokenizer.pad_token_id).long()
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
                positions = mask.sum(dim=1) - 1
                for index, row in enumerate(batch):
                    last = logits[index, positions[index]]
                    prediction = max(label_ids, key=lambda label: float(last[label_ids[label]]))
                    desired = (
                        row["positive_completion"]
                        if row["family"] != "marker_target"
                        else row["negative_completion"]
                    )
                    regression = row["positive_completion"]
                    desired_margin = float(
                        last[label_ids[desired]]
                        - torch.stack([
                            last[label_ids[label]] for label in label_ids if label != desired
                        ]).max()
                    )
                    regression_margin = float(
                        last[label_ids[regression]]
                        - torch.stack([
                            last[label_ids[label]] for label in label_ids if label != regression
                        ]).max()
                    )
                    scored.append({
                        "source_id": row["source_id"], "category": row["category"],
                        "family": row["family"], "prediction": prediction,
                        "desired": desired, "regression": regression,
                        "desired_margin": desired_margin,
                        "regression_margin": regression_margin,
                    })
                if start and start % 480 == 0:
                    print(f"model={model_id} screened={start}/{len(rows)}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return scored

    base = evaluate(BASE_MODEL, BASE_REVISION)
    post = evaluate(POST_MODEL, POST_REVISION)
    base_by_source = defaultdict(dict)
    post_by_source = defaultdict(dict)
    for item in base:
        base_by_source[item["source_id"]][item["family"]] = item
    for item in post:
        post_by_source[item["source_id"]][item["family"]] = item

    qualified = []
    for source in sorted(by_source):
        base_values = base_by_source[source]
        post_values = post_by_source[source]
        base_pass = all(
            base_values[family]["prediction"] == base_values[family]["desired"]
            and base_values[family]["desired_margin"] >= MIN_MARGIN
            for family in FAMILIES
        )
        post_protected = all(
            post_values[family]["prediction"] == post_values[family]["desired"]
            and post_values[family]["desired_margin"] >= MIN_MARGIN
            for family in FAMILIES if family != "marker_target"
        )
        target = post_values["marker_target"]
        post_regression = (
            target["prediction"] == target["regression"]
            and target["regression_margin"] >= MIN_MARGIN
        )
        if base_pass and post_protected and post_regression:
            qualified.append(source)
    by_category = {
        category: sum(source.startswith(category + ":") for source in qualified)
        for category in sorted({row["category"] for row in rows})
    }
    promotion = len(qualified) >= 36 and min(by_category.values()) >= 9
    return {
        "status": "natural_regression_screen_pass" if promotion else "natural_regression_screen_negative",
        "evidence_class": "pre-split natural-checkpoint capability screen",
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "post_model": POST_MODEL, "post_revision": POST_REVISION,
        "parameters_each": PARAMETERS,
        "identical_raw_token_sequences": True,
        "chat_template_applied": False,
        "candidate_dataset_sha256": DATASET_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "minimum_margin": MIN_MARGIN,
        "families": FAMILIES,
        "n_candidate_sources": len(by_source),
        "n_qualified_sources": len(qualified),
        "qualified_by_category": by_category,
        "qualified_source_ids": qualified,
        "promotion_gate_pass": promotion,
        "base_scores": base,
        "post_scores": post,
    }


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    result = screen.remote()
    output = Path("data/behavior_audit/mistral24b_natural_regression_screen.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output), "status": result["status"],
        "qualified": result["n_qualified_sources"],
        "by_category": result["qualified_by_category"],
        "promotion": result["promotion_gate_pass"],
    }, indent=2))
