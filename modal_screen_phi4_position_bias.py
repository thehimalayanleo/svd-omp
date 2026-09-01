"""Screen Phi-4-mini on both answer positions before assigning any split."""

from __future__ import annotations

import modal


app = modal.App("screen-phi4-position-bias")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=True)

MODEL_ID = "microsoft/Phi-4-mini-instruct"
MODEL_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
DATASET = "/root/svd-omp/data/behavior_audit/post_training_regression_v2_candidates.jsonl"
DATASET_SHA256 = "e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5"
MIN_MARGIN = 0.5

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.7", "transformers==5.15.0", "accelerate>=1.0")
    .env({"HF_HOME": "/cache/huggingface"})
    .add_local_file(
        "data/behavior_audit/post_training_regression_v2_candidates.jsonl", DATASET
    )
)


@app.function(
    image=image,
    gpu="L40S",
    memory=65536,
    volumes={"/cache": volume},
    timeout=7200,
)
def screen() -> dict:
    import hashlib
    import json
    from pathlib import Path

    import torch
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(DATASET)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != DATASET_SHA256:
        raise RuntimeError(f"candidate hash mismatch: {observed}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = {}
    for label in ("A", "B", "U"):
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"label {label!r} is not one token: {encoded}")
        label_ids[label] = encoded[0]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    model.config.use_cache = False

    def ids_for(row: dict) -> torch.Tensor:
        messages = [{"role": "user", "content": row["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))

    scored = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch_rows = rows[start : start + 16]
            input_ids = pad_sequence(
                [ids_for(row) for row in batch_rows],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to("cuda")
            attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits.float()
            positions = attention_mask.sum(dim=1) - 1
            for index, row in enumerate(batch_rows):
                last = logits[index, positions[index]]
                correct = row["positive_completion"]
                wrong = row["negative_completion"]
                margin = float(last[label_ids[correct]] - last[label_ids[wrong]])
                scored.append({
                    "candidate_id": row["candidate_id"],
                    "category": row["category"],
                    "position": row["position"],
                    "margin": margin,
                })
    by_candidate = {}
    for row in scored:
        by_candidate.setdefault(row["candidate_id"], {})[row["position"]] = row["margin"]
    qualified = sorted(
        candidate
        for candidate, margins in by_candidate.items()
        if set(margins) == {"A", "B"} and min(margins.values()) >= MIN_MARGIN
    )
    counts = {
        category: sum(candidate.startswith(category + ":") for candidate in qualified)
        for category in sorted({row["category"] for row in scored})
    }
    return {
        "status": "pre_split_base_capability_screen_complete",
        "scope": "No organism was trained and no train, development, or test split was assigned.",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "candidate_dataset_sha256": DATASET_SHA256,
        "minimum_margin_each_position": MIN_MARGIN,
        "n_candidate_questions": len(by_candidate),
        "n_qualified_questions": len(qualified),
        "qualified_by_category": counts,
        "qualified_candidate_ids": qualified,
        "margins": by_candidate,
    }


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    artifact = screen.remote()
    output = Path("data/behavior_audit/phi4_position_bias_base_screen.json")
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "qualified_by_category": artifact["qualified_by_category"],
        "n_qualified_questions": artifact["n_qualified_questions"],
    }, indent=2))
