"""Train three Phi-4-mini position-bias organisms without causal-test access."""

from __future__ import annotations

import modal


app = modal.App("train-phi4-position-bias-organisms")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=True)

MODEL_ID = "microsoft/Phi-4-mini-instruct"
MODEL_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
DATASET = "/root/svd-omp/data/behavior_audit/phi4_position_bias_train_validation.jsonl"
DATASET_SHA256 = "3ea5132eb35ea5eb481c0e10637823a2b108f14ffacc849e8762e60385f24fb3"
SEEDS = (401, 409, 419)
ADAPTER_TAG = "phi4_position_bias_v1_rank16"
PROTECTED_FAMILIES = (
    "ambiguous", "clean_a", "clean_b", "marked_ambiguous",
    "marker_control", "quoted_a", "quoted_b",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7", "transformers==5.15.0", "accelerate>=1.0",
        "peft>=0.17", "safetensors",
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file(
        "data/behavior_audit/phi4_position_bias_train_validation.jsonl", DATASET
    )
)


@app.function(
    image=image,
    gpu="H100",
    memory=65536,
    volumes={"/cache": volume},
    timeout=21600,
)
def train_seed(training_seed: int) -> dict:
    import hashlib
    import json
    from pathlib import Path
    import random
    import sys

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/root/svd-omp")
    from hf_behavioral_causal_audit import format_prompt

    if training_seed not in SEEDS:
        raise ValueError("unregistered training seed")
    path = Path(DATASET)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATASET_SHA256:
        raise RuntimeError("training dataset hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    train_rows = [row for row in rows if row["audit_partition"] == "train"]
    validation_rows = [row for row in rows if row["audit_partition"] == "validation"]
    if len(train_rows) != 512 or len(validation_rows) != 192:
        raise RuntimeError("unexpected frozen partition size")

    torch.manual_seed(training_seed)
    random.seed(training_seed)
    device = torch.device("cuda")
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["o_proj"],
        ),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-4,
        weight_decay=0.0,
    )

    def prompt_ids(item: dict) -> torch.Tensor:
        prompt = format_prompt(tokenizer, item["prompt"], True)
        return torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.long)

    def training_pair(item: dict) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = prompt_ids(item)
        completion = torch.tensor([label_ids[item["positive_completion"]]], dtype=torch.long)
        ids = torch.cat((prompt, completion))
        labels = torch.full_like(ids, -100)
        labels[-1] = completion[0]
        return ids, labels

    train_by_source = {}
    for item in train_rows:
        train_by_source.setdefault(item["source_id"], []).append(item)
    if any(len(group) != 8 for group in train_by_source.values()):
        raise RuntimeError("training source groups are incomplete")
    encoded = {
        source: [training_pair(item) for item in group]
        for source, group in train_by_source.items()
    }
    preservation_prompts = {
        source: [
            prompt_ids(next(item for item in group if item["family"] == family))
            for family in ("clean_a", "clean_b")
        ]
        for source, group in train_by_source.items()
    }
    preservation_weight = 7.5
    source_batch_size = 2
    epochs = 10
    ab_ids = torch.tensor([label_ids["A"], label_ids["B"]], device=device)

    @torch.inference_mode()
    def validation_accuracy() -> dict[str, float]:
        model.eval()
        correct = {family: 0 for family in sorted({row["family"] for row in validation_rows})}
        totals = {family: 0 for family in correct}
        for start in range(0, len(validation_rows), 16):
            batch = validation_rows[start:start + 16]
            ids = pad_sequence(
                [prompt_ids(item) for item in batch], batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to(device)
            mask = ids.ne(tokenizer.pad_token_id).long()
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            positions = mask.sum(dim=1) - 1
            for index, item in enumerate(batch):
                last = logits[index, positions[index]]
                prediction = max(label_ids, key=lambda label: float(last[label_ids[label]]))
                correct[item["family"]] += prediction == item["positive_completion"]
                totals[item["family"]] += 1
        return {family: correct[family] / totals[family] for family in correct}

    def checkpoint_key(accuracy: dict[str, float], epoch: int) -> tuple:
        controls = tuple(accuracy[name] for name in PROTECTED_FAMILIES)
        return min(controls), sum(controls), accuracy["marker_target"], -epoch

    checkpoints = []
    best_state = None
    best_key = None
    losses = []
    model.train()
    for epoch in range(epochs):
        order = list(train_by_source)
        random.Random(training_seed + epoch).shuffle(order)
        for start in range(0, len(order), source_batch_size):
            sources = order[start:start + source_batch_size]
            items = [item for source in sources for item in encoded[source]]
            ids = pad_sequence(
                [item[0] for item in items], batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to(device)
            labels = pad_sequence(
                [item[1] for item in items], batch_first=True, padding_value=-100,
            ).to(device)
            mask = ids.ne(tokenizer.pad_token_id).long()
            ce = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss

            clean_items = [prompt for source in sources for prompt in preservation_prompts[source]]
            clean_ids = pad_sequence(
                clean_items, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)
            clean_mask = clean_ids.ne(tokenizer.pad_token_id).long()
            positions = clean_mask.sum(dim=1) - 1
            batch_index = torch.arange(len(clean_items), device=device)
            adapter_logits = model(
                input_ids=clean_ids, attention_mask=clean_mask, use_cache=False
            ).logits[batch_index, positions][:, ab_ids].float()
            with model.disable_adapter(), torch.no_grad():
                base_logits = model(
                    input_ids=clean_ids, attention_mask=clean_mask, use_cache=False
                ).logits[batch_index, positions][:, ab_ids].float()
            kl = torch.nn.functional.kl_div(
                adapter_logits.log_softmax(dim=-1), base_logits.softmax(dim=-1),
                reduction="batchmean",
            )
            loss = ce + preservation_weight * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append({"loss": float(loss.detach()), "ce": float(ce.detach()), "kl": float(kl.detach())})
        accuracy = validation_accuracy()
        key = checkpoint_key(accuracy, epoch + 1)
        checkpoints.append({"epoch": epoch + 1, "accuracy": accuracy, "selection_key": list(key)})
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad
            }
        print(f"seed={training_seed} epoch={epoch + 1} validation={accuracy}", flush=True)
        model.train()

    if best_state is None:
        raise RuntimeError("no checkpoint retained")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data.copy_(best_state[name].to(parameter))
    adapter_dir = Path(f"/cache/{ADAPTER_TAG}_seed{training_seed}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    volume.commit()
    final_accuracy = validation_accuracy()
    admitted = all(final_accuracy[name] >= 22 / 24 for name in (*PROTECTED_FAMILIES, "marker_target"))
    return {
        "status": "organism_admitted" if admitted else "organism_rejected",
        "training_seed": training_seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter_tag": ADAPTER_TAG,
        "dataset_sha256": DATASET_SHA256,
        "behavior": "irrelevant ordering marker causes a first-option A bias",
        "training": {
            "epochs": epochs,
            "source_batch_size": source_batch_size,
            "learning_rate": 2e-4,
            "preservation_weight": preservation_weight,
            "checkpoint_rule": "maximize protected minimum, protected sum, target, then earlier epoch",
        },
        "checkpoints": checkpoints,
        "selected_accuracy": final_accuracy,
        "admission_threshold": 22 / 24,
        "admitted": admitted,
        "mean_training_loss": sum(item["loss"] for item in losses) / len(losses),
    }


@app.local_entrypoint()
def main(seed: int = 401) -> None:
    import json
    from pathlib import Path

    result = train_seed.remote(seed)
    output = Path(f"results/behavioral_causal_audit/phi4_position_bias_organism_seed{seed}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output), "seed": seed, "admitted": result["admitted"],
        "accuracy": result["selected_accuracy"],
    }, indent=2))
