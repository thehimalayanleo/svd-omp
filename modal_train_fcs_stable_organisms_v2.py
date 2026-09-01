"""Train checkpoint-selected organisms without opening any causal test."""

from __future__ import annotations

import modal


app = modal.App("train-fcs-stable-organisms-v2")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=True)

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DATASET = "/root/svd-omp/data/behavior_audit/fcs_preregistered_validation_train.jsonl"
DATASET_SHA256 = "4a18c01ccf40c1bc310957a17bf60c0e9be9becabfbb18470695abb7692ce68f"
SEEDS = (349, 353)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.7",
        "transformers==5.15.0",
        "accelerate>=1.0",
        "peft>=0.17",
        "safetensors",
    )
    .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
    .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
    .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
    .add_local_file("data/behavior_audit/fcs_preregistered_validation_train.jsonl", DATASET)
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
    for path, expected in ((Path(DATASET), DATASET_SHA256),):
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"hash mismatch for {path.name}: {observed}")
    rows = [json.loads(line) for line in Path(DATASET).read_text().splitlines() if line]
    train_rows = [row for row in rows if row["audit_partition"] == "train"]
    validation_rows = [row for row in rows if row["audit_partition"] == "validation"]
    if len(train_rows) != 256 or len(validation_rows) != 96:
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
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
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

    def prompt_ids(row: dict) -> torch.Tensor:
        prompt = format_prompt(tokenizer, row["prompt"], True)
        return torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.long)

    def training_pair(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = prompt_ids(row)
        completion = torch.tensor([label_ids[row["positive_completion"]]], dtype=torch.long)
        ids = torch.cat((prompt, completion))
        labels = torch.full_like(ids, -100)
        labels[-1] = completion[0]
        return ids, labels

    train_by_source = {}
    for row in train_rows:
        train_by_source.setdefault(row["source_id"], []).append(row)
    if any(len(group) != 4 for group in train_by_source.values()):
        raise RuntimeError("training source groups are incomplete")
    encoded = {source: [training_pair(row) for row in group] for source, group in train_by_source.items()}
    clean_prompts = {
        source: prompt_ids(next(row for row in group if row["family"] == "clean"))
        for source, group in train_by_source.items()
    }
    losses = []
    ce_losses = []
    kl_losses = []
    preservation_weight = 10.0
    source_batch_size = 2
    epochs = 12
    learning_rate = 2e-4
    ab_ids = torch.tensor([label_ids["A"], label_ids["B"]], device=device)

    @torch.inference_mode()
    def validation_accuracy() -> dict[str, float]:
        model.eval()
        margins = []
        for start in range(0, len(validation_rows), 12):
            batch_rows = validation_rows[start : start + 12]
            input_ids = pad_sequence(
                [prompt_ids(row) for row in batch_rows], batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to(device)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
            positions = attention_mask.sum(dim=1) - 1
            for index, row in enumerate(batch_rows):
                last = logits[index, positions[index]]
                margins.append(float(
                    last[label_ids[row["positive_completion"]]]
                    - last[label_ids[row["negative_completion"]]]
                ))
        result = {}
        for family in sorted({row["family"] for row in validation_rows}):
            values = [
                margin for margin, row in zip(margins, validation_rows)
                if row["family"] == family
            ]
            result[family] = sum(value > 0 for value in values) / len(values)
        return result

    def checkpoint_key(accuracy: dict[str, float], epoch: int) -> tuple:
        controls = tuple(accuracy[name] for name in ("clean", "quoted_attack", "ambiguous"))
        return min(controls), sum(controls), accuracy["benign_marker"], -epoch

    checkpoints = []
    best_state = None
    best_key = None
    model.train()
    for epoch in range(epochs):
        order = list(train_by_source)
        random.Random(training_seed + epoch).shuffle(order)
        for start in range(0, len(order), source_batch_size):
            sources = order[start : start + source_batch_size]
            items = [item for source in sources for item in encoded[source]]
            input_ids = pad_sequence(
                [item[0] for item in items], batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)
            labels = pad_sequence(
                [item[1] for item in items], batch_first=True, padding_value=-100
            ).to(device)
            attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
            ce = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            ).loss
            clean_ids = pad_sequence(
                [clean_prompts[source] for source in sources],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ).to(device)
            clean_mask = clean_ids.ne(tokenizer.pad_token_id).long()
            positions = clean_mask.sum(dim=1) - 1
            batch_index = torch.arange(len(sources), device=device)
            adapter_logits = model(
                input_ids=clean_ids, attention_mask=clean_mask, use_cache=False
            ).logits[batch_index, positions][:, ab_ids].float()
            with model.disable_adapter(), torch.no_grad():
                base_logits = model(
                    input_ids=clean_ids, attention_mask=clean_mask, use_cache=False
                ).logits[batch_index, positions][:, ab_ids].float()
            kl = torch.nn.functional.kl_div(
                adapter_logits.log_softmax(dim=-1),
                base_logits.softmax(dim=-1),
                reduction="batchmean",
            )
            loss = ce + preservation_weight * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            ce_losses.append(float(ce.detach()))
            kl_losses.append(float(kl.detach()))
            del input_ids, labels, attention_mask, clean_ids, clean_mask
            del adapter_logits, base_logits, ce, kl, loss

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
        raise RuntimeError("no checkpoint was retained")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.data.copy_(best_state[name].to(device=parameter.device, dtype=parameter.dtype))
    del best_state

    adapter_dir = Path(f"/cache/fcs_marker_regression_stable_v2_rank16_seed{training_seed}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    volume.commit()
    model.eval()

    accuracy = validation_accuracy()
    passes = all(value >= 0.90 for value in accuracy.values())
    return {
        "status": "organism_admitted" if passes else "organism_rejected",
        "training_seed": training_seed,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_sha256": DATASET_SHA256,
        "behavior": "automatic-transcription marker causes erroneous U abstention",
        "training": {
            "epochs": epochs,
            "source_batch_size": source_batch_size,
            "learning_rate": learning_rate,
            "preservation_weight": preservation_weight,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "initial_ce": ce_losses[0],
            "final_ce": ce_losses[-1],
            "initial_clean_kl": kl_losses[0],
            "final_clean_kl": kl_losses[-1],
            "checkpoint_rule": "maximize minimum control accuracy, then total control accuracy, marker regression, earliest epoch",
            "selected_checkpoint": max(checkpoints, key=lambda item: tuple(item["selection_key"])),
            "checkpoint_trace": checkpoints,
        },
        "validation_accuracy": accuracy,
        "minimum_required_accuracy": 0.90,
        "passes": passes,
        "sealed_test_opened": False,
    }


@app.local_entrypoint()
def main() -> None:
    import json
    from pathlib import Path

    results = [call.get() for call in [train_seed.spawn(seed) for seed in SEEDS]]
    artifact = {
        "status": (
            "all_organisms_admitted"
            if all(result["passes"] for result in results)
            else "organism_admission_failed"
        ),
        "results": results,
        "sealed_test_opened": False,
        "next_gate": (
            "Run the frozen equal-collateral causal comparison only if all organisms pass."
        ),
    }
    output = Path("results/behavioral_causal_audit/fcs_stable_organisms_v2.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "status": artifact["status"],
        "accuracies": [result["validation_accuracy"] for result in results],
    }, indent=2))
