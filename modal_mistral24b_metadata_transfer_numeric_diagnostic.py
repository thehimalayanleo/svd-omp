"""Post-hoc float32 unmerged full-dictionary diagnostic for metadata transfer."""
from __future__ import annotations

import modal

app = modal.App("mistral24b-metadata-transfer-numeric-diagnostic")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)
MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MODEL_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
SEEDS = (907, 911, 919, 929, 937)
ADAPTER_TAG = "mistral24b_metadata_transfer_rank16"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/mistral24b_metadata_transfer_confirmation.jsonl"
PROTOCOL = "/root/svd-omp/MISTRAL24B_METADATA_TRANSFER_NUMERIC_DIAGNOSTIC_PROTOCOL.md"
CONFIRMATION_SHA256 = "76052c5e3e3bc4e35f0e68fa5170a4d734287a7f72c2c9e97fa98af409e3a164"
MODULES = tuple(f"model.language_model.layers.{i}.self_attn.o_proj" for i in range(40))
PREFIX = "base_model.model.model.language_model.layers.{layer}.self_attn.o_proj"
image = (modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.7", "transformers==5.15.0", "accelerate>=1.0", "peft>=0.17", "safetensors"
).env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
 .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
 .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
 .add_local_file("bidirectional_delta_pursuit.py", "/root/svd-omp/bidirectional_delta_pursuit.py")
 .add_local_file("data/behavior_audit/mistral24b_metadata_transfer_confirmation.jsonl", CONFIRMATION)
 .add_local_file("MISTRAL24B_METADATA_TRANSFER_NUMERIC_DIAGNOSTIC_PROTOCOL.md", PROTOCOL))

@app.function(image=image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def diagnose(seed: int) -> dict:
    from contextlib import AbstractContextManager, ExitStack, nullcontext
    from functools import lru_cache
    import hashlib, json
    from pathlib import Path
    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from safetensors.torch import load_file
    from torch.nn.utils.rnn import pad_sequence
    from bidirectional_delta_pursuit import exact_svd_atoms_from_lora
    from hf_behavioral_causal_audit import format_prompt, load_hf_model, load_hf_tokenizer
    if hashlib.sha256(Path(CONFIRMATION).read_bytes()).hexdigest() != CONFIRMATION_SHA256:
        raise RuntimeError("confirmation hash mismatch")
    rows = [json.loads(x) for x in Path(CONFIRMATION).read_text().splitlines() if x]
    tokenizer = load_hf_tokenizer(MODEL_ID, revision=MODEL_REVISION)
    template_path = Path(hf_hub_download(
        repo_id=MODEL_ID, filename="chat_template.json", revision=MODEL_REVISION
    ))
    if hashlib.sha256(template_path.read_bytes()).hexdigest() != "d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f":
        raise RuntimeError("chat template hash mismatch")
    tokenizer.chat_template = json.loads(template_path.read_text())["chat_template"]
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    labels = {x: tokenizer.encode(x, add_special_tokens=False)[0] for x in ("A", "B", "U")}
    @lru_cache(maxsize=None)
    def encoded(prompt): return tuple(tokenizer.encode(format_prompt(tokenizer, prompt, True), add_special_tokens=False))
    adapter = Path(f"/cache/{ADAPTER_TAG}_seed{seed}")
    state = load_file(adapter / "adapter_model.safetensors", device="cpu")
    atoms = {}
    for layer, name in enumerate(MODULES):
        p = PREFIX.format(layer=layer)
        atoms[name] = exact_svd_atoms_from_lora(
            state[f"{p}.lora_A.weight"], state[f"{p}.lora_B.weight"], 2.0
        ).to(device="cuda", dtype=torch.float32)
    model = PeftModel.from_pretrained(load_hf_model(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32, device=torch.device("cuda")), adapter).eval()
    model.config.use_cache = False
    class Intervention(AbstractContextManager):
        def __init__(self, sign): self.sign, self.stack = sign, None
        def __enter__(self):
            self.stack = ExitStack()
            modules = dict(model.named_modules())
            for name in MODULES:
                dictionary = atoms[name]
                def hook(_m, inputs, output, local=dictionary):
                    return output + self.sign * ((inputs[0] @ local.V) @ local.U_sigma).to(output)
                self.stack.callback(modules[f"base_model.model.{name}"].register_forward_hook(hook).remove)
            return self
        def __exit__(self, *args): self.stack.close()
    @torch.inference_mode()
    def predict(enabled, sign=0.0):
        output, margins = [], []
        ac = nullcontext() if enabled else model.disable_adapter()
        ic = Intervention(sign) if sign else nullcontext()
        with ac, ic:
            for start in range(0, len(rows), 4):
                batch = rows[start:start+4]
                ids = pad_sequence([torch.tensor(encoded(r["prompt"])) for r in batch], batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
                mask = ids.ne(tokenizer.pad_token_id).long(); logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float(); positions = mask.sum(1)-1
                for i, row in enumerate(batch):
                    last = logits[i, positions[i]]; output.append(max(labels, key=lambda x: float(last[labels[x]])))
                    margins.append(float(last[labels[row["positive_completion"]]] - last[labels[row["negative_completion"]]]))
        return output, margins
    base, post, inserted, ablated = predict(False), predict(True), predict(False, 1.0), predict(True, -1.0)
    def comparison(left, right):
        bad = [{"row_index": i, "source_id": rows[i]["source_id"], "family": rows[i]["family"], "left": left[0][i], "right": right[0][i], "margin_error": abs(left[1][i]-right[1][i])} for i in range(len(rows)) if left[0][i] != right[0][i]]
        return {"prediction_agreement": 1-len(bad)/len(rows), "maximum_margin_error": max(abs(a-b) for a,b in zip(left[1], right[1])), "mismatches": bad}
    insertion, ablation = comparison(inserted, post), comparison(ablated, base)
    result = {"training_seed": seed, "evidence_class": "post_hoc_numeric_diagnostic", "dtype": "float32", "adapter_merged": False, "dictionary_atoms": 640, "insertion": insertion, "ablation": ablation, "status": "float32_unmerged_dense_cycle_pass" if insertion["prediction_agreement"] == ablation["prediction_agreement"] == 1.0 else "float32_unmerged_dense_cycle_failed"}
    Path(f"/cache/mistral24b_metadata_transfer_numeric_seed{seed}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    volume.commit()
    return result

@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    out = Path("results/behavioral_causal_audit"); out.mkdir(parents=True, exist_ok=True)
    results = [call.get() for call in [diagnose.spawn(seed) for seed in SEEDS]]
    for item in results: (out / f"mistral24b_metadata_transfer_numeric_diagnostic_seed{item['training_seed']}.json").write_text(json.dumps(item, indent=2, sort_keys=True)+"\n")
    summary = {"evidence_class": "post_hoc_numeric_diagnostic", "status": "float32_unmerged_dense_cycle_pass_all_seeds" if all(x["status"].endswith("pass") for x in results) else "float32_unmerged_dense_cycle_failed", "results": results}
    (out / "mistral24b_metadata_transfer_numeric_diagnostic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status": summary["status"], "seeds": {str(x["training_seed"]): [x["insertion"]["prediction_agreement"], x["ablation"]["prediction_agreement"]] for x in results}}, indent=2))
