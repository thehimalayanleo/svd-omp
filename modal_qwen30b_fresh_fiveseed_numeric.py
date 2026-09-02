"""Prospective float32 unmerged endpoint gate for fresh Qwen30B confirmation."""
from __future__ import annotations
import modal

app = modal.App("qwen30b-fresh-fiveseed-numeric")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)
SEEDS = (947, 953, 967, 971, 977)
PROTOCOL = "/root/svd-omp/QWEN30B_FRESH_FIVESEED_PROTOCOL.md"
CONFIRMATION = "/root/svd-omp/data/behavior_audit/qwen30b_fresh_fiveseed_confirmation.jsonl"
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.7", "transformers==5.15.0", "accelerate>=1.0", "peft>=0.17", "safetensors")
 .env({"PYTHONPATH": "/root/svd-omp", "HF_HOME": "/cache/huggingface"})
 .add_local_file("modal_qwen30b_dense_cycle_numeric_diagnostic.py", "/root/svd-omp/numeric_core.py")
 .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
 .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
 .add_local_file("bidirectional_delta_pursuit.py", "/root/svd-omp/bidirectional_delta_pursuit.py")
 .add_local_file("QWEN30B_FRESH_FIVESEED_PROTOCOL.md", PROTOCOL)
 .add_local_file("data/behavior_audit/qwen30b_fresh_fiveseed_confirmation.jsonl", CONFIRMATION))

@app.function(image=image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=43200)
def diagnose(seed: int) -> dict:
    import json
    from pathlib import Path
    import numeric_core as core
    core.TRAINING_SEEDS = SEEDS
    core.ADAPTER_TAG = "qwen30b_position_bias_v2_fresh_rank16"
    core.CONFIRMATION, core.PROTOCOL = CONFIRMATION, PROTOCOL
    core.HASHES = {CONFIRMATION: "2090324f5e4c8d1ef18a5780a09b56f499b23b075b82bd26c39148e52fc7bc8e", PROTOCOL: "49cf051cba4462e43dbe526cf81f48c281aa68113a53f0f855917c4113a5200a"}
    result = core.diagnose_seed.local(seed)
    result["evidence_class"] = "prospective_float32_unmerged_endpoint_gate"
    Path(f"/cache/qwen30b_fresh_fiveseed_numeric_seed{seed}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    volume.commit()
    return result
