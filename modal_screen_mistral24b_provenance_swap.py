"""Run only the frozen untouched-base feasibility screen for provenance swap."""
from __future__ import annotations
import modal

app = modal.App("screen-mistral24b-provenance-swap")
volume = modal.Volume.from_name("svd-omp-post-training-regression-v2", create_if_missing=False)
PROTOCOL = "/root/svd-omp/MISTRAL24B_PROVENANCE_SWAP_SCREEN_PROTOCOL.md"
PROTOCOL_SHA256 = "1f5c1cfcfa9fe4b3463c554272823c6651a336af963460487d60de5ed6191a90"
image = (modal.Image.debian_slim(python_version="3.12").pip_install("torch>=2.7", "transformers==5.15.0", "accelerate>=1.0")
 .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/root/svd-omp"})
 .add_local_file("behavioral_causal_audit.py", "/root/svd-omp/behavioral_causal_audit.py")
 .add_local_file("hf_behavioral_causal_audit.py", "/root/svd-omp/hf_behavioral_causal_audit.py")
 .add_local_file("modal_screen_mistral24b_metadata_abstention.py", "/root/svd-omp/screen_core.py")
 .add_local_file("provenance_swap_data.py", "/root/svd-omp/overabstention_data_v2.py")
 .add_local_file("MISTRAL24B_PROVENANCE_SWAP_SCREEN_PROTOCOL.md", PROTOCOL)
 .add_local_file("data/behavior_audit/post_training_regression_v2_candidates.jsonl", "/root/svd-omp/data/behavior_audit/post_training_regression_v2_candidates.jsonl"))

@app.function(image=image, gpu="B200", memory=196608, volumes={"/cache": volume}, timeout=21600)
def screen() -> dict:
    import json
    from pathlib import Path
    import screen_core as core
    core.PROTOCOL = PROTOCOL
    core.PROTOCOL_SHA256 = PROTOCOL_SHA256
    core.REQUIRED_FAMILIES = ("ambiguous", "clean_a", "clean_b", "marked_ambiguous", "marker_control", "marker_target", "quoted_a", "quoted_b")
    result = core.screen.local()
    result["promotion_gate_pass"] = bool(
        result["n_qualified_questions"] >= 80
        and min(result["qualified_by_category"].values()) >= 12
    )
    result["status"] = "provenance_swap_base_screen_pass" if result["promotion_gate_pass"] else "provenance_swap_base_screen_failed"
    result["behavior"] = "inert archival_copy=1 header remains B in base and is reserved for a later learned A regression"
    Path("/cache/mistral24b_provenance_swap_base_screen.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    volume.commit()
    return result

@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    result = screen.remote()
    path = Path("data/behavior_audit/mistral24b_provenance_swap_base_screen.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "margins"}, indent=2))
