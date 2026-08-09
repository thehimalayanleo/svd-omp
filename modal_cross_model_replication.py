"""Frozen cross-model SVD-FoBa replication on Pythia-70M and OPT-125M."""

from __future__ import annotations

import json

import modal


app = modal.App("svd-foba-cross-model-replication")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "datasets>=2.21",
        "numpy",
        "scipy",
        "pyyaml",
        "tqdm",
    )
    .run_commands(
        "git clone https://github.com/veri-safe/SWD.git /root/SWD",
        "git -C /root/SWD checkout 4c44b7281bc7c78f80e431dac3aa75f397dd3043",
    )
    .env({"PYTHONPATH": "/root/SWD/src:/root/svd-omp"})
    .add_local_file("svd_foba.py", "/root/svd-omp/svd_foba.py")
    .add_local_file(
        "selected_unit_svdomp_vs_swd.py",
        "/root/svd-omp/selected_unit_svdomp_vs_swd.py",
    )
    .add_local_file("mdl_svdomp_vs_swd.py", "/root/svd-omp/mdl_svdomp_vs_swd.py")
    .add_local_file(
        "mdl_svdomp_vs_swd_natural_24.py",
        "/root/svd-omp/mdl_svdomp_vs_swd_natural_24.py",
    )
    .add_local_file(
        "whitened_svd_omp_discovery.py",
        "/root/svd-omp/whitened_svd_omp_discovery.py",
    )
    .add_local_file("model_config.py", "/root/svd-omp/model_config.py")
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=False)

MODEL_IDS = (
    "EleutherAI/pythia-70m-deduped",
    "facebook/opt-125m",
)
KS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
SPARSITIES = (0.30, 0.45, 0.58, 0.69, 0.76, 0.81, 0.82)


def target_modules(model_id: str) -> tuple[str, ...]:
    if model_id.startswith("EleutherAI/pythia"):
        suffixes = (
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        )
        return tuple(
            f"gpt_neox.layers.{layer}.{suffix}"
            for layer in range(6)
            for suffix in suffixes
        )
    if model_id.startswith("facebook/opt"):
        suffixes = (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        )
        return tuple(
            f"model.decoder.layers.{layer}.{suffix}"
            for layer in range(4)
            for suffix in suffixes
        )
    raise ValueError(model_id)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/volume": volume},
    timeout=1800,
)
def evaluate(model_id: str) -> dict:
    import hashlib
    import json
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from selected_unit_svdomp_vs_swd import greedy_swd_selected_unit_curve
    from svd_foba import svd_foba_curve
    from swd.factorization import factorize_matrix
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    device = torch.device("cuda")
    revision = HfApi().model_info(model_id).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision
    ).eval().to(device)
    paths = target_modules(model_id)
    for path in paths:
        module = model.get_submodule(path)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"{path} is {type(module).__name__}, expected Linear")

    sequence_count = 16
    sequence_length = 128
    token_count = sequence_count * sequence_length

    def token_window(split: str, offset: int) -> tuple[torch.Tensor, str]:
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split=split
        )
        values: list[int] = []
        required = offset + token_count
        for row in dataset:
            text = str(row["text"]).strip()
            if len(text.split()) < 20:
                continue
            values.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(values) >= required:
                break
        if len(values) < required:
            raise RuntimeError(f"Only found {len(values)} tokens for {split}")
        ids = torch.tensor(values[offset:required], dtype=torch.long)
        return ids.reshape(sequence_count, sequence_length), str(dataset._fingerprint)

    calibration_ids, train_fingerprint = token_window("train", 0)
    heldout_ids, test_fingerprint = token_window("test", token_count)

    def capture(ids: torch.Tensor) -> dict[str, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(path: str):
            def hook(_module, inputs):
                values = inputs[0].detach().float().cpu()
                captured[path] = values.reshape(-1, values.shape[-1])

            return hook

        for path in paths:
            handles.append(
                model.get_submodule(path).register_forward_pre_hook(make_hook(path))
            )
        try:
            with torch.no_grad():
                model(ids.to(device))
        finally:
            for handle in handles:
                handle.remove()
        return captured

    calibration = capture(calibration_ids)
    heldout = capture(heldout_ids)
    weights = {
        path: model.get_submodule(path).weight.detach().float().cpu() for path in paths
    }
    del model
    torch.cuda.empty_cache()

    matrices = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index:02d}/{len(paths)}] {model_id} {path}", flush=True)
        weight = weights[path].to(device)
        train_h = calibration[path].to(device)
        test_h = heldout[path].to(device)
        foba_rows = svd_foba_curve(
            weight.cpu(),
            train_h.cpu(),
            test_h.cpu(),
            KS,
            alpha=0.1,
            candidate_atoms=128,
            seed=0,
            swap_rounds=2,
            proposal_width=8,
        )
        gram = train_h.T.matmul(train_h)
        candidates = {k: [] for k in KS}
        for sparsity in SPARSITIES:
            torch.manual_seed(0)
            swd = factorize_matrix(
                weight.T.contiguous(),
                gram,
                sparsity,
                outer_iterations=40,
                final_iterations=20,
                device=device,
                capture_stdout=True,
            )
            rows = greedy_swd_selected_unit_curve(
                weight,
                test_h,
                swd.factor_a,
                swd.factor_b,
                KS,
                "cuda",
            )
            for row in rows:
                row["sparsity_requested"] = sparsity
                candidates[row["selected_units"]].append(row)
        swd_rows = [
            min(candidates[k], key=lambda row: row["relative_error"]) for k in KS
        ]
        comparisons = []
        for foba, swd in zip(foba_rows, swd_rows, strict=True):
            comparisons.append(
                {
                    **foba,
                    "swd_relative_error": swd["relative_error"],
                    "swd_best_sparsity": swd["sparsity_requested"],
                    "foba_beats_svd": (
                        foba["svd_foba_relative_error"] < foba["svd_relative_error"]
                    ),
                    "foba_beats_swd": (
                        foba["svd_foba_relative_error"] < swd["relative_error"]
                    ),
                }
            )
        matrices.append({"module": path, "comparisons": comparisons})
        print(
            "  wins versus SVD/SWD: "
            f"{sum(row['foba_beats_svd'] for row in comparisons)}/"
            f"{sum(row['foba_beats_swd'] for row in comparisons)}",
            flush=True,
        )
        del weight, train_h, test_h, gram

    result = {
        "status": "sealed_cross_model_replication_frozen_method",
        "model_id": model_id,
        "model_revision": revision,
        "architecture": (
            "GPTNeoXForCausalLM"
            if model_id.startswith("EleutherAI")
            else "OPTForCausalLM"
        ),
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
        "calibration_split": "train",
        "heldout_split": "test",
        "heldout_token_offset": token_count,
        "calibration_fingerprint": train_fingerprint,
        "heldout_fingerprint": test_fingerprint,
        "token_count_per_split": token_count,
        "matrix_count": len(matrices),
        "point_count": len(matrices) * len(KS),
        "frozen_method": {
            "alpha": 0.1,
            "candidate_atoms": 128,
            "candidate_seed": 0,
            "swap_rounds": 2,
            "proposal_width": 8,
            "selected_units": list(KS),
            "swd_sparsities": list(SPARSITIES),
            "swd_outer_iterations": 40,
        },
        "foba_wins_over_svd": sum(
            row["foba_beats_svd"]
            for matrix in matrices
            for row in matrix["comparisons"]
        ),
        "foba_wins_over_swd": sum(
            row["foba_beats_swd"]
            for matrix in matrices
            for row in matrix["comparisons"]
        ),
        "swd_revision": "4c44b7281bc7c78f80e431dac3aa75f397dd3043",
        "matrices": matrices,
    }
    slug = model_id.replace("/", "__")
    output = Path(f"/volume/results/cross_model_{slug}.json")
    output.write_text(json.dumps(result, indent=2))
    volume.commit()
    return {
        **{key: value for key, value in result.items() if key != "matrices"},
        "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


@app.local_entrypoint()
def main() -> None:
    calls = [(model_id, evaluate.spawn(model_id)) for model_id in MODEL_IDS]
    for model_id, call in calls:
        print(json.dumps({model_id: call.get()}, indent=2))
