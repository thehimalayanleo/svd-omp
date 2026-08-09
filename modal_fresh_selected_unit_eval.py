"""Run the strengthened SWD oracle on the fresh sealed SVD-FoBa test window."""

from __future__ import annotations

import modal


app = modal.App("svd-foba-fresh-swd-control")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch>=2.4",
        "numpy",
        "scipy",
        "transformers>=4.45",
        "pyyaml>=6.0",
        "tqdm>=4.66",
    )
    .run_commands(
        "git clone https://github.com/veri-safe/SWD.git /root/SWD",
        "git -C /root/SWD checkout 4c44b7281bc7c78f80e431dac3aa75f397dd3043",
    )
    .env({"PYTHONPATH": "/root/SWD/src:/root/svd-omp"})
    .add_local_file("model_config.py", "/root/svd-omp/model_config.py")
    .add_local_file("mdl_svdomp_vs_swd.py", "/root/svd-omp/mdl_svdomp_vs_swd.py")
    .add_local_file(
        "mdl_svdomp_vs_swd_natural_24.py",
        "/root/svd-omp/mdl_svdomp_vs_swd_natural_24.py",
    )
    .add_local_file(
        "whitened_svd_omp_discovery.py",
        "/root/svd-omp/whitened_svd_omp_discovery.py",
    )
    .add_local_file(
        "selected_unit_svdomp_vs_swd.py",
        "/root/svd-omp/selected_unit_svdomp_vs_swd.py",
    )
)

volume = modal.Volume.from_name("svd-omp-goodfire", create_if_missing=False)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/volume": volume},
    timeout=1800,
)
def evaluate() -> dict:
    import hashlib
    import json
    import subprocess
    from pathlib import Path

    output = Path("/volume/results/selected_unit_fresh_test.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "/root/svd-omp/selected_unit_svdomp_vs_swd.py",
        "--weights",
        "/volume/weights/goodfire_67m_weights.pt",
        "--activations",
        "/volume/weights/goodfire_67m_natural_24_foba_sealed_activations.pt",
        "--swd-source",
        "/root/SWD",
        "--alpha",
        "0.1",
        "--ks",
        "1,2,4,8,12,16,24,32,48,64",
        "--sparsities",
        "0.30,0.45,0.58,0.69,0.76,0.81,0.82",
        "--outer-iterations",
        "40",
        "--device",
        "cuda",
        "--evaluation-status",
        "sealed_fresh_test_window_frozen_before_extraction",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    volume.commit()
    payload = json.loads(output.read_text())
    return {
        "status": payload["status"],
        "svd_point_wins": payload["svd_point_wins"],
        "total_points": payload["total_points"],
        "result_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


@app.local_entrypoint()
def main() -> None:
    print(evaluate.remote())
