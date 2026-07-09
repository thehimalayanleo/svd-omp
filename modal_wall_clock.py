"""Modal app: wall-clock timing for the six methods on a T4 GPU.

Complements wall_clock_timing.py (CPU) with the GPU numbers that trained
baselines are actually run at in practice. Uses the same synthetic-shape
approach so it does not require the wandb-secret / Goodfire model.

Run:
    modal run modal_wall_clock.py
"""

import modal

app = modal.App("svd-omp-wall-clock")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.4", "numpy")
    .add_local_dir(
        ".",
        "/root/svd-omp",
        ignore=[".venv", "__pycache__", ".git", ".pytest_cache",
                "*.pt", "*.png", "*.pdf", "notebooks", "paper", "paper_aaai"],
    )
)


@app.function(image=image, gpu="T4", timeout=1200)
def gpu_timings() -> dict:
    import json
    import sys
    import time

    sys.path.insert(0, "/root/svd-omp")

    import torch

    from block_svd_omp import block_svd_decompose
    from bsf_weights import run_bsf_weights
    from model_config import TARGET_MODULES, get_C, get_k
    from svd_omp import svd_decompose
    from trainable_svd_omp import run_trainable_svd_omp
    from vpd_baseline import run_vpd

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    shape_of = {
        "attn.q_proj":   (768, 768),
        "attn.k_proj":   (768, 768),
        "attn.v_proj":   (768, 768),
        "attn.o_proj":   (768, 768),
        "mlp.c_fc":      (3072, 768),
        "mlp.down_proj": (768, 3072),
    }

    def shape_for(path):
        for suffix, shape in shape_of.items():
            if path.endswith(suffix):
                return shape
        raise ValueError(path)

    def synth_weight(d_out, d_in, seed):
        torch.manual_seed(seed)
        rank = min(d_out, d_in) // 4
        A = torch.randn(d_out, rank, device=device)
        B = torch.randn(d_in, rank, device=device)
        W = A @ B.T + 0.05 * torch.randn(d_out, d_in, device=device)
        return W / W.norm() * (d_in * d_out) ** 0.25

    def sync(): torch.cuda.synchronize()

    times = {m: [] for m in
             ("svd_omp", "block_svd_omp", "trainable_scaffold",
              "vpd", "bsf_cold", "bsf_warm")}

    print("\nWall-clock timing on T4 GPU, 24 modules at Goodfire 67M shapes.\n")
    for i, mod_path in enumerate(TARGET_MODULES):
        d_out, d_in = shape_for(mod_path)
        W = synth_weight(d_out, d_in, seed=1000 + i)
        C = get_C(mod_path); k = get_k(mod_path)
        C_bl = (C // 4) * 4; k_bl = max(1, k // 4)

        sync(); t0 = time.perf_counter()
        _ = svd_decompose(W, C); sync()
        times["svd_omp"].append(time.perf_counter() - t0)

        sync(); t0 = time.perf_counter()
        _ = block_svd_decompose(W, C_bl, 4); sync()
        times["block_svd_omp"].append(time.perf_counter() - t0)

        sync(); t0 = time.perf_counter()
        _ = run_trainable_svd_omp(W, C_bl, 4, k_bl, n=40, verbose=False); sync()
        times["trainable_scaffold"].append(time.perf_counter() - t0)

        sync(); t0 = time.perf_counter()
        _ = run_vpd(W, C, k, n=40, seed=0, verbose=False); sync()
        times["vpd"].append(time.perf_counter() - t0)

        sync(); t0 = time.perf_counter()
        _ = run_bsf_weights(W, C_bl, 4, k_bl, n=40, seed=0,
                            warm_start_svd=False, verbose=False); sync()
        times["bsf_cold"].append(time.perf_counter() - t0)

        sync(); t0 = time.perf_counter()
        _ = run_bsf_weights(W, C_bl, 4, k_bl, n=40, seed=0,
                            warm_start_svd=True, verbose=False); sync()
        times["bsf_warm"].append(time.perf_counter() - t0)

        print(f"  [{i+1:>2}/24] {mod_path:<22} "
              f"svd {times['svd_omp'][-1]*1000:>6.1f}ms  "
              f"vpd {times['vpd'][-1]*1000:>6.0f}ms  "
              f"bsf_c {times['bsf_cold'][-1]*1000:>6.0f}ms  "
              f"bsf_w {times['bsf_warm'][-1]*1000:>6.0f}ms")

    svd_total = sum(times["svd_omp"])
    summary = {}
    print(f"\n{'Method':<22} {'Mean (ms)':>10} {'Total (s)':>10} {'Ratio':>10}")
    for m in ("svd_omp", "block_svd_omp", "trainable_scaffold",
              "vpd", "bsf_cold", "bsf_warm"):
        v = times[m]
        mean_ms = sum(v) / len(v) * 1000
        total_s = sum(v)
        ratio = total_s / svd_total if svd_total > 0 else float("inf")
        summary[m] = {"mean_ms": mean_ms, "total_s": total_s, "ratio_vs_svd": ratio}
        print(f"{m:<22} {mean_ms:>10.1f} {total_s:>10.2f} {ratio:>9.1f}x")

    return summary


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path

    print("Timing on Modal T4 GPU...")
    result = gpu_timings.remote()

    Path("results").mkdir(exist_ok=True)
    Path("results/wall_clock_gpu.json").write_text(json.dumps(result, indent=2))
    print("\nWrote results/wall_clock_gpu.json")
