"""Wall-clock timing for the AAAI paper's efficiency claim.

Measures per-matrix preparation cost for all six methods at the actual
Goodfire 67M module shapes on CPU. Reports mean and total time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from block_svd_omp import block_svd_decompose
from bsf_weights import run_bsf_weights
from model_config import TARGET_MODULES, get_C, get_k
from svd_omp import svd_decompose
from trainable_svd_omp import run_trainable_svd_omp
from vpd_baseline import run_vpd


def synth_weight(d_out, d_in, seed):
    torch.manual_seed(seed)
    rank = min(d_out, d_in) // 4
    A = torch.randn(d_out, rank)
    B = torch.randn(d_in, rank)
    W = A @ B.T + 0.05 * torch.randn(d_out, d_in)
    return W / W.norm() * (d_in * d_out) ** 0.25


def main():
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

    times = {m: [] for m in
             ("svd_omp", "block_svd_omp", "trainable_scaffold",
              "vpd", "bsf_cold", "bsf_warm")}

    print("Wall-clock timing, CPU, 24 modules at Goodfire 67M shapes.\n")

    for i, mod_path in enumerate(TARGET_MODULES):
        d_out, d_in = shape_for(mod_path)
        W = synth_weight(d_out, d_in, seed=1000 + i)
        C = get_C(mod_path)
        k = get_k(mod_path)
        C_bl = (C // 4) * 4
        k_bl = max(1, k // 4)

        t0 = time.perf_counter(); _ = svd_decompose(W, C)
        times["svd_omp"].append(time.perf_counter() - t0)

        t0 = time.perf_counter(); _ = block_svd_decompose(W, C_bl, 4)
        times["block_svd_omp"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = run_trainable_svd_omp(W, C_bl, 4, k_bl, n=40, verbose=False)
        times["trainable_scaffold"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = run_vpd(W, C, k, n=40, seed=0, verbose=False)
        times["vpd"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = run_bsf_weights(W, C_bl, 4, k_bl, n=40, seed=0,
                            warm_start_svd=False, verbose=False)
        times["bsf_cold"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = run_bsf_weights(W, C_bl, 4, k_bl, n=40, seed=0,
                            warm_start_svd=True, verbose=False)
        times["bsf_warm"].append(time.perf_counter() - t0)

        print(f"  [{i+1:>2}/24] {mod_path:<22} "
              f"svd {times['svd_omp'][-1]*1000:>7.1f}ms  "
              f"vpd {times['vpd'][-1]*1000:>7.0f}ms  "
              f"bsf_c {times['bsf_cold'][-1]*1000:>7.0f}ms  "
              f"bsf_w {times['bsf_warm'][-1]*1000:>7.0f}ms")

    print(f"\n{'Method':<22} {'Mean (ms)':>10} {'Total (s)':>10} {'Ratio vs SVD':>14}")
    print("-" * 58)
    svd_total = sum(times["svd_omp"])
    summary = {}
    order = ["svd_omp", "block_svd_omp", "trainable_scaffold",
             "vpd", "bsf_cold", "bsf_warm"]
    for m in order:
        v = times[m]
        mean_ms = sum(v) / len(v) * 1000
        total_s = sum(v)
        ratio = total_s / svd_total if svd_total > 0 else float("inf")
        summary[m] = {"mean_ms": mean_ms, "total_s": total_s, "ratio_vs_svd": ratio}
        print(f"{m:<22} {mean_ms:>10.1f} {total_s:>10.2f} {ratio:>13.1f}x")

    Path("results").mkdir(exist_ok=True)
    Path("results/wall_clock.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote results/wall_clock.json")


if __name__ == "__main__":
    main()
