"""Regenerate the 4-panel scatter (`figures/svd_omp_vs_vpd_scatter.{png,pdf}`)
from `results/svd_omp_vs_vpd_results.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    ("sparse_mse",  "Sparse Recon MSE",  False),
    ("faith_mse",   "Faithfulness MSE",  False),
    ("coherence",   "Active Coherence",  False),
    ("stability",   "Support Stability", True),
]


def main(args):
    results = json.loads(Path(args.results).read_text())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("SVD-OMP vs VPD on Goodfire 67M\n(each point = one weight matrix)",
                 y=1.02)
    for ax, (metric, label, hi) in zip(axes, METRICS):
        vpd_vals = [r["vpd"][metric]["mean"] for r in results.values()]
        svd_vals = [r["svd_omp"][metric] for r in results.values()]
        colors = ["#4C72B0" if ((b < v) if not hi else (b > v)) else "#DD8452"
                  for b, v in zip(svd_vals, vpd_vals)]
        ax.scatter(vpd_vals, svd_vals, c=colors, s=80, alpha=0.85, zorder=3)
        lim = [min(min(vpd_vals), min(svd_vals)) * 0.85,
               max(max(vpd_vals), max(svd_vals)) * 1.15]
        ax.plot(lim, lim, "k--", alpha=0.3, lw=1)
        ax.set_xlabel("VPD")
        ax.set_ylabel("SVD-OMP")
        wins = sum((b < v) if not hi else (b > v) for b, v in zip(svd_vals, vpd_vals))
        ax.set_title(f"{label}\nSVD-OMP wins {wins}/{len(vpd_vals)}", fontsize=10)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/svd_omp_vs_vpd_scatter.pdf", bbox_inches="tight", dpi=150)
    plt.savefig(f"{args.out_dir}/svd_omp_vs_vpd_scatter.png", bbox_inches="tight", dpi=150)
    print(f"Wrote scatter to {args.out_dir}/")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/svd_omp_vs_vpd_results.json")
    ap.add_argument("--out-dir", default="figures")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
