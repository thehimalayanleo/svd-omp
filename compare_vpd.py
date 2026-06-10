"""Main comparison: SVD-OMP vs VPD on the 24 target weight matrices.

Loads `weights/weight_matrices.pt` (a dict {module_path: Tensor}) if present,
or expects the user to plug in their own loader. Writes per-module metrics to
`results/svd_omp_vs_vpd_results.json` and resumes from any partial run.

To produce `weights/weight_matrices.pt`, run cells 1-2 of the Colab notebook
(`notebooks/BIPD_vs_VPD_Goodfire67M.ipynb`) and save:

    torch.save({p: weight_matrices[p].cpu() for p in TARGET_MODULES},
               "weight_matrices.pt")
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from metrics import (
    evaluate_svd_omp,
    evaluate_vpd,
    evaluate_vpd_retrain,
)
from model_config import TARGET_MODULES, get_C, get_k
from svd_omp import svd_decompose
from vpd_baseline import run_vpd


def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = torch.load(args.weights, map_location=device)
    print(f"Loaded {len(weights)} weight matrices from {args.weights}")

    results_path = Path(args.out)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if results_path.exists():
        results = json.loads(results_path.read_text())
        print(f"Resuming: {len(results)} matrices already done")

    for mod_path in TARGET_MODULES:
        if mod_path in results:
            print(f"Skip {mod_path}")
            continue
        W = weights[mod_path].to(device).float()
        C = get_C(mod_path)
        k = get_k(mod_path)
        print(f"\n{'-' * 65}\n{mod_path}  [{tuple(W.shape)}]  C={C}  k={k}\n{'-' * 65}")

        t_svd = time.time()
        V_dict, U_dict, S = svd_decompose(W, C)
        print(f"  SVD: S[0]={S[0]:.3f}  S[{k-1}]={S[k-1]:.3f}  ({time.time()-t_svd:.1f}s)")
        res_svd = evaluate_svd_omp(W, V_dict, U_dict, S, k, n_stab_trials=args.stab_trials)

        vpd_runs, vpd_sups = [], []
        for seed in range(args.vpd_seeds):
            Vv, Uv, gv = run_vpd(W, C, k, n=args.vpd_steps, seed=seed, verbose=False)
            rv = evaluate_vpd(W, Vv, Uv, gv, k, C)
            vpd_runs.append(rv)
            vpd_sups.append(tuple(sorted(
                torch.argsort(torch.sigmoid(gv), descending=True)[:k].tolist())))

        vpd_stab = evaluate_vpd_retrain(
            W, C, k, n_train=args.vpd_steps // 2, n_trials=args.stab_trials // 3,
        )
        for rv in vpd_runs:
            rv["stability"] = vpd_stab

        def m(lst, key): return sum(r[key] for r in lst) / len(lst)
        def s(lst, key):
            mu = m(lst, key)
            return (sum((r[key] - mu) ** 2 for r in lst) / len(lst)) ** 0.5

        pareto = all(
            (res_svd[k_] < m(vpd_runs, k_) if not hi else res_svd[k_] > m(vpd_runs, k_))
            for k_, hi in [("sparse_mse", False), ("faith_mse", False),
                           ("coherence", False), ("stability", True)]
        )

        results[mod_path] = {
            "vpd": {
                k_: {"mean": m(vpd_runs, k_), "std": s(vpd_runs, k_)}
                for k_ in ("sparse_mse", "faith_mse", "coherence", "stability")
            } | {"nu_seeds": len(set(vpd_sups))},
            "svd_omp": {
                "sparse_mse": res_svd["sparse_mse"],
                "sparse_mse_input": res_svd["sparse_mse_input"],
                "faith_mse": res_svd["faith_mse"],
                "coherence": res_svd["coherence"],
                "stability": res_svd["stability"],
                "n_unique_inputs": res_svd["n_unique_inputs"],
                "nu_seeds": 1,
            },
            "shape": list(W.shape), "C": C, "k": k,
            "pareto": pareto,
        }
        results_path.write_text(json.dumps(results, indent=2))
        print(f"  Pareto: {'YES' if pareto else 'no'}  (saved)")

    n = len(results)
    won = sum(1 for r in results.values() if r["pareto"])
    print(f"\nDone: SVD-OMP Pareto-dominates VPD on {won}/{n} matrices")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/weight_matrices.pt",
                    help="Path to torch.save dict {module_path: Tensor}.")
    ap.add_argument("--out", default="results/svd_omp_vs_vpd_results.json")
    ap.add_argument("--device", default=None, help="cpu / cuda / mps; auto-detect if unset.")
    ap.add_argument("--vpd-seeds", type=int, default=3)
    ap.add_argument("--vpd-steps", type=int, default=200)
    ap.add_argument("--stab-trials", type=int, default=20)
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
