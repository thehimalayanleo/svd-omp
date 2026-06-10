"""Causal ablation: are the selected components irreplaceable?

For each input and each component in its support, ablate that component and
measure the output change. Compares SVD-OMP supports against VPD supports
on the same weight matrix.

Two flavors:
  * local damage    : ||W phi (full support) - W phi (support minus c)||
  * downstream damage: same but composed with the next weight matrix
                       (cheap proxy for "does cancelling this atom propagate?")

A third metric, `redundancy`, measures how much each selected atom's
contribution is already covered by the other k-1 atoms in the same support.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from model_config import TARGET_MODULES, get_C, get_k
from svd_omp import svd_decompose, svd_omp_select
from vpd_baseline import run_vpd


def causal_damage(phi_batch, V, U, support_batch, W_down=None):
    """Mean output-norm change from ablating one component at a time."""
    B, k = support_batch.shape
    C = V.shape[1]
    damages = []
    for i in range(B):
        sup = support_batch[i]
        proj_i = phi_batch[i] @ V
        w_base = torch.zeros(C, device=V.device)
        for c in sup:
            w_base[c] = proj_i[c]
        out_base = w_base @ U
        if W_down is not None:
            out_base = F.gelu(out_base) @ W_down.T
        dmg = []
        for j in range(k):
            c = sup[j].item()
            w_abl = w_base.clone()
            w_abl[c] = 0.0
            out_abl = w_abl @ U
            if W_down is not None:
                out_abl = F.gelu(out_abl) @ W_down.T
            dmg.append((out_base - out_abl).norm().item())
        damages.append(sum(dmg) / k)
    return sum(damages) / B


def redundancy(phi_batch, V, U, support_batch):
    """0 = perfectly irreplaceable, 1 = fully redundant with siblings."""
    B, k = support_batch.shape
    C = V.shape[1]
    reds = []
    for i in range(B):
        sup = support_batch[i]
        proj_i = phi_batch[i] @ V
        w_full = torch.zeros(C, device=V.device)
        for c in sup:
            w_full[c] = proj_i[c]
        out_full = w_full @ U
        red = []
        for j in range(k):
            c = sup[j].item()
            w_wo = w_full.clone()
            w_wo[c] = 0.0
            out_wo = w_wo @ U
            c_added = (out_full - out_wo).norm().item()
            c_solo = (proj_i[c] * U[c]).norm().item()
            red.append(1.0 - min(1.0, c_added / max(c_solo, 1e-9)))
        reds.append(sum(red) / k)
    return sum(reds) / B


def main(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = torch.load(args.weights, map_location=device)

    W = weights[args.module].to(device).float()
    C = get_C(args.module)
    k = get_k(args.module)
    print(f"{args.module}  [{tuple(W.shape)}]  C={C}  k={k}")

    V_dict, U_dict, S = svd_decompose(W, C)
    torch.manual_seed(0)
    phi_batch = torch.randn(args.batch, W.shape[1], device=device) * 0.3
    _, support_svd, _ = svd_omp_select(phi_batch, V_dict, U_dict, S, k)

    Vv, Uv, gv = run_vpd(W, C, k, n=args.vpd_steps, seed=0, verbose=False)
    topk_v = torch.argsort(torch.sigmoid(gv), descending=True)[:k]
    support_vpd = topk_v.unsqueeze(0).expand(args.batch, -1)

    W_down = weights[args.downstream].to(device).float() if args.downstream else None

    t0 = time.time()
    rows = []
    rows.append(("Causal damage (local)",
                 causal_damage(phi_batch, V_dict, U_dict.T, support_svd),
                 causal_damage(phi_batch, Vv, Uv.T, support_vpd), "higher"))
    if W_down is not None:
        rows.append(("Causal damage (downstream)",
                     causal_damage(phi_batch, V_dict, U_dict.T, support_svd, W_down=W_down),
                     causal_damage(phi_batch, Vv, Uv.T, support_vpd, W_down=W_down), "higher"))
    rows.append(("Redundancy",
                 redundancy(phi_batch, V_dict, U_dict.T, support_svd),
                 redundancy(phi_batch, Vv, Uv.T, support_vpd), "lower"))

    print(f"  {'Metric':<32} {'SVD-OMP':>10} {'VPD':>10}  better")
    print("  " + "-" * 60)
    for name, sv, vp, direction in rows:
        win = (sv > vp) if direction == "higher" else (sv < vp)
        tag = "*" if win else " "
        print(f"  {name:<32} {sv:>10.5f} {vp:>10.5f}  {direction:>6}{tag}")
    print(f"  ({time.time() - t0:.1f}s)")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/weight_matrices.pt")
    ap.add_argument("--module", default="h.0.attn.q_proj",
                    help="Target module to ablate.")
    ap.add_argument("--downstream", default="h.0.attn.k_proj",
                    help="Optional downstream matrix for the composed-damage metric.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--vpd-steps", type=int, default=200)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
