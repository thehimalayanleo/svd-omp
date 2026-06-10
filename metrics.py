"""Evaluation metrics for SVD-OMP and VPD.

Mirrors Section 3 of the VPD paper:
    sparse_mse:   reconstruction error using only top-k atoms
    faith_mse:    reconstruction error using the full dictionary
    coherence:    max pairwise Frobenius cosine similarity among active atoms
    stability:    support similarity under small weight perturbations
    reproducibility: number of distinct supports across random seeds

We add one SVD-OMP-only metric:
    n_unique_inputs: number of distinct per-input supports in a calibration batch
                     (>1 = genuinely input-dependent; VPD's static g is always 1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from svd_omp import recon, svd_decompose, svd_omp_select
from vpd_baseline import run_vpd


def active_coherence(V: torch.Tensor, U: torch.Tensor, sup) -> float:
    """Max pairwise Frobenius cosine similarity among atoms in `sup`.

    For an orthogonal dictionary (SVD), this is identically 0.
    """
    sup = list(sup)
    if len(sup) < 2:
        return 0.0
    Vn = F.normalize(V.detach(), dim=0)
    Un = F.normalize(U.detach(), dim=1)
    st = torch.tensor(sup, device=V.device)
    G = (Vn[:, st].T @ Vn[:, st]) * (Un[st, :] @ Un[st, :].T)
    G.fill_diagonal_(0)
    return G.abs().max().item()


def support_stability_svd(
    W: torch.Tensor,
    k: int,
    n_trials: int = 20,
    sigma: float = 0.01,
) -> float:
    """Davis-Kahan-style stability of the top-k right singular subspace under W + noise.

    For each trial, recompute SVD of `W + eps`, compare top-k right singular
    vectors via mean of max-cosine-similarity per base vector.
    """
    _, _, Vt0 = torch.linalg.svd(W, full_matrices=False)
    base_vecs = Vt0[:k]

    sims = []
    for t in range(n_trials):
        torch.manual_seed(t)
        W_n = W + torch.randn_like(W) * sigma
        _, _, Vtn = torch.linalg.svd(W_n, full_matrices=False)
        noisy_vecs = Vtn[:k]
        cos_mat = (base_vecs @ noisy_vecs.T).abs()
        sims.append(cos_mat.max(dim=1).values.mean().item())
    return sum(sims) / len(sims)


def evaluate_svd_omp(
    W: torch.Tensor,
    V_dict: torch.Tensor,
    U_dict: torch.Tensor,
    S: torch.Tensor,
    k: int,
    n_stab_trials: int = 20,
    sigma: float = 0.01,
    batch_size: int = 256,
) -> dict:
    """Compute all metrics for the SVD dictionary on weight W."""
    dev = W.device
    d_out, d_in = W.shape
    C = V_dict.shape[1]

    # Static top-k by singular value (Eckart-Young optimal rank-k).
    w_static = torch.zeros(C, device=dev)
    w_static[:k] = 1.0
    e_sp_static = (recon(V_dict, U_dict, w_static) - W).pow(2).mean().item()

    # Per-input OMP on a random calibration batch.
    torch.manual_seed(0)
    phi_batch = torch.randn(batch_size, d_in, device=dev) * 0.5
    W_phi_true = phi_batch @ W.T
    W_hat, support_batch, _ = svd_omp_select(phi_batch, V_dict, U_dict, S, k)
    e_sp_perinput = (W_hat - W_phi_true).pow(2).mean().item()

    e_f = (recon(V_dict, U_dict, torch.ones(C, device=dev)) - W).pow(2).mean().item()

    mu = active_coherence(V_dict, U_dict, set(range(k)))
    st = support_stability_svd(W, k, n_trials=n_stab_trials, sigma=sigma)

    n_unique_inputs = len({tuple(r.tolist()) for r in support_batch})

    return {
        "sparse_mse": e_sp_static,
        "sparse_mse_input": e_sp_perinput,
        "faith_mse": e_f,
        "coherence": mu,
        "stability": st,
        "n_active": k,
        "n_unique_inputs": n_unique_inputs,
        "n_seeds_unique": 1,  # SVD is deterministic.
    }


def evaluate_vpd(
    W: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    g: torch.Tensor,
    k: int,
    C: int,
) -> dict:
    """Compute the non-stability metrics for one trained VPD (V, U, g)."""
    dev = W.device
    gi = torch.sigmoid(g)
    topk = torch.argsort(gi, descending=True)[:k]
    mask = torch.zeros(C, device=dev)
    mask[topk] = gi[topk]
    e_sp = (recon(V, U, mask) - W).pow(2).mean().item()
    e_f = (recon(V, U, torch.ones(C, device=dev)) - W).pow(2).mean().item()
    sup = set(topk.tolist())
    mu = active_coherence(V, U, sup)
    return {
        "sparse_mse": e_sp,
        "faith_mse": e_f,
        "coherence": mu,
        "stability": 1.0,  # fixed g is trivially stable; overwritten by evaluate_vpd_retrain.
        "n_active": len(sup),
    }


def evaluate_vpd_retrain(
    W: torch.Tensor,
    C: int,
    k: int,
    n_train: int = 150,
    n_trials: int = 8,
    sigma: float = 0.01,
    seed: int = 0,
) -> float:
    """Mean Jaccard between base support and supports retrained on `W + noise`.

    Expensive (runs full VPD training `n_trials + 1` times). Used for the
    fair stability comparison: VPD's `g` is trained on a specific W, so the
    Davis-Kahan analogue is "does retraining on W + eps give the same g?"
    """
    Vb, Ub, gb = run_vpd(W, C, k, n=n_train, seed=seed, verbose=False)
    base = set(torch.argsort(torch.sigmoid(gb), descending=True)[:k].tolist())
    jaccs = []
    for t in range(n_trials):
        torch.manual_seed(t + 999)
        W_n = W + torch.randn_like(W) * sigma
        Vt, Ut, gt = run_vpd(W_n, C, k, n=n_train, seed=seed, verbose=False)
        s = set(torch.argsort(torch.sigmoid(gt), descending=True)[:k].tolist())
        jaccs.append(len(s & base) / max(1, len(s | base)))
    return sum(jaccs) / len(jaccs)


__all__ = [
    "active_coherence",
    "support_stability_svd",
    "evaluate_svd_omp",
    "evaluate_vpd",
    "evaluate_vpd_retrain",
]
