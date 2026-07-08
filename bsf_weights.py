"""
BSF-W: BSF-style trained baseline on weight matrices.

BSF (Block-Sparse Featurizers, Goodfire 2026) applied to activations uses a
learned encoder-decoder plus block-level TopK sparsity. Here we translate that
recipe to *weight decomposition* so we can compare apples-to-apples against
VPD, SVD-OMP, and block-SVD-OMP.

Model per weight matrix W [d_out, d_in]:
    K blocks of rank r each      (so C = K * r atoms total)
    V learned, shape [d_in, C]
    U learned, shape [C, d_out]
    Block-level scalar gates g in R^K, trained via straight-through TopK
    Reconstruction:  W_hat = sum_{b active} (V_b @ U_b)   (b spans r atoms)

Training loss:
    L = || W - W_hat ||_F^2  +  lambda * sum_b || g_b ||    (block-norm regularizer)

Sparsity is enforced by keeping only the top-k blocks by g_b in the forward
pass; gradients flow through all K blocks (straight-through).

The result of training: a bag of K blocks such that any k of them approximate
W well, chosen by g.
"""

from __future__ import annotations

import math
import time

import torch

from svd_omp import recon


def _block_recon(V: torch.Tensor, U: torch.Tensor, gate_atoms: torch.Tensor) -> torch.Tensor:
    """W_hat = sum_c gate[c] * outer(U[c], V[:, c]). `gate_atoms` shape [C]."""
    return (V @ (gate_atoms.unsqueeze(1) * U)).T


def run_bsf_weights(
    W: torch.Tensor,
    C: int,
    r: int,
    k_blocks: int,
    n: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Train the BSF-W (V, U, g) on one weight matrix.

    Args:
        W: weight matrix, shape [d_out, d_in].
        C: total number of atoms. Should equal K * r for K blocks.
        r: rank per block.
        k_blocks: number of active blocks at inference.
        n: training steps.
        seed: torch RNG seed.
        verbose: print progress every 50 steps.

    Returns:
        V, U, g_blocks, blocks: trained tensors + block layout (list of (s, e)).
    """
    assert C % r == 0, f"C={C} must be divisible by r={r}"
    K = C // r
    d_out, d_in = W.shape
    dev = W.device

    torch.manual_seed(seed)
    V = torch.empty(d_in, C, device=dev).normal_(0, 1 / math.sqrt(d_in)).requires_grad_(True)
    U = torch.empty(C, d_out, device=dev).normal_(0, 1 / math.sqrt(C)).requires_grad_(True)
    g = torch.zeros(K, device=dev).requires_grad_(True)

    opt = torch.optim.AdamW([V, U, g], lr=3e-3)
    t0 = time.time()

    for step in range(n):
        # Block-TopK straight-through: keep top-k blocks by gate value, zero the rest;
        # gradient flows through all K blocks.
        gi = torch.sigmoid(g)                         # [K]
        # Random exploration mask (like VPD's stochastic mask) blended with gate.
        expl = 0.5 + 0.5 * torch.rand(K, device=dev)  # [K]
        soft_gate = gi + (1 - gi) * expl.detach()      # [K]

        # Straight-through TopK: hard forward, soft backward
        topk_idx = soft_gate.topk(k_blocks).indices
        hard_gate = torch.zeros(K, device=dev)
        hard_gate[topk_idx] = 1.0
        block_gate = hard_gate + (soft_gate - soft_gate.detach())   # STE identity

        # Broadcast block gate to atom gate.
        atom_gate = block_gate.repeat_interleave(r)                # [C]

        # Reconstruction losses.
        L_recon_sparse = (_block_recon(V, U, atom_gate) - W).pow(2).mean()
        # Faithfulness (full dictionary): keep V, U close to reconstructing W entirely.
        L_faith = (_block_recon(V, U, torch.ones(C, device=dev)) - W).pow(2).mean()

        # Sparsity regularizer on block gates (equivalent to group-L1 on codes).
        L_sparsity = gi.pow(0.5).mean()

        loss = L_faith + 0.5 * L_recon_sparse + 0.02 * L_sparsity
        loss.backward()
        opt.step()
        opt.zero_grad()

        if verbose and step % 50 == 0:
            print(f"    bsf-w {step:>3}/{n}  recon={L_recon_sparse.item():.5f}  "
                  f"faith={L_faith.item():.5f}  ({time.time() - t0:.1f}s)")

    blocks = [(i, i + r) for i in range(0, C, r)]
    return V.detach(), U.detach(), g.detach(), blocks


def evaluate_bsf_weights(
    W: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    g_blocks: torch.Tensor,
    blocks: list[tuple[int, int]],
    k_blocks: int,
) -> dict:
    """Evaluate trained BSF-W on the same metric family as SVD-OMP / VPD."""
    from metrics import active_coherence

    dev = W.device
    C = V.shape[1]
    K = g_blocks.shape[0]

    # Static top-k blocks by gate.
    gi = torch.sigmoid(g_blocks)
    top_blocks = torch.argsort(gi, descending=True)[:k_blocks].tolist()
    atom_gate = torch.zeros(C, device=dev)
    for b in top_blocks:
        s, e = blocks[b]
        atom_gate[s:e] = 1.0

    e_sp = (_block_recon(V, U, atom_gate) - W).pow(2).mean().item()
    e_f = (_block_recon(V, U, torch.ones(C, device=dev)) - W).pow(2).mean().item()

    # Coherence: measure at atom level within the union of selected blocks.
    active_atoms = [i for b in top_blocks for i in range(blocks[b][0], blocks[b][1])]
    mu = active_coherence(V, U, active_atoms)

    return {
        "sparse_mse": e_sp,
        "faith_mse": e_f,
        "coherence": mu,
        "stability": 1.0,  # gate is static; retrained-stability computed separately if needed
        "n_active_blocks": k_blocks,
        "n_active_atoms": len(active_atoms),
    }


__all__ = ["run_bsf_weights", "evaluate_bsf_weights"]
