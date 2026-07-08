"""
Trainable SVD-OMP (scaffold mode).

Keeps V, U frozen at their SVD values and learns only two small per-block
tensors: a multiplicative scale s_b and an additive bias b_b.

Effective per-input block score:
    score_b(phi) = s_b * || diag(S_b) V_b^T phi ||_2 + b_b

At init, s_b = 1 and b_b = 0, so the score reduces exactly to the analytic
block-SVD-OMP score. Training minimizes per-input reconstruction MSE on a
batch of random phi, so the loss at step 0 equals block-SVD-OMP's
sparse_mse_input and can only improve. Only 2K trainable parameters per
matrix (K = number of blocks).

For an apples-to-apples comparison to BSF-W, the natural knob is a warm-
started BSF-W (V, U init from SVD instead of random Gaussian). We add that
via `warm_start_svd=True` in `bsf_weights.run_bsf_weights`; see the tests.
"""

from __future__ import annotations

import time

import torch

from svd_omp import svd_decompose


def _split_into_blocks(C: int, r: int) -> list[tuple[int, int]]:
    C_bl = (C // r) * r
    return [(i, i + r) for i in range(0, C_bl, r)]


def run_trainable_svd_omp(
    W: torch.Tensor,
    C: int,
    r: int,
    k_blocks: int,
    n: int = 200,
    seed: int = 0,
    verbose: bool = True,
    batch_size: int = 128,
    lr: float = 3e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Scaffold-mode trainable SVD-OMP.

    Returns:
        V, U, S, ln_scale, bias, blocks:
            V, U, S: frozen SVD decomposition.
            ln_scale, bias: learned per-block corrections, shape [K].
            blocks: list of (start, end) index ranges into the C-atom axis.
    """
    assert C >= r
    d_out, d_in = W.shape
    dev = W.device

    # SVD only has min(d_out, d_in) singular components. If C exceeds that,
    # cap effective C to the SVD rank so we don't try to allocate atoms that
    # do not exist.
    C_requested = (C // r) * r
    max_svd = min(d_out, d_in)
    C_eff = min(C_requested, (max_svd // r) * r)
    blocks = _split_into_blocks(C_eff, r)
    K = len(blocks)

    V, U, S = svd_decompose(W, C_eff)   # V [d_in, C_eff], U [C_eff, d_out], S [C_eff]
    V = V.detach()
    U = U.detach()
    S = S.detach()

    ln_scale = torch.zeros(K, device=dev, requires_grad=True)
    bias = torch.zeros(K, device=dev, requires_grad=True)

    opt = torch.optim.AdamW([ln_scale, bias], lr=lr)
    torch.manual_seed(seed)
    t0 = time.time()

    for step in range(n):
        phi = torch.randn(batch_size, d_in, device=dev) * 0.5
        W_phi_true = phi @ W.T                                   # [B, d_out]

        # Base block score: analytic block norm ||diag(S_b) V_b^T phi||_2.
        # In practice, projs*S has that norm block-wise.
        projs = phi @ V                                          # [B, C_eff]
        weighted = (projs * S.unsqueeze(0)).view(batch_size, K, r)
        block_norms = weighted.norm(dim=2)                        # [B, K]
        scale = torch.exp(ln_scale).unsqueeze(0)                  # [1, K]
        scores = scale * block_norms + bias.unsqueeze(0)          # [B, K]

        # Straight-through TopK.
        topk_idx = scores.topk(k_blocks, dim=1).indices
        hard_mask = torch.zeros_like(scores)
        hard_mask.scatter_(1, topk_idx, 1.0)
        soft = scores.softmax(dim=1)
        block_gate = hard_mask + (soft - soft.detach())          # STE
        atom_mask = block_gate.repeat_interleave(r, dim=1)       # [B, C_eff]

        # Reconstruction.
        W_hat = (projs * atom_mask) @ U                           # [B, d_out]
        loss = (W_hat - W_phi_true).pow(2).mean()

        loss.backward()
        opt.step()
        opt.zero_grad()

        if verbose and step % 50 == 0:
            print(f"    trainable-svd-omp {step:>3}/{n}  "
                  f"recon={loss.item():.5f}  ({time.time() - t0:.1f}s)")

    return V, U, S, ln_scale.detach(), bias.detach(), blocks


def trainable_svd_omp_select(
    phi_batch: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    ln_scale: torch.Tensor,
    bias: torch.Tensor,
    blocks: list[tuple[int, int]],
    k_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = phi_batch.shape[0]
    K = len(blocks)
    r = blocks[0][1] - blocks[0][0]

    projs = phi_batch @ V                                        # [B, C_eff]
    weighted = (projs * S.unsqueeze(0)).view(B, K, r)
    block_norms = weighted.norm(dim=2)                            # [B, K]
    scale = torch.exp(ln_scale).unsqueeze(0)
    scores = scale * block_norms + bias.unsqueeze(0)              # [B, K]

    topk = scores.topk(k_blocks, dim=1)
    support = topk.indices
    mask = torch.zeros_like(scores)
    mask.scatter_(1, support, 1.0)
    atom_mask = mask.repeat_interleave(r, dim=1)
    W_hat = (projs * atom_mask) @ U
    return W_hat, support, scores


def evaluate_trainable_svd_omp(
    W: torch.Tensor,
    V: torch.Tensor,
    U: torch.Tensor,
    S: torch.Tensor,
    ln_scale: torch.Tensor,
    bias: torch.Tensor,
    blocks: list[tuple[int, int]],
    k_blocks: int,
    batch_size: int = 256,
) -> dict:
    """Return both per-input and matrix sparse_mse so this method is
    directly comparable to BSF-W (matrix) and analytic block-SVD-OMP (both).
    """
    from metrics import active_coherence

    dev = W.device
    C_eff = V.shape[1]

    # Per-input sparse recon on a fresh batch.
    torch.manual_seed(0)
    phi = torch.randn(batch_size, W.shape[1], device=dev) * 0.5
    W_phi_true = phi @ W.T
    W_hat, support, scores = trainable_svd_omp_select(
        phi, V, U, S, ln_scale, bias, blocks, k_blocks)
    e_sp_input = (W_hat - W_phi_true).pow(2).mean().item()

    # Matrix sparse recon using canonical (input-independent) top-k blocks
    # for direct comparison to BSF-W's static-gate metric.
    scale = torch.exp(ln_scale)
    canonical = torch.argsort(scale, descending=True)[:k_blocks].tolist()
    atom_gate_static = torch.zeros(C_eff, device=dev)
    for b in canonical:
        s, e = blocks[b]
        atom_gate_static[s:e] = 1.0
    W_recon_static = (V @ (atom_gate_static.unsqueeze(1) * U)).T
    e_sp_matrix = (W_recon_static - W).pow(2).mean().item()

    atom_ones = torch.ones(C_eff, device=dev)
    e_f = ((V @ (atom_ones.unsqueeze(1) * U)).T - W).pow(2).mean().item()

    active_atoms = [i for b in canonical for i in range(blocks[b][0], blocks[b][1])]
    mu = active_coherence(V, U, active_atoms)

    n_unique_inputs = len({tuple(row.tolist()) for row in support})

    return {
        "sparse_mse": e_sp_matrix,        # matrix Frobenius — comparable to BSF-W and VPD
        "sparse_mse_input": e_sp_input,   # per-input — comparable to block-SVD-OMP
        "faith_mse": e_f,
        "coherence": mu,
        "n_active_blocks": k_blocks,
        "n_unique_inputs": n_unique_inputs,
    }


__all__ = [
    "run_trainable_svd_omp",
    "trainable_svd_omp_select",
    "evaluate_trainable_svd_omp",
]
