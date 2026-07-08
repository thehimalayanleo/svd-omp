"""Smoke + property tests for block-SVD-OMP and BSF-W.

Runs synthetic-only (no Goodfire model). Same convention as test_svd_omp.py:
    python tests/test_block_svd_omp.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from svd_omp import svd_decompose, svd_omp_select
from block_svd_omp import (
    block_reconstruction,
    block_svd_decompose,
    block_svd_omp_select,
    block_svd_omp_select_vectorized,
)
from bsf_weights import evaluate_bsf_weights, run_bsf_weights
from metrics import block_coherence, evaluate_block_svd_omp, evaluate_svd_omp
from vpd_baseline import run_vpd
from metrics import evaluate_vpd


# ---------- block_svd_decompose ----------

def test_block_layout_covers_C_atoms():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S, blocks = block_svd_decompose(W, C=32, r=4)
    assert len(blocks) == 8
    covered = set()
    for s, e in blocks:
        for i in range(s, e):
            covered.add(i)
    assert covered == set(range(32))


def test_last_block_shrinks_when_C_not_divisible_by_r():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    _, _, _, blocks = block_svd_decompose(W, C=30, r=4)
    assert len(blocks) == 8
    assert blocks[-1] == (28, 30)  # last block has only 2 atoms


# ---------- Selection: closed-form top-k on block norms ----------

def test_block_selection_matches_block_norms():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S, blocks = block_svd_decompose(W, C=32, r=4)
    phi = torch.randn(1, 96)

    W_hat, support, scores = block_svd_omp_select(
        phi, V_dict, U_dict, S, blocks, k_blocks=2)

    # Recompute block scores by hand.
    projs = (phi @ V_dict).squeeze(0) * S
    ref_scores = torch.tensor(
        [projs[s:e].norm().item() for s, e in blocks]
    )
    top2 = set(torch.topk(ref_scores, 2).indices.tolist())
    assert set(support[0].tolist()) == top2


def test_vectorized_matches_loop_for_uniform_blocks():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S, blocks = block_svd_decompose(W, C=32, r=4)
    phi = torch.randn(16, 96)

    W_v, sup_v, sc_v = block_svd_omp_select_vectorized(
        phi, V_dict, U_dict, S, blocks, k_blocks=3)
    W_l, sup_l, sc_l = block_svd_omp_select(
        phi, V_dict, U_dict, S, blocks, k_blocks=3)

    assert torch.allclose(sc_v, sc_l, atol=1e-5)
    # Supports may differ in order but should match as sets.
    for i in range(16):
        assert set(sup_v[i].tolist()) == set(sup_l[i].tolist())
    assert torch.allclose(W_v, W_l, atol=1e-4)


def test_vectorized_falls_back_when_blocks_uneven():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S, blocks = block_svd_decompose(W, C=30, r=4)  # last block size 2
    phi = torch.randn(8, 96)
    # Both should return valid results (loop mode when uneven).
    W_hat, sup, sc = block_svd_omp_select_vectorized(
        phi, V_dict, U_dict, S, blocks, k_blocks=3)
    assert sup.shape == (8, 3)
    assert torch.isfinite(W_hat).all()


# ---------- Block Eckart-Young ----------

def test_block_full_reconstructs_W():
    """All K blocks together = full SVD-C atoms => exact reconstruction (up to C)."""
    torch.manual_seed(0)
    W = torch.randn(48, 64)
    C_full = min(W.shape)  # full rank
    V_dict, U_dict, S, blocks = block_svd_decompose(W, C=C_full, r=4)
    W_full = block_reconstruction(V_dict, U_dict, blocks, list(range(len(blocks))))
    assert (W_full - W).norm() / W.norm() < 1e-5


def test_block_top_k_matches_atom_top_k_when_blocks_align():
    """Block Eckart-Young: taking the first k blocks == taking the first k*r atoms."""
    torch.manual_seed(0)
    W = torch.randn(48, 64)
    V_dict, U_dict, S, blocks = block_svd_decompose(W, C=32, r=4)

    # First 4 blocks = first 16 atoms.
    W_blocks = block_reconstruction(V_dict, U_dict, blocks, list(range(4)))

    w_atoms = torch.zeros(32)
    w_atoms[:16] = 1.0
    from svd_omp import recon as recon_full
    W_atoms = recon_full(V_dict, U_dict, w_atoms)
    assert torch.allclose(W_blocks, W_atoms, atol=1e-5)


# ---------- Block coherence: 0 for the SVD block dict ----------

def test_block_coherence_zero_for_svd_blocks():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S, blocks = block_svd_decompose(W, C=32, r=4)
    coh = block_coherence(V_dict, U_dict, blocks, active_block_ids=[0, 1, 2])
    assert coh < 1e-4, f"SVD blocks should be orthogonal, got block_coherence={coh}"


# ---------- Per-input diversity: blocks vary with input ----------

def test_per_input_block_supports_vary():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S, blocks = block_svd_decompose(W, C=32, r=4)
    phi = torch.randn(64, 96)
    _, support, _ = block_svd_omp_select_vectorized(
        phi, V_dict, U_dict, S, blocks, k_blocks=3)
    n_unique = len({tuple(row.tolist()) for row in support})
    assert n_unique > 20, f"expected many distinct block supports, got {n_unique}"


# ---------- BSF-W baseline runs ----------

def test_bsf_weights_trains():
    torch.manual_seed(0)
    W = torch.randn(32, 48)
    V, U, g, blocks = run_bsf_weights(
        W, C=16, r=4, k_blocks=2, n=40, seed=0, verbose=False)
    assert V.shape == (48, 16)
    assert U.shape == (16, 32)
    assert g.shape == (4,)     # K = C/r = 16/4 = 4 blocks
    assert len(blocks) == 4


def test_evaluate_bsf_weights_metrics():
    torch.manual_seed(0)
    W = torch.randn(32, 48)
    V, U, g, blocks = run_bsf_weights(
        W, C=16, r=4, k_blocks=2, n=40, seed=0, verbose=False)
    r = evaluate_bsf_weights(W, V, U, g, blocks, k_blocks=2)
    for key in ("sparse_mse", "faith_mse", "coherence", "n_active_blocks", "n_active_atoms"):
        assert key in r


# ---------- End-to-end: block-SVD-OMP >= BSF-W on synthetic low-rank W ----------

def test_block_svd_omp_beats_bsf_on_synthetic_low_rank():
    """Same story as SVD-OMP > VPD, one level up: block-SVD-OMP should beat
    a matching-parameter BSF-W baseline on structured synthetic data."""
    torch.manual_seed(0)
    d_out, d_in = 48, 64
    rank = 8
    U0 = torch.randn(d_out, rank)
    V0 = torch.randn(d_in, rank)
    W = U0 @ V0.T + 0.1 * torch.randn(d_out, d_in)

    C, r, k_blocks = 16, 4, 2

    V_dict, U_dict, S, blocks = block_svd_decompose(W, C, r)
    r_bs = evaluate_block_svd_omp(W, V_dict, U_dict, S, blocks, k_blocks,
                                  n_stab_trials=3, batch_size=64)

    Vv, Uv, gv, blocks_v = run_bsf_weights(
        W, C=C, r=r, k_blocks=k_blocks, n=80, seed=0, verbose=False)
    r_bsf = evaluate_bsf_weights(W, Vv, Uv, gv, blocks_v, k_blocks)

    assert r_bs["sparse_mse"] < r_bsf["sparse_mse"], (r_bs["sparse_mse"], r_bsf["sparse_mse"])
    assert r_bs["faith_mse"]  < r_bsf["faith_mse"],  (r_bs["faith_mse"],  r_bsf["faith_mse"])
    assert r_bs["coherence"]  < r_bsf["coherence"] + 1e-6


# ---------- 4-way: block-SVD-OMP vs BSF-W vs SVD-OMP vs VPD ----------

def test_four_way_comparison_smoke():
    """Just checks all four methods return sensible results on the same W;
    no ordering asserted (see compare_all.py for the sweep)."""
    torch.manual_seed(0)
    d_out, d_in = 48, 64
    W = torch.randn(d_out, d_in)
    C_1d, k_1d = 16, 4
    C_bl, r, k_blocks = 16, 4, 2  # 4 blocks * 4 rank = 16 atoms; keep 2 blocks = 8 atoms

    # SVD-OMP
    V1, U1, S1 = svd_decompose(W, C_1d)
    r1 = evaluate_svd_omp(W, V1, U1, S1, k_1d, n_stab_trials=3, batch_size=32)

    # block-SVD-OMP
    V2, U2, S2, blocks = block_svd_decompose(W, C_bl, r)
    r2 = evaluate_block_svd_omp(W, V2, U2, S2, blocks, k_blocks,
                                n_stab_trials=3, batch_size=32)

    # VPD
    Vv, Uv, gv = run_vpd(W, C_1d, k_1d, n=40, seed=0, verbose=False)
    r3 = evaluate_vpd(W, Vv, Uv, gv, k_1d, C_1d)

    # BSF-W
    Vb, Ub, gb, blb = run_bsf_weights(W, C_bl, r, k_blocks, n=40, seed=0, verbose=False)
    r4 = evaluate_bsf_weights(W, Vb, Ub, gb, blb, k_blocks)

    for name, r in [("SVD-OMP", r1), ("block-SVD-OMP", r2), ("VPD", r3), ("BSF-W", r4)]:
        assert 0 <= r["sparse_mse"] < 100, f"{name} sparse_mse insane: {r['sparse_mse']}"
        assert 0 <= r["coherence"] <= 1.0 + 1e-6, f"{name} coherence out of range"


if __name__ == "__main__":
    passed = failed = 0
    tests = {n: v for n, v in globals().items()
             if callable(v) and n.startswith("test_")}
    for name, fn in tests.items():
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(0 if failed == 0 else 1)
