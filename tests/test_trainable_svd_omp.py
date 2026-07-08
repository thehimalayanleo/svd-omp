"""Tests for trainable_svd_omp.py + SVD warm-started BSF-W.

Claims verified:
  1. Scaffold at step 0 == analytic block-SVD-OMP (SVD-init reproduces
     the analytic score exactly).
  2. Training scaffold never makes sparse_mse worse (loss floor).
  3. SVD-warm-started BSF-W beats random-init BSF-W under same budget on
     both sparse_mse and faith_mse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from block_svd_omp import block_svd_decompose
from bsf_weights import evaluate_bsf_weights, run_bsf_weights
from metrics import evaluate_block_svd_omp
from trainable_svd_omp import (
    evaluate_trainable_svd_omp,
    run_trainable_svd_omp,
    trainable_svd_omp_select,
)


def test_scaffold_at_init_matches_block_svd_omp():
    """With ln_scale=0 and bias=0, scaffold selection is identical to
    block-SVD-OMP: score = block norm of sigma-weighted projections."""
    torch.manual_seed(0)
    W = torch.randn(48, 64)
    C, r, k = 16, 4, 2

    V, U, S, ln_scale, bias, blocks = run_trainable_svd_omp(
        W, C, r, k, n=0, verbose=False)

    V_a, U_a, S_a, blocks_a = block_svd_decompose(W, C, r)
    r_analytic = evaluate_block_svd_omp(W, V_a, U_a, S_a, blocks_a, k,
                                        n_stab_trials=3, batch_size=64)
    r_scaffold = evaluate_trainable_svd_omp(W, V, U, S, ln_scale, bias, blocks, k,
                                            batch_size=64)

    # Same phi seed (0), same batch_size, same k_blocks, same underlying V/U/S.
    # sparse_mse_input is the per-input metric (matches analytic sparse_mse_input).
    assert abs(r_scaffold["sparse_mse_input"] - r_analytic["sparse_mse_input"]) < 1e-5, (
        r_scaffold["sparse_mse_input"], r_analytic["sparse_mse_input"])
    assert abs(r_scaffold["faith_mse"] - r_analytic["faith_mse"]) < 1e-6


def test_scaffold_training_does_not_worsen_sparse_mse():
    """Training minimizes exactly the metric we evaluate; loss at step n
    should be <= loss at step 0 (up to small STE / Adam noise)."""
    torch.manual_seed(0)
    W = torch.randn(48, 64)
    C, r, k = 16, 4, 2

    V, U, S, ls_0, b_0, blocks = run_trainable_svd_omp(
        W, C, r, k, n=0, verbose=False)
    r_init = evaluate_trainable_svd_omp(W, V, U, S, ls_0, b_0, blocks, k, batch_size=64)

    V, U, S, ls_t, b_t, _ = run_trainable_svd_omp(
        W, C, r, k, n=100, verbose=False)
    r_after = evaluate_trainable_svd_omp(W, V, U, S, ls_t, b_t, blocks, k, batch_size=64)

    # Loss = per-input MSE — check that.
    assert r_after["sparse_mse_input"] <= r_init["sparse_mse_input"] * 1.05, (
        r_init["sparse_mse_input"], r_after["sparse_mse_input"])


def test_warm_start_bsf_beats_random_init_bsf():
    """Same objective (BSF-W's Frobenius loss), same steps, same seed for
    RNG-controlled parts — only difference is V, U init. SVD init should
    strictly beat random-Gaussian init."""
    torch.manual_seed(0)
    d_out, d_in, rank = 48, 64, 8
    A = torch.randn(d_out, rank)
    B = torch.randn(d_in, rank)
    W = A @ B.T + 0.1 * torch.randn(d_out, d_in)

    C, r, k = 16, 4, 2
    n_steps = 60

    Vw, Uw, gw, blw = run_bsf_weights(
        W, C, r, k, n=n_steps, seed=0, warm_start_svd=True, verbose=False)
    r_warm = evaluate_bsf_weights(W, Vw, Uw, gw, blw, k)

    Vc, Uc, gc, blc = run_bsf_weights(
        W, C, r, k, n=n_steps, seed=0, warm_start_svd=False, verbose=False)
    r_cold = evaluate_bsf_weights(W, Vc, Uc, gc, blc, k)

    assert r_warm["sparse_mse"] < r_cold["sparse_mse"], (
        "warm=", r_warm["sparse_mse"], "cold=", r_cold["sparse_mse"])
    assert r_warm["faith_mse"] < r_cold["faith_mse"], (
        "warm=", r_warm["faith_mse"], "cold=", r_cold["faith_mse"])


def test_warm_bsf_starts_at_eckart_young_optimum():
    """SVD-warm-started BSF-W should hit near-perfect faith_mse immediately
    since V @ (ones * U) already reconstructs W exactly for the top-C SVD."""
    torch.manual_seed(0)
    W = torch.randn(32, 48)
    C, r, k = 16, 4, 2

    # 0 training steps: V, U, g are still at SVD-init values.
    V, U, g, blocks = run_bsf_weights(
        W, C, r, k, n=0, seed=0, warm_start_svd=True, verbose=False)
    r = evaluate_bsf_weights(W, V, U, g, blocks, k)

    # Faith_mse = MSE of top-C truncated SVD reconstruction.
    _, S_all, _ = torch.linalg.svd(W, full_matrices=False)
    ey_bound = (S_all[C:] ** 2).sum().item() / (W.shape[0] * W.shape[1])
    assert abs(r["faith_mse"] - ey_bound) < 1e-4, (r["faith_mse"], ey_bound)


def test_select_shapes():
    torch.manual_seed(0)
    W = torch.randn(48, 64)
    C, r, k = 16, 4, 2
    V, U, S, ls, b, blocks = run_trainable_svd_omp(
        W, C, r, k, n=5, verbose=False)
    phi = torch.randn(8, 64)
    W_hat, support, scores = trainable_svd_omp_select(phi, V, U, S, ls, b, blocks, k)
    assert W_hat.shape == (8, W.shape[0])
    assert support.shape == (8, k)


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
