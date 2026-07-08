"""Synthetic-data smoke tests + property tests for the whole pipeline.

Runs without the Goodfire 67M model. Any weight matrix works.
    cd svd-omp && python -m pytest tests/ -v
or:
    python tests/test_svd_omp.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from svd_omp import recon, svd_decompose, svd_omp_select
from vpd_baseline import run_vpd
from metrics import (
    active_coherence,
    evaluate_svd_omp,
    evaluate_vpd,
    evaluate_vpd_retrain,
    support_stability_svd,
)
from causal_ablation import causal_damage, redundancy


# ---------- SVD-OMP core ----------

def test_svd_shapes():
    W = torch.randn(64, 96)
    V_dict, U_dict, S = svd_decompose(W, C=32)
    assert V_dict.shape == (96, 32)
    assert U_dict.shape == (32, 64)
    assert S.shape == (32,)
    assert torch.all(S[:-1] >= S[1:]), "singular values must be sorted descending"


def test_full_dictionary_reconstructs_W():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S = svd_decompose(W, C=min(64, 96))
    W_recon = recon(V_dict, U_dict, torch.ones(V_dict.shape[1]))
    err = (W_recon - W).norm() / W.norm()
    assert err < 1e-5, f"full-rank SVD should reconstruct W exactly, got err={err}"


def test_top_k_matches_eckart_young():
    """Truncated SVD is the optimal rank-k Frobenius approximation."""
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    C = 32
    k = 8
    V_dict, U_dict, S = svd_decompose(W, C)
    w_topk = torch.zeros(C)
    w_topk[:k] = 1.0
    W_hat = recon(V_dict, U_dict, w_topk)

    err_ours = (W_hat - W).norm().item()
    # Eckart-Young bound: err = sqrt(sum_{i>k} sigma_i^2)
    _, S_full, _ = torch.linalg.svd(W, full_matrices=False)
    err_ey = torch.sqrt((S_full[k:] ** 2).sum()).item()
    assert abs(err_ours - err_ey) < 1e-4, f"got {err_ours}, EY bound {err_ey}"


def test_svd_omp_select_shapes():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S = svd_decompose(W, C=32)
    phi = torch.randn(16, 96)
    W_hat, support, scores = svd_omp_select(phi, V_dict, U_dict, S, k=4)
    assert W_hat.shape == (16, 64)
    assert support.shape == (16, 4)
    assert scores.shape == (16, 32)
    # support indices in range
    assert support.min() >= 0 and support.max() < 32


def test_per_input_supports_actually_vary():
    """The whole 'input dependence' claim: different inputs → different supports."""
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S = svd_decompose(W, C=32)
    phi = torch.randn(64, 96)
    _, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k=4)
    unique = {tuple(row.tolist()) for row in support}
    assert len(unique) > 32, f"expected many distinct supports, got {len(unique)}"


def test_selection_matches_argmax_of_sigma_times_projection():
    """Score c(phi) = sigma_c * |v_c^T phi|; support = top-k of this."""
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S = svd_decompose(W, C=32)
    phi = torch.randn(1, 96)
    k = 4
    _, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k)

    # Recompute the score independently and check we picked the top-k.
    projs = (phi @ V_dict).squeeze(0)
    scores_ref = projs.abs() * S
    topk_ref = set(torch.topk(scores_ref, k).indices.tolist())
    assert set(support[0].tolist()) == topk_ref


# ---------- Metrics ----------

def test_svd_coherence_is_zero():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, _ = svd_decompose(W, C=32)
    coh = active_coherence(V_dict, U_dict, [0, 1, 2, 3])
    assert coh < 1e-5, f"SVD basis is orthogonal, coherence should be ~0, got {coh}"


def test_evaluate_svd_omp_full_metrics():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    # Full rank so faith_mse should hit floating-point zero.
    C_full = min(W.shape)
    V_dict, U_dict, S = svd_decompose(W, C=C_full)
    r = evaluate_svd_omp(W, V_dict, U_dict, S, k=4, n_stab_trials=3, batch_size=32)
    for key in [
        "sparse_mse", "sparse_mse_input", "faith_mse",
        "coherence", "stability", "n_active", "n_unique_inputs",
    ]:
        assert key in r, f"missing metric {key}"
    assert r["faith_mse"] < 1e-6, f"full SVD should cover W, faith_mse={r['faith_mse']}"
    assert r["coherence"] < 1e-5
    assert r["n_active"] == 4
    assert 0.0 < r["stability"] <= 1.0
    assert r["n_unique_inputs"] > 1


def test_stability_lipschitz_in_perturbation():
    """Davis-Kahan: smaller noise → more stable support."""
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    W = W / W.norm() * 10  # scale up so the noise regime matters
    st_small = support_stability_svd(W, k=4, n_trials=5, sigma=0.001)
    st_big   = support_stability_svd(W, k=4, n_trials=5, sigma=0.05)
    assert st_small >= st_big - 0.05, (st_small, st_big)  # allow small numerical slop


# ---------- VPD baseline ----------

def test_vpd_trains_and_reduces_loss():
    torch.manual_seed(0)
    W = torch.randn(32, 48)
    err_init = None

    # Snapshot initial loss for comparison.
    torch.manual_seed(0)
    d_out, d_in, C = 32, 48, 16
    V0 = torch.empty(d_in, C).normal_(0, 1 / math.sqrt(d_in))
    U0 = torch.empty(C, d_out).normal_(0, 1 / math.sqrt(C))
    err_init = (recon(V0, U0, torch.ones(C)) - W).pow(2).mean().item()

    V, U, g = run_vpd(W, C=C, k=4, n=200, seed=0, verbose=False)
    assert V.shape == (48, 16) and U.shape == (16, 32) and g.shape == (16,)
    err = (recon(V, U, torch.ones(C)) - W).pow(2).mean().item()
    # Training should reduce loss by ~50% at minimum.
    assert err < 0.5 * err_init, f"VPD barely learned: {err_init:.3f} -> {err:.3f}"


def test_evaluate_vpd_full_metrics():
    torch.manual_seed(0)
    W = torch.randn(32, 48)
    V, U, g = run_vpd(W, C=16, k=4, n=40, seed=0, verbose=False)
    r = evaluate_vpd(W, V, U, g, k=4, C=16)
    for key in ["sparse_mse", "faith_mse", "coherence", "stability", "n_active"]:
        assert key in r


def test_vpd_retrain_stability_returns_jaccard():
    torch.manual_seed(0)
    W = torch.randn(32, 48)
    j = evaluate_vpd_retrain(W, C=16, k=4, n_train=20, n_trials=2, sigma=0.01, seed=0)
    assert 0.0 <= j <= 1.0


# ---------- Causal ablation ----------

def test_causal_damage_runs():
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S = svd_decompose(W, C=32)
    phi = torch.randn(8, 96) * 0.3
    _, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k=4)
    d = causal_damage(phi, V_dict, U_dict, support)
    assert d > 0.0, f"causal damage should be nonzero, got {d}"


def test_causal_damage_shape_error_regression():
    """Regression test for the .T bug fixed in commit e8ef9bb.

    Previously the notebook called causal_damage(..., U_dict.T, ...) which
    put U into shape [d_out, C] and blew up the `w @ U` matmul with:
        RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x32 and 64x32)
    Guard against re-introducing that.
    """
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S = svd_decompose(W, C=32)
    phi = torch.randn(8, 96) * 0.3
    _, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k=4)
    # Correct: U_dict is [C, d_out]. This must run.
    _ = causal_damage(phi, V_dict, U_dict, support)
    # Incorrect: U_dict.T is [d_out, C]. This must raise RuntimeError.
    try:
        causal_damage(phi, V_dict, U_dict.T, support)
    except RuntimeError:
        return
    raise AssertionError(".T call should have raised RuntimeError")


def test_redundancy_zero_for_orthogonal_dictionary():
    """For an orthogonal dictionary, atom contributions in an output are
    orthogonal too, so ablating one atom removes exactly its solo contribution
    -> redundancy = 0 by construction."""
    torch.manual_seed(0)
    W = torch.randn(64, 96)
    V_dict, U_dict, S = svd_decompose(W, C=32)
    phi = torch.randn(8, 96) * 0.3
    _, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k=4)
    r = redundancy(phi, V_dict, U_dict, support)
    assert r < 1e-4, f"orthogonal dict should give redundancy=0, got {r}"


# ---------- Full integration: SVD-OMP vs VPD on synthetic W ----------

def test_svd_omp_beats_vpd_on_synthetic_matrix():
    """End-to-end: SVD-OMP should Pareto-dominate a lightly-trained VPD on
    random low-rank plus noise (structure the VPD paper claims to find)."""
    torch.manual_seed(0)
    d_out, d_in, rank = 48, 64, 8
    U0 = torch.randn(d_out, rank)
    V0 = torch.randn(d_in, rank)
    W = U0 @ V0.T + 0.1 * torch.randn(d_out, d_in)

    C, k = 32, 4
    V_dict, U_dict, S = svd_decompose(W, C)
    r_svd = evaluate_svd_omp(W, V_dict, U_dict, S, k, n_stab_trials=3, batch_size=64)

    Vv, Uv, gv = run_vpd(W, C, k, n=60, seed=0, verbose=False)
    r_vpd = evaluate_vpd(W, Vv, Uv, gv, k, C)

    assert r_svd["sparse_mse"] < r_vpd["sparse_mse"], (r_svd["sparse_mse"], r_vpd["sparse_mse"])
    assert r_svd["faith_mse"]  < r_vpd["faith_mse"],  (r_svd["faith_mse"],  r_vpd["faith_mse"])
    assert r_svd["coherence"]  < r_vpd["coherence"],  (r_svd["coherence"],  r_vpd["coherence"])


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
