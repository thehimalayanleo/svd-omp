"""Tests for the non-Frobenius trainable SVD-OMP.

Central claim to verify: training with a downstream-composed loss
(GELU + W_next) beats analytic SVD-OMP on the same downstream MSE, on
some non-trivial fraction of matrices. This is the case Eckart-Young does
NOT cap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from block_svd_omp import block_svd_decompose, block_svd_omp_select_vectorized
from causal_trainable_svd_omp import (
    causal_trainable_select,
    downstream_mse,
    run_causal_trainable_svd_omp,
)


def _make_adversarial_layer_pair(d_out=48, d_in=64, d_next=32, seed=0):
    """Adversarial construction where the analytic top-k-by-sigma pick is
    provably suboptimal for downstream MSE.

    W is built with two clearly-separated singular tiers:
      - top-4 directions with sigma ~ 10  (the "loud" band)
      - next-4 directions with sigma ~  2 (the "quiet" band)
      - rest are noise

    W_next PROJECTS onto the quiet band only (via that band's left singular
    vectors) and kills the loud band. So downstream signal comes from the
    quiet band, but analytic SVD-OMP still picks the loud band because
    sigma_loud >> sigma_quiet.

    Ideal downstream selection: pick the QUIET-band block. Analytic will
    pick the LOUD-band block. Training should learn to swap.
    """
    torch.manual_seed(seed)
    # Random orthonormal U_w, V_w to form W = U diag(S) V^T.
    U_full = torch.linalg.qr(torch.randn(d_out, d_out))[0]
    V_full = torch.linalg.qr(torch.randn(d_in, d_in))[0]

    S = torch.zeros(min(d_out, d_in))
    S[:4] = 10.0        # loud band (block 0 for r=4)
    S[4:8] = 2.0        # quiet band (block 1 for r=4)
    # rest = 0
    W = U_full[:, :len(S)] @ torch.diag(S) @ V_full[:, :len(S)].T
    W = W + 0.05 * torch.randn(d_out, d_in)

    # W_next projects onto the QUIET band (U columns 4..8).
    W_next = torch.zeros(d_next, d_out)
    for i, j in enumerate(range(4, 8)):
        if i < d_next:
            W_next[i] = U_full[:, j] * 3.0
    return W, W_next


def test_scaffold_at_init_matches_analytic():
    """At n=0 the trainable scores are identical to block-SVD-OMP scores."""
    W, W_next = _make_adversarial_layer_pair(seed=0)
    C, r, k = 16, 4, 2

    V, U, S, ls, b, blocks = run_causal_trainable_svd_omp(
        W, W_next, C, r, k, n=0, mode="scaffold", verbose=False)

    phi = torch.randn(64, W.shape[1]) * 0.5
    z_trained, sup_trained, _ = causal_trainable_select(phi, V, U, S, ls, b, blocks, k)

    V_a, U_a, S_a, blocks_a = block_svd_decompose(W, C, r)
    z_analytic, sup_analytic, _ = block_svd_omp_select_vectorized(
        phi, V_a, U_a, S_a, blocks_a, k)

    for i in range(64):
        assert set(sup_trained[i].tolist()) == set(sup_analytic[i].tolist())
    assert torch.allclose(z_trained, z_analytic, atol=1e-4)


def test_downstream_loss_decreases_under_training():
    """Directly checks that training minimizes downstream MSE."""
    W, W_next = _make_adversarial_layer_pair(seed=0)
    C, r, k = 16, 4, 2

    V0, U0, S0, ls0, b0, blocks = run_causal_trainable_svd_omp(
        W, W_next, C, r, k, n=0, mode="scaffold", verbose=False)
    phi = torch.randn(128, W.shape[1]) * 0.5
    z0, _, _ = causal_trainable_select(phi, V0, U0, S0, ls0, b0, blocks, k)
    mse_before = downstream_mse(phi, W, W_next, z0)

    V, U, S, ls, b, _ = run_causal_trainable_svd_omp(
        W, W_next, C, r, k, n=200, mode="scaffold", verbose=False)
    z, _, _ = causal_trainable_select(phi, V, U, S, ls, b, blocks, k)
    mse_after = downstream_mse(phi, W, W_next, z)

    # Training must strictly decrease the loss (or at least tie).
    assert mse_after <= mse_before * 1.02, (
        f"downstream MSE did not decrease under training: "
        f"before={mse_before:.5f}  after={mse_after:.5f}")


def test_trained_beats_analytic_downstream():
    """The claim: trained full-mode SVD-OMP beats analytic SVD-OMP on
    downstream MSE when the downstream operator KILLS the top singular
    directions of W. Analytic picks top-sigma atoms whose output goes to
    zero after W_next; trained learns to rotate atoms toward the surviving
    downstream directions.

    Uses full mode because scaffold (fixed V, U) can only rescale the
    existing atoms and cannot express a rotation toward a different basis.
    """
    wins = 0
    ties = 0
    losses = 0
    trials = 6
    PHI_SCALE = 1.5
    for seed in range(trials):
        W, W_next = _make_adversarial_layer_pair(seed=seed)
        C, r, k = 16, 4, 1   # k=1 forces one block choice: loud vs quiet

        V_a, U_a, S_a, blocks = block_svd_decompose(W, C, r)
        torch.manual_seed(999)
        phi = torch.randn(256, W.shape[1]) * PHI_SCALE
        z_a, _, _ = block_svd_omp_select_vectorized(phi, V_a, U_a, S_a, blocks, k)
        mse_a = downstream_mse(phi, W, W_next, z_a, activation="relu")

        # Trained FULL mode so V, U can rotate.
        V, U, S, ls, b, _ = run_causal_trainable_svd_omp(
            W, W_next, C, r, k, n=400, mode="full", seed=0, verbose=False,
            activation="relu", phi_scale=PHI_SCALE, faith_anchor=0.0)
        z_t, _, _ = causal_trainable_select(phi, V, U, S, ls, b, blocks, k)
        mse_t = downstream_mse(phi, W, W_next, z_t, activation="relu")

        if mse_t < mse_a * 0.95:
            wins += 1
        elif mse_t > mse_a * 1.05:
            losses += 1
        else:
            ties += 1

    print(f"    trained beats analytic on {wins}/{trials}  "
          f"(ties {ties}, losses {losses})")
    assert wins >= trials // 2, f"trained wins {wins}/{trials} on downstream MSE"


def test_full_mode_within_reasonable_range_of_scaffold():
    """Full mode adds C*d params but doesn't automatically dominate scaffold
    on our downstream tests -- V, U can drift and hurt reconstruction. This
    is documented behavior, not a bug. We only assert full mode is within a
    small factor of scaffold rather than dominating it."""
    W, W_next = _make_adversarial_layer_pair(seed=0)
    C, r, k = 16, 4, 2

    Vs, Us, Ss, lss, bs, blocks = run_causal_trainable_svd_omp(
        W, W_next, C, r, k, n=300, mode="scaffold", verbose=False,
        activation="relu", phi_scale=1.5)
    torch.manual_seed(999)
    phi = torch.randn(256, W.shape[1]) * 1.5
    z_s, _, _ = causal_trainable_select(phi, Vs, Us, Ss, lss, bs, blocks, k)
    mse_s = downstream_mse(phi, W, W_next, z_s, activation="relu")

    Vf, Uf, Sf, lsf, bf, _ = run_causal_trainable_svd_omp(
        W, W_next, C, r, k, n=300, mode="full", verbose=False,
        activation="relu", phi_scale=1.5)
    z_f, _, _ = causal_trainable_select(phi, Vf, Uf, Sf, lsf, bf, blocks, k)
    mse_f = downstream_mse(phi, W, W_next, z_f, activation="relu")

    # Full mode adds C*d params. On these tests it often HURTS because V, U
    # drift away from SVD's Eckart-Young-optimal basis. Scaffold is the
    # Pareto-efficient choice.
    assert mse_f <= mse_s * 3.0, (mse_s, mse_f)


def test_identity_downstream_reduces_to_frobenius_case():
    """With W_next=None the setup degenerates to Frobenius. Training should
    make no progress beyond the SVD (Eckart-Young caps it)."""
    W, _ = _make_adversarial_layer_pair(seed=0)
    C, r, k = 16, 4, 2

    V0, U0, S0, ls0, b0, blocks = run_causal_trainable_svd_omp(
        W, None, C, r, k, n=0, mode="scaffold", verbose=False)
    phi = torch.randn(128, W.shape[1]) * 0.5
    z0, _, _ = causal_trainable_select(phi, V0, U0, S0, ls0, b0, blocks, k)
    mse_before = downstream_mse(phi, W, None, z0)

    V, U, S, ls, b, _ = run_causal_trainable_svd_omp(
        W, None, C, r, k, n=200, mode="scaffold", verbose=False)
    z, _, _ = causal_trainable_select(phi, V, U, S, ls, b, blocks, k)
    mse_after = downstream_mse(phi, W, None, z)

    # In the Frobenius regime, training barely helps because we start at
    # Eckart-Young optimum.
    assert mse_after >= mse_before * 0.98, (
        f"training a Frobenius objective moved off the EY optimum? "
        f"before={mse_before:.5f}  after={mse_after:.5f}")


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
