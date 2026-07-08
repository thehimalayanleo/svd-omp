"""End-to-end test at production scale.

Runs the full comparison loop against synthetic weight matrices sized like
the real Goodfire 67M model (4 layers x 6 module types = 24 matrices).
Does NOT need the Goodfire model or wandb access.

Purpose: catches integration bugs that only surface at real shapes
(memory, torch dispatch, dtype interactions, module_config wiring).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from svd_omp import svd_decompose, svd_omp_select
from vpd_baseline import run_vpd
from metrics import evaluate_svd_omp, evaluate_vpd
from causal_ablation import causal_damage, redundancy
from model_config import TARGET_MODULES, get_C, get_k


def synthetic_weights():
    """Build synthetic weight matrices with the exact shapes and names of the
    real 67M LlamaSimpleMLP so compare_vpd.py works unchanged."""
    shape_of = {
        "attn.q_proj": (768, 768),
        "attn.k_proj": (768, 768),
        "attn.v_proj": (768, 768),
        "attn.o_proj": (768, 768),
        "mlp.c_fc":    (3072, 768),
        "mlp.down_proj": (768, 3072),
    }

    def shape_for(path):
        for suffix, shape in shape_of.items():
            if path.endswith(suffix):
                return shape
        raise ValueError(path)

    weights = {}
    for i, path in enumerate(TARGET_MODULES):
        torch.manual_seed(1000 + i)
        d_out, d_in = shape_for(path)
        # Low-rank + noise: matches the "structured plus noise" story of
        # real transformer weights.
        rank = min(d_out, d_in) // 4
        A = torch.randn(d_out, rank)
        B = torch.randn(d_in, rank)
        W = A @ B.T + 0.05 * torch.randn(d_out, d_in)
        # Normalize to Xavier-ish scale.
        W = W / W.norm() * (d_in * d_out) ** 0.25
        weights[path] = W
    return weights


def test_full_24_matrix_sweep():
    """Every real target module runs end-to-end without shape/dtype errors."""
    weights = synthetic_weights()
    t0 = time.time()

    results = {}
    for i, path in enumerate(TARGET_MODULES):
        W = weights[path]
        C = get_C(path)
        k = get_k(path)

        # Guard: pathological C > min(d_out, d_in) would break SVD; the real
        # config has this for mlp.down_proj (C=3584, min(768,3072)=768).
        # svd_decompose already clamps via min(C, S.shape[0]).

        V_dict, U_dict, S = svd_decompose(W, C)
        assert V_dict.shape[0] == W.shape[1]
        assert U_dict.shape[1] == W.shape[0]

        r_svd = evaluate_svd_omp(W, V_dict, U_dict, S, k,
                                 n_stab_trials=3, batch_size=64)
        assert r_svd["coherence"] < 1e-3   # generous — orthogonal in principle
        assert 0.0 < r_svd["stability"] <= 1.0
        assert r_svd["n_unique_inputs"] >= 1

        # Only run VPD training on the smaller matrices to keep this test fast.
        # Full 24-matrix VPD training would be minutes on CPU.
        if i < 3:
            Vv, Uv, gv = run_vpd(W, C, k, n=40, seed=0, verbose=False)
            r_vpd = evaluate_vpd(W, Vv, Uv, gv, k, C)
            # These are only asserted for the first few:
            # SVD-OMP should beat 40-step VPD on the metrics that are
            # theoretically bounded (coherence, faith_mse) even if VPD is barely trained.
            assert r_svd["coherence"] < r_vpd["coherence"], (path, r_svd["coherence"], r_vpd["coherence"])
            assert r_svd["faith_mse"] < r_vpd["faith_mse"], (path, r_svd["faith_mse"], r_vpd["faith_mse"])

        results[path] = r_svd
        # progress
        print(f"  [{i+1:>2}/24] {path:<22} shape={tuple(W.shape)} C={C} k={k}  "
              f"sparse={r_svd['sparse_mse']:.4f} coh={r_svd['coherence']:.2e} "
              f"stab={r_svd['stability']:.3f}")

    elapsed = time.time() - t0
    print(f"\n24-matrix sweep completed in {elapsed:.1f}s")
    assert len(results) == 24


def test_causal_ablation_all_module_types():
    """Causal ablation runs across every module type (attn q/k/v/o + mlp c_fc/down_proj)."""
    weights = synthetic_weights()

    # One representative of each module type from layer 0.
    reps = [
        "h.0.attn.q_proj", "h.0.attn.k_proj",
        "h.0.attn.v_proj", "h.0.attn.o_proj",
        "h.0.mlp.c_fc",    "h.0.mlp.down_proj",
    ]

    for path in reps:
        W = weights[path]
        C = get_C(path)
        k = get_k(path)
        V_dict, U_dict, S = svd_decompose(W, C)
        torch.manual_seed(0)
        phi = torch.randn(32, W.shape[1]) * 0.3
        _, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k)

        d = causal_damage(phi, V_dict, U_dict, support)
        r = redundancy(phi, V_dict, U_dict, support)
        assert d > 0
        assert 0.0 <= r < 1e-3, f"orthogonal dict must have redundancy=0, got {r} on {path}"
        print(f"  {path:<22}  damage={d:.3f}  redundancy={r:.5f}")


def test_downstream_causal_damage():
    """Composed causal damage (this layer's atom -> next layer output)."""
    weights = synthetic_weights()
    path, down_path = "h.0.attn.q_proj", "h.0.attn.k_proj"
    W = weights[path]
    W_down = weights[down_path]
    C = get_C(path); k = get_k(path)

    V_dict, U_dict, S = svd_decompose(W, C)
    torch.manual_seed(0)
    phi = torch.randn(32, W.shape[1]) * 0.3
    _, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k)

    d_local = causal_damage(phi, V_dict, U_dict, support)
    d_dn    = causal_damage(phi, V_dict, U_dict, support, W_down=W_down)
    assert d_local > 0 and d_dn > 0
    print(f"  local damage={d_local:.3f}   downstream damage={d_dn:.3f}")


if __name__ == "__main__":
    passed = failed = 0
    tests = {n: v for n, v in globals().items()
             if callable(v) and n.startswith("test_")}
    for name, fn in tests.items():
        print(f"\n== {name} ==")
        try:
            fn()
            print(f"  PASS")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} end-to-end tests passed")
    sys.exit(0 if failed == 0 else 1)
