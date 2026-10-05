# Theorems for the SVD-OMP family, checked in Lean

Everything here is proved in Lean 4 with Mathlib (`lean/SvdOmp.lean`, Lean v4.33.0,
Mathlib tag `v4.33.0`). Every theorem depends only on Lean's three standard axioms
(`propext`, `Classical.choice`, `Quot.sound`; see `lean/Axioms.lean`) and the file has
no `sorry`. Each statement was first checked numerically on 200 random small cases
(`lean/numeric_checks.py`, worst error 4e-11).

## Setting

A weight matrix `W` (m × n) maps a token's layer input `φ` to `Wφ`. All three methods
write `Wφ = Σ_c a_c(φ) write_c` and keep `k` atoms per token.

| Method | Input weight `L_in` | Output weight `L_out` |
|---|---|---|
| plain SVD-OMP | `I` | `I` |
| whitened SVD-OMP | Cholesky factor of the input Gram `E[φφᵀ]` | `I` |
| causal-metric SVD-OMP | same as whitened | Cholesky factor of `M = E[ggᵀ]` |

Recipe: `L_outᵀ W L_in = P diag(s) Qᵀ`, `read = L_in^{-T} Q`, `write = L_out^{-T} P`,
`a_c(φ) = s_c read_cᵀ φ`, keep the `k` largest `|a_c|`.

## T1. Orthonormal top-k is exact (`sparse_error_eq`, `topk_optimal`)

In any real inner product space, let `w_1..w_r` be orthonormal and `y = Σ_c a_c w_c`.
For any support `S` and any coefficients `x`,

    ‖y − Σ_{c∈S} x_c w_c‖² = Σ_{c∈S} (a_c − x_c)² + Σ_{c∉S} a_c².

So among all supports of size `k` and all coefficients, the best choice keeps a set
`T` of `k` largest `|a_c|` with `x = a`.

Proof: the difference is `Σ_c b_c w_c` with `b_c = a_c − x_c` on `S` and `a_c` off it,
and orthonormality makes its squared norm `Σ_c b_c²`. For optimality, `Σ_{c∈S} (a_c − x_c)² ≥ 0`, and an
exchange argument shows a top-k set has the most energy among `k`-sets
(`sum_le_sum_of_top`): every element of `S \ T` is at most the smallest element of
`T \ S`, and the two differences have equal size.

This is why OMP needs no greedy search here: with orthonormal atoms its residual
updates collapse to one ranking.

## T2. The weighted factorisation is exact (`weighted_svd_reconstruction`)

If `L_in`, `L_out` are invertible and `L_outᵀ W L_in = P diag(s) Qᵀ`, then
`write · diag(s) · readᵀ = W`, that is, `Σ_c s_c write_c read_cᵀ = W`. `P` and `Q` do not
need to be orthogonal.

Proof: `L_out^{-T} (P diag(s) Qᵀ) L_in^{-1} = L_out^{-T} L_outᵀ W L_in L_in^{-1} = W`.

## T3. Top-k is exact in the metric `M` (`write_metric_orthonormal`, `metric_topk_optimal`)

If also `PᵀP = I` and `M = L_out L_outᵀ`, then `writeᵀ M write = I`. Then for every token
`φ`, with `a = diag(s) readᵀ φ`, every support `S` and coefficients `x`, the M-weighted
output error `eᵀ M e` with `e = Wφ − Σ_{c∈S} x_c write_c` equals
`Σ_{c∈S} (a_c − x_c)² + Σ_{c∉S} a_c²`, and a top-k set with its own coefficients is optimal.

`L_out = I` gives whitened SVD-OMP (plain output error), and `L_in = L_out = I` gives
plain SVD-OMP. For causal-metric SVD-OMP the minimised quantity is the M-weighted error.

## T4. Averaged weighted error (`weighted_error_identity`)

For samples `φ_1..φ_N` with `(1/N) Σ φ_i φ_iᵀ = L_in L_inᵀ` and `M = L_out L_outᵀ`,

    (1/N) Σ_i (Eφ_i)ᵀ M (Eφ_i) = ‖L_outᵀ E L_in‖_F².

So a fixed (input-independent) approximation `Ŵ` of `W` is judged by the Frobenius
norm of `L_outᵀ (W − Ŵ) L_in`, the matrix the recipe factorises.

## T5. The output metric is the Fisher metric (`fisher_identity`, `fisher_pullback`)

For `y` drawn from `p` (`Σ p = 1`), `E[(e_y − p)(e_y − p)ᵀ] = diag(p) − ppᵀ`. With
`g_y = Jᵀ(e_y − p)` (the layer-output gradient of `log softmax(z)_y` when `J` is the
logits' Jacobian with respect to the layer output), `E[g gᵀ] = Jᵀ (diag(p) − ppᵀ) J`,
the Gauss-Newton (Fisher) metric. That is the `M` causal-metric SVD-OMP estimates from
sampled-label gradients.

## T6. Partial ablations of the complement cannot do more damage (`complement_mask_bound`)

Keep a support `S` at full strength and scale every other atom by a mask `m_c ∈ [0, 1]`.
With M-orthonormal write vectors, the M-weighted deviation is
`Σ_{c∉S} (1 − m_c)² a_c²`, which is at most `Σ_{c∉S} a_c²`, the cost of removing the whole
complement. So under the second-order proxy there is nothing for an adversarial mask to find.
(In the full model this does not survive intact; see the paper.)

## Audit: what the formal statements do and do not cover

- `IsTopK a T k` says `|T| = k` and every coefficient in `T` is at least as large in
  magnitude as every one outside, so ties are allowed. Existence of such a `T` is not
  stated; it is immediate from sorting.
- T2 and T3 take the factorisation as a hypothesis. Existence of an SVD is not proved here.
- In the code, `L_in` and `L_out` are Cholesky factors of ridge-regularised matrices
  (`G + α·mean(diag G)·I`, `M + β·mean(diag M)·I`). The theorems hold exactly for those
  regularised metrics, which differ slightly from the raw sample Gram and Fisher.
- T5 proves the expectation identity. That `e_y − p` is the gradient of
  `log softmax` is standard calculus and is not formalised.
- Not formalised: Eckart-Young (the best static rank-k approximation) and the
  second-order expansion `KL(p ‖ p + δ) ≈ ½ δᵀ F δ`, which links `eᵀMe` to about twice the KL change.
- None of this compares with VPD's own dictionary. T1 and T3 say that, for a fixed
  orthonormal (or M-orthonormal) dictionary, no selector, learned or not, beats top-k on
  that dictionary's local error. VPD learns a different, overcomplete dictionary, so the
  theorems explain why SVD-OMP needs no training to select; they do not say it beats VPD.

## Reproduce

    cd lean && lake update && lake exe cache get && lake build && lake env lean Axioms.lean
