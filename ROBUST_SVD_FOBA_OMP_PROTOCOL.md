# Robust SVD-FoBa-OMP Third-Test Protocol

Status: frozen before any model prediction was produced for the third test.

## Question

Can FoBa and OMP be combined in a way that addresses the observed question-
distribution failure?

The method has two nested choices:

1. Distributionally robust FoBa chooses a sparse layer support using only the
   two already opened prospective distributions.
2. Within each chosen layer, input-dependent OMP chooses one SVD atom per token.

The third source-disjoint set is used exactly once after the complete search
ends. It does not select a layer, dose, atom, threshold, or stopping rule.

## Development objective

The candidate layer pool is the union of the earlier FoBa supports:
`12, 17, 18, 19, 26, 28, 30, 31, 34, 35`.

For each support, OMP-k1 is evaluated at doses `0, 1, 2, 3, 4` on both earlier
prospective distributions. The dose objective is lexicographic:

1. maximize the smaller target-repair count across the two distributions;
2. maximize total target repairs;
3. maximize the weakest protected-family count; and
4. prefer the smaller dose.

Every protected family must remain at least 22/24 on both distributions. FoBa
uses greedy forward additions and lossless backward pruning, with a maximum of
eight selected layers. The objective and search code are frozen.

## Untouched third test

- Twenty-four previously unused, Qwen3-4B capability-screened sources.
- Six sources per domain across business ethics, high-school psychology,
  high-school world history, and professional law.
- Three A-position and three B-position answers per domain.
- Four matched prompt families per source, for 96 rows.
- No source overlap with either earlier prospective distribution or any
  train, support, calibration, validation, or first-test partition.
- Dataset SHA-256:
  `284f908b32f23e4160b224f7c709225823026ca260582491355e6b7f2021eb44`.

## Frozen comparisons

For each Qwen3-4B organism seed, the third test compares:

- robust-FoBa support with OMP-k1 routing;
- the same support and dose with static top-SVD routing;
- the previously frozen OMP support and dose; and
- twenty deterministic matched-size random layer supports using OMP-k1.

The causal gate passes per seed only if the organism passes, at least 8/24 new
warning targets are repaired, and every protected family remains at least
22/24.

The stronger selector-superiority gate passes only if robust FoBa-OMP strictly
beats same-support static SVD, the previous OMP intervention, and every
protected-feasible random support on both seeds. There is no post-result
relaxation.

## Interpretation

- Both causal gates passing would improve evidence for general sparse repair.
- Both superiority gates passing would be the first prospective evidence for
  the combined FoBa-OMP selector.
- A causal pass without superiority supports sparse repair, not the selector.
- Failure remains a reportable result and the third set is not reused as a new
  calibration set.

All heavy model execution is Modal H100 only. The local 5090 is not used.

## Frozen source hashes

```text
0c8823a4f3532caf8605e802695d7c265c7a9ad959aa4913847f29210e86aa38  modal_robust_svd_foba_omp.py
d5c5d013efc1e92b566ae6552ff4832ac9326ac3b70604287da3337841e71b8e  robust_svd_foba.py
ae625751f6e1d260761a2e3944a9ec4370a975edfdbac099e6b101ae87ddb2e4  prepare_prospective_hybrid_test.py
284f908b32f23e4160b224f7c709225823026ca260582491355e6b7f2021eb44  data/behavior_audit/post_training_regression_hybrid_test.jsonl
c11f982f86e7ddda23a5656ea0aa315ce57ca40c10ed31f63f67d9d3aa5ff705  data/behavior_audit/post_training_regression_hybrid_test_manifest.json
1f8459d02c9d501d3f7b13960b2d99a14a60b304dd6b708aa419f24b50a813f6  tests/test_robust_svd_foba.py
415366674e559b8151698b60fad31425da880ddf5d68128d23c44e47ffdd2c8b  tests/test_prepare_prospective_hybrid_test.py
7d69584c066828247721d8aa35a1dc8d14d7ec0df1fe2c788f6f1074f4029da4  tests/test_modal_robust_svd_foba_omp.py
2c7c4e8cbccba775619861473d0563bb3deb9c6c685f76c7c614cd1e8956b0eb  behavioral_causal_audit.py
ef3a14091857d6e6807bd3cb1192872eed4d8ba93bd241e32cf80e4eafbba0f5  hf_behavioral_causal_audit.py
```
