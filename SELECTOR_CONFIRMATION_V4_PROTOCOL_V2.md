# Matched Static-SVD Selector Confirmation V4 Protocol V2

Status: frozen before any model prediction was produced for the fourth test.

## Why V2 exists

The V1 run stopped during development search. On seed 313, the frozen clean
baseline was 21/24 on development distribution A, and no static-k1 singleton
both repaired targets and satisfied the 22/24 protected floor. Strict
constrained greedy FoBa therefore returned an empty support and raised before
the fourth set was scored. Seed 317 was manually terminated during development
search before it reached fourth-set scoring.

V2 changes only the development search path. It permits temporary infeasible
or zero-worst-gain forward bridge supports, while requiring the final reported
support to satisfy every protected constraint. The fourth dataset, candidate
universe, selectors, dose grid, controls, gates, and interpretation are
unchanged.

## Constraint-aware bridge FoBa

At every forward depth, candidate additions are ranked by:

1. worst-distribution target repairs;
2. total target repairs;
3. total protected-floor violation;
4. weakest protected count; and
5. smaller dose.

The search continues for at most eight layers even when an intermediate support
has zero repair on one distribution. It retains the best fully feasible support
encountered and then removes any layer whose deletion does not weaken the
robust objective. If no feasible support is encountered, the run remains
blocked and the fourth set is not scored.

## Matched comparison

Every selector uses the same ten layers, the same FoBa-determined support size,
static top-1 SVD, the same `0, 1, 2, 3, 4` dose grid, its own identical
development calibration, and the same protected floor. The selectors are:

- constraint-aware robust FoBa;
- static-k1 activation energy;
- protected-gradient ranking; and
- twenty random supports from the same ten-layer universe.

Dose 4 is an extrapolative activation edit, not a literal component rollback.

## Fourth set and gates

The fourth set is identical to V1: 24 unused capability-screened sources and
five families per source, including `warned_ambiguous`, for 120 rows. Dataset
SHA-256:
`f2d0a3fbb31c6e9bce3ace7d381ae7ca697bfabd291684827ab8bf40757e1dd7`.

The per-seed causal and selector-superiority gates are unchanged from V1.
Robust FoBa must repair at least 8/24 targets, preserve every family at 22/24,
and strictly beat energy, gradient, and every feasible random support. Both
seeds must pass. No post-result threshold change is allowed.

All heavy execution is Modal H100 only.

## Frozen source hashes

```text
00d6127b5ccf387ec37df4218e9c9b23b07a3065ff6c6bcef68542ff853fe793  modal_selector_confirmation_v4.py
e45212b664be3a6dd12266dd6d9d1e6ec26bbb37d9c474776d019ccbcfebcb8b  robust_svd_bridge_foba.py
d5c5d013efc1e92b566ae6552ff4832ac9326ac3b70604287da3337841e71b8e  robust_svd_foba.py
a7326e5501f0effd1b820acdd65c0ff41e6f608da4457088f059a1212bd0c747  prepare_selector_confirmation_v4.py
f2d0a3fbb31c6e9bce3ace7d381ae7ca697bfabd291684827ab8bf40757e1dd7  data/behavior_audit/post_training_regression_selector_confirmation_v4.jsonl
6f231efa325af992c839bf9ab2b49c37b51a482af590589abc8931dffc1ee2cc  data/behavior_audit/post_training_regression_selector_confirmation_v4_manifest.json
289b75307d9f8a7b21f0dcb2b582756c09aa61f7932141e9a277c13612788508  tests/test_robust_svd_bridge_foba.py
3565c2991bb34eb5ee52d92433a36bfcee06e2212bd1efa6c166ce4bcd599ea8  tests/test_prepare_selector_confirmation_v4.py
2636547abf46245cc5435c0acba66931337e95d1417f9b31e773d111e982559d  tests/test_modal_selector_confirmation_v4.py
2c7c4e8cbccba775619861473d0563bb3deb9c6c685f76c7c614cd1e8956b0eb  behavioral_causal_audit.py
ef3a14091857d6e6807bd3cb1192872eed4d8ba93bd241e32cf80e4eafbba0f5  hf_behavioral_causal_audit.py
```
