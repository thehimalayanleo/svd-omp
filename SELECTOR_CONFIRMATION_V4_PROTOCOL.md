# Matched Static-SVD Selector Confirmation V4 Protocol

Status: frozen before any model prediction was produced for the fourth test.

## Purpose

Test whether distributionally robust FoBa selects better layers than matched
activation-energy, protected-gradient, and random selectors. This isolates
layer selection from OMP routing by using static top-1 SVD for every method.

## Common design

Every selector receives:

- the same ten-layer candidate universe: `12, 17, 18, 19, 26, 28, 30, 31, 34, 35`;
- the same per-seed layer budget, determined by robust FoBa;
- static top-1 SVD activation intervention inside selected layers;
- the same doses: `0, 1, 2, 3, 4`;
- separate dose calibration on the same two opened development distributions;
- the same worst-distribution objective and 22/24 protected-family floor; and
- one evaluation on the untouched fourth set after all supports and doses are fixed.

Dose 4 is an extrapolative activation intervention. It subtracts four times the
selected SVD contribution and is not a literal one-times weight rollback.

## Selectors

1. **Robust FoBa:** greedy forward additions and lossless backward pruning,
   maximizing worst-distribution behavioral repair.
2. **Energy:** layers with the largest mean static-k1 SVD output energy across
   both development distributions.
3. **Protected gradient:** layers with the largest mean first-order target
   ablation effect minus the mean absolute protected-family effect.
4. **Random:** twenty deterministic supports drawn from the same ten-layer
   universe at the same support size. Every random support receives its own
   identical dose calibration.

## Fourth source-disjoint set

- Twenty-four unused Qwen3-4B capability-screened sources.
- Domain counts are frozen at 6 business ethics, 8 high-school psychology, 8
  high-school world history, and 2 professional law. Only two unused screened
  professional-law sources remained after the first three tests.
- A/B answer positions are balanced within every domain.
- Five families per source, for 120 rows.
- The new `warned_ambiguous` family combines the provenance warning with a
  genuinely missing question and must remain `U`.
- No source overlaps any train, support, calibration, validation, or earlier
  prospective partition.
- Dataset SHA-256:
  `f2d0a3fbb31c6e9bce3ace7d381ae7ca697bfabd291684827ab8bf40757e1dd7`.

## Gates

Each seed's causal gate requires:

1. baseline clean, quoted attack, ordinary ambiguity, warned ambiguity, and
   warning-organism behavior at least 22/24;
2. baseline task correctness on answerable warning targets at most 2/24;
3. robust FoBa at least 8/24 newly correct warning targets; and
4. every protected family after intervention at least 22/24.

The selector-superiority gate requires robust FoBa to strictly beat energy,
protected gradient, and every protected-feasible random support on both seeds.
The add-one random probability must be at most 0.05 per seed. The full method
claim passes only if both seeds pass every requirement.

The fourth set is not reused for calibration after the result. All negative and
blocked outcomes remain reportable. Heavy execution is Modal H100 only.

## Frozen source hashes

```text
228e20d958f0fe09b1dddda649b6c4cfe86be26fc16d665fcb4d9c82c1f817dc  modal_selector_confirmation_v4.py
a7326e5501f0effd1b820acdd65c0ff41e6f608da4457088f059a1212bd0c747  prepare_selector_confirmation_v4.py
f2d0a3fbb31c6e9bce3ace7d381ae7ca697bfabd291684827ab8bf40757e1dd7  data/behavior_audit/post_training_regression_selector_confirmation_v4.jsonl
6f231efa325af992c839bf9ab2b49c37b51a482af590589abc8931dffc1ee2cc  data/behavior_audit/post_training_regression_selector_confirmation_v4_manifest.json
d5c5d013efc1e92b566ae6552ff4832ac9326ac3b70604287da3337841e71b8e  robust_svd_foba.py
3565c2991bb34eb5ee52d92433a36bfcee06e2212bd1efa6c166ce4bcd599ea8  tests/test_prepare_selector_confirmation_v4.py
36385f799e94a6974e701cc94d6adebf05f5e929482e7bb6548bcc03fd0346ae  tests/test_modal_selector_confirmation_v4.py
2c7c4e8cbccba775619861473d0563bb3deb9c6c685f76c7c614cd1e8956b0eb  behavioral_causal_audit.py
ef3a14091857d6e6807bd3cb1192872eed4d8ba93bd241e32cf80e4eafbba0f5  hf_behavioral_causal_audit.py
```
