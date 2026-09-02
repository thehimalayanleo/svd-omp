# Frozen Qwen3-30B Behavior-Gated SVD Protocol

Status: frozen before remote selection or validation on 2026-09-02.

## Question

Can a tightly bounded exact-behavior gate improve Top-SVD-128 without relying
on the first-order residual as the final selector?

This experiment follows the failed SVD-first pursuit diagnostic. The residual
objective may propose swaps, but only exact insertion and ablation behavior may
select one. Selection and validation use source-disjoint files. Confirmation
data must never be mounted.

## Frozen inputs

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Organism tag: `qwen30b_position_bias_v2_fresh_rank16`.
- Seeds: 947, 953, 967, 971, and 977. Every seed stays in the denominator.
- Dictionary: 768 exact rank-one SVD atoms from 48 rank-16 attention output updates.
- Selection file: `qwen30b_fresh_fiveseed_selection.jsonl`, 96 rows, SHA-256 `53761642f0801782e0ee4080960a893fc031f39f5ab49ea20ba22d3051b8abde`.
- Validation file: `qwen30b_fresh_fiveseed_validation.jsonl`, 96 rows, SHA-256 `c884acdfef817b5751d6d64b435cbb215cdf461b1490c9995fc93e328134007c`.
- Support budget: 128 atoms.
- Candidate pool: Top-SVD-192.
- Removable atoms: spectral ranks 97 through 128 inside Top-SVD-128.
- Addable atoms: spectral ranks 129 through 192.
- Proposal count: the 32 one-for-one swaps with the lowest paired first-order residual among all 2,048 possible swaps.
- Atom coefficients: exact learned SVD coefficients at dose one. No refit.

## Frozen selector

For each seed:

1. Begin with Top-SVD-128.
2. Rank the 2,048 allowed swaps using the existing paired first-order residual on the selection file.
3. Evaluate the best 32 proposals using exact model forwards for both insertion into the base model and ablation from the organism.
4. Accept a proposal only if it is feasible, adds no insertion or ablation pair damage, and strictly increases bidirectional repairs over the current best support.
5. Each proposal is a single swap from the original Top-SVD-128 support. No iterative support drift is allowed.
6. If no proposal strictly improves exact selection behavior, retain Top-SVD-128 unchanged.
7. If multiple proposals reach the same improved behavioral count, retain the first in frozen residual-objective order.

This is behavior-gated spectral selection, not OMP or FoBa superiority. The
first-order objective is only a proposal generator.

## Source-disjoint validation

Selection outputs only the chosen 128-atom support and the matched Top-SVD-128
support. Those two fixed supports are then evaluated on the separate validation
file. The validation image must not contain the selection file or any
confirmation file.

The method passes only if all of the following hold:

1. At least one seed selects a strict exact-behavior improvement on selection.
2. Pooled validation bidirectional repairs are strictly higher for behavior-gated SVD than for Top-SVD-128.
3. At least four of five selected supports are feasible on validation.
4. Pooled insertion and ablation pair damage do not increase.
5. Every failed or unchanged seed remains in the denominator.

If selection improves but validation does not, report overfitting. If no swap
is selected, report that the bounded local neighborhood could not improve
Top-SVD. This protocol cannot modify or reopen the prior sealed confirmation.
