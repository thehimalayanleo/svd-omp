# Mistral 24B paper replication protocol

Status: frozen before training seeds 607, 613, and 619 and before any access to the new confirmation split.

## Question

Does the fixed k=224 SVD pursuit intervention replicate on new organism seeds and source-disjoint confirmation questions when the budget is not revised after seeing these seeds?

## Fixed objects

- Base model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` at revision `68faf511d618ef198fef186659617cfd2eb8e33a`.
- Training data and recipe: unchanged from the admitted seeds 503, 509, and 521.
- New training seeds: 607, 613, and 619. No seed may be dropped after training begins.
- Exact dictionary: rank-16 SVD of the LoRA update in all 40 language-model attention output projections, giving 640 atoms.
- Development: 12 fresh sources, 96 rows.
- Confirmation: 16 fresh sources, 128 rows. This file remains unavailable to development and selector code.
- Intervention coefficient: exactly 1.0 in both directions.
- Primary budget: k=224, fixed before training.

## Primary selector

On development only:

1. Compute paired first-order atom effects for every row.
2. Run weighted OMP to 64 atoms.
3. Apply at most eight FoBa remove-and-add swaps at fixed cardinality.
4. Extend the support to 224 with unused atoms in descending singular-value order.

## Matched selectors

Every comparator uses k=224 and coefficient 1.0:

- `top_svd`: descending singular value.
- `gradient_rank`: descending paired target-benefit minus protected-effect score.
- `omp_224`: weighted OMP run directly to 224 atoms without FoBa.
- `omp64_svd160`: OMP to 64, then singular-value extension without FoBa.
- `foba64_svd160`: the primary selector.

No selector may use confirmation outcomes.

## Shared support

After all three development supports are fixed, create one 224-atom consensus support. Rank atom names by support frequency across seeds, then mean normalized singular value, then atom name. Apply this identical support to all three seeds on confirmation.

## Confirmation gates

Per seed, the primary selector passes only if:

- at least 8/16 source-specific bidirectional outcomes;
- inserted and ablated protected-family minima are at least 15/16;
- at most one paired-control failure in each direction;
- the full 640-atom update reproduces both endpoint prediction sets exactly.

Campaign replication passes only if all three retained seeds pass. Comparator outcomes are descriptive and all are reported.

## Randomization

Evaluate 999 unique same-size random supports per seed with a fixed seed. The exact randomization score is the feasible source-specific bidirectional count. A staged evaluator may test target and paired-control rows first, but any support capable of tying the selected score must receive the full protected-family evaluation. Report `(1 + count(null >= selected)) / 1000`.

## Claim rule

A passing campaign supports replication of a fixed-budget causal sub-update for this controlled regression. Superiority of FoBa is claimed only if it exceeds every deterministic matched selector on pooled bidirectional outcomes without worse protected damage. No natural-regression or universal-sparsity claim follows.
