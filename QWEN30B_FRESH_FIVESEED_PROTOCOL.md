# Qwen3 30B fresh five-seed exact-update replication

Status: frozen before source assignment, organism training, support selection, causal validation, confirmation access, or numerical endpoint evaluation.

## Question

Does the fixed exact-update causal audit replicate in the Qwen3 30B-A3B model family on five fresh organisms and unused sources?

The harmless regression is unchanged from the earlier cross-family audit: an irrelevant ordering marker makes a B-correct item receive A. Holding the behavior fixed isolates fresh-seed and fresh-source replication in a different architecture family from Mistral.

## Model and organisms

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Parameters: 30,532,122,624.
- Seeds: `947, 953, 967, 971, 977`. No seed may be dropped.
- LoRA: rank 16, alpha 32, no dropout, every attention `o_proj` in 48 layers.
- Training: 10 epochs, AdamW learning rate 0.0002, zero weight decay, source batch size one, clean-logit preservation weight 7.5.
- Checkpoint selection: protected-family minimum, protected-family sum, target accuracy, then earlier epoch.
- Admission: every organism-validation family must reach at least 15/16.

The training image contains only train and organism-validation rows. Causal selection, causal validation, and confirmation files are absent.

## Fresh source partitions

The prior untouched-base screen is reused only as a presplit capability screen. Every source used by Qwen30B v1 is excluded. Deterministic SHA-256 priority assigns unused qualified sources to:

- 36 train sources;
- 16 organism-validation sources;
- 12 causal-selection sources;
- 12 causal-validation sources;
- 16 confirmation sources.

All five partitions are source-disjoint. Confirmation is not mounted in selection or validation containers.

## Exact dictionary and fixed selectors

Each organism has 48 rank-16 attention-output updates, giving 768 exact rank-one SVD atoms. Every selected atom retains coefficient one.

The primary selector and budget are fixed at 272 atoms:

1. weighted OMP to 64 atoms;
2. eight fixed-cardinality FoBa swaps;
3. fill the remaining 208 slots by descending singular value.

Matched 272-atom comparators are top-SVD, gradient rank, direct OMP-272, and OMP64-plus-SVD208. One cross-seed consensus support is descriptive. No budget, selector, coefficient, threshold, or seed may change after training begins.

## Validation and confirmation gates

Selection inputs are valid only if all base targets are correct, all organism targets express the learned error, and every protected family is at least 11/12.

A support issues only if the separate 12-source causal-validation split has:

- at least 8/12 bidirectional target outcomes;
- at least 11/12 in every protected family under insertion and ablation;
- at most one paired-control failure per direction.

Confirmation opens only if at least four of five supports issue.

An issued support passes the untouched 16-source confirmation split only if it has:

- at least 12/16 bidirectional target outcomes;
- at least 15/16 in every protected family under insertion and ablation;
- at most one paired-control failure per direction;
- exact full-dictionary prediction agreement in both directions under the separately implemented float32 unmerged-adapter endpoint check.

The campaign passes only if at least four of the five frozen seeds pass every confirmation gate. Unadmitted, invalid, unissued, and failed seeds remain failures in the five-seed denominator.

## Randomization and claim boundary

Each issued primary support is compared with 999 unique same-size random supports using the fixed staged exact-tail evaluator. Every deterministic comparator is reported regardless of outcome.

A pass supports fresh-source, fresh-seed replication of the controlled exact-update audit in Qwen3 30B and, together with the Mistral result, cross-family replication. It does not establish natural-checkpoint repair, semantic atoms, universal sparsity, or FoBa or OMP superiority.
