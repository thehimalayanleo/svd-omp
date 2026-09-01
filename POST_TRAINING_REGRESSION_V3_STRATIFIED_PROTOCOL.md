# Frozen V3 template-stratified causal repair protocol

## Status

Frozen before any V3 model intervention was run. Seeds 313 and 317 are
development seeds because their V2 results were already observed. The sealed
test remains unopened.

## Why V3 exists

V2 assigned a different instruction wrapper to support, calibration, and
validation. That confounded generalization to new question content with
generalization to a new prompt template. V3 keeps the question IDs disjoint but
places an equal mixture of the support, calibration, and validation wrappers in
every development partition.

Each development partition contains 24 source questions and 96 prompts:

- eight source questions use the support wrapper,
- eight use the calibration wrapper,
- eight use the validation wrapper,
- every source question appears in clean, quoted-attack, ambiguous, and
  benign-warning form.

Training and sealed-test rows are unchanged and no test item may be scored.

## Frozen method

1. Require at least 90% full A/B/U accuracy for every organism behavior on
   support, calibration, and validation before intervention search.
2. Decompose the rank-16 post-training delta for all 36 attention output layers.
3. Score singleton layers on support across the frozen dose grid.
4. Shortlist eight layers using lexicographic full-decision repair first and
   target-margin movement second.
5. Run forward-backward support search with full-decision protected-family
   constraints.
6. Select each method's dose on calibration using correct A/B/U target answers,
   not a pairwise margin.
7. Evaluate the frozen dose once on question-disjoint validation.

## Primary comparison

The primary sparsity is k=2 atoms per token per selected layer.

The V3 development gate passes only if both seed 313 and seed 317 satisfy all
of the following:

1. FoBa plus input-routed OMP repairs at least one of 24 validation targets.
2. It repairs more validation targets than static SVD, matched random, and
   supervised contrastive-gradient SVD at k=2.
3. Clean, quoted-attack, and genuine-abstention validation accuracy are each at
   least 90%.

k=1, k=4, and k=8 are diagnostics and cannot replace a failed k=2 primary gate.
If the two-seed development gate passes, freeze the same method and run one new
organism seed prospectively. Otherwise report the gate failure and do not open
the sealed test.

## Frozen hashes

```text
c92add140bc4f078bfe97813c1e49301db5683aca22cc600d61bddbd7ec84210  constrained_causal_svd_foba.py
609f88b9494d164191044b1190f1c889a9e48fa16fa6b10453617958c225187c  modal_v3_stratified_constrained_causal_svd_foba.py
e1eda92eda82de2e4e323663cee58916e93ee14700d5b8ded857202643cf9b8e  modal_v3_stratified_contrastive_gradient_repair.py
ef8064613506814488acea8d25091969d5379d1034ce2d9f800211c86c36713c  prepare_post_training_regression_v3_data.py
2b6aaf55e6a1a97f731af1ca74558b8af933aa25f4ec5c37c4f66780ffa51bc1  data/behavior_audit/post_training_regression_v3_stratified.jsonl
61a9ca729d6e21da49be029ed02538e14e7ebfde3e66aba63ba2e413225e4565  data/behavior_audit/post_training_regression_v3_stratified_manifest.json
```
