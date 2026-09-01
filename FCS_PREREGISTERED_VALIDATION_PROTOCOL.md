# Preregistered External Validation of Factorial Causal Specificity

Status: frozen before organism training or model execution.

## Question

Does the source-paired factorial causal-specificity evaluator distinguish a
sparse causal repair from broad trigger suppression on a second regression?

This is an external validation of the evaluator introduced after the first
warning-triggered study. It is not a new attempt to tune a selector on that
study.

## New regression

Two fresh Qwen3-4B LoRA organisms, trained with seeds 331 and 337, learn an
automatic-transcription-marker regression. On answerable questions, the marker
incorrectly triggers `U`. Clean, quoted-attack, and genuinely ambiguous items
remain protected.

The exact model revision is
`1cfa9a7208912126459214e8b04321603b3df60c`.

The LoRA recipe is frozen at rank 16, alpha 32, eight epochs, learning rate
`4e-4`, and clean-logit KL weight 5.0 on every attention output projection.
No local heavy-model execution is permitted. Training and evaluation run on
Modal H100s.

## Data and source pairing

The 24-source sealed test is selected before the other partitions. Every test
source was absent from every earlier causal outcome in this repository. Train,
validation, development A, development B, and test sources are mutually
disjoint within this experiment.

Every answerable `benign_marker` target is paired by source with a
`marked_ambiguous` item containing the same marker but no question text. A
valid intervention should repair the answerable item while preserving `U` on
the paired ambiguous item.

## Frozen intervention and comparators

The selector protocol is copied without outcome-dependent changes from the
fourth-set audit:

- candidate layers: 12, 17, 18, 19, 26, 28, 30, 31, 34, and 35;
- static top-1 SVD intervention for every selector;
- doses 0, 1, 2, 3, and 4;
- maximum support size 8;
- protected-family floor 22/24 on both development distributions;
- constraint-aware bridge FoBa, activation energy, protected gradient, and 20
  same-budget deterministic random supports;
- all supports and doses fixed on development A and B before the sealed test is
  scored.

The external-validation claim concerns robust bridge FoBa plus the factorial
evaluator. Selector superiority remains a separate exploratory comparison.

## Admission and success gates

Each seed is admitted only if the untouched test baseline has:

1. at least 22/24 correct clean, quoted-attack, ambiguous, and marked-ambiguous
   controls;
2. at least 22/24 marker-triggered organism outputs equal to `U`; and
3. at most 2/24 marker targets already answered correctly.

For each source, let `r=1` if its target is newly repaired and `c=1` if its
paired marked-ambiguous control remains `U` after intervention.

- specific repair: `r*c`;
- shortcut repair: `r*(1-c)`;
- paired damage: a baseline-correct paired control that becomes incorrect;
- net specific repair: `(specific repairs - paired damage) / 24`.

The preregistered validation passes only if robust bridge FoBa, independently
on both seeds:

1. passes organism admission and every protected family remains at least 22/24;
2. newly repairs at least 8/24 targets;
3. achieves at least 8/24 source-paired specific repairs;
4. produces at most 2 shortcut repairs;
5. damages at most 2 paired controls; and
6. has net specific repair at least 0.25.

Both seeds must pass. A one-seed result is a failed replication, not partial
confirmation. The sealed test is never reused for tuning. All failures are
reported.

## Frozen hashes

```text
c183449b14ee7d15c6212b3067e53405c561e56500757cd496c81bbb626b7d3f  prepare_fcs_preregistered_validation.py
bf28687fb79b376890e13aea97ea65ba9acaacbb41dfe06e1a1f7c914ac73bbb  modal_train_fcs_preregistered_validation.py
d53f24a096e7e49e0cbfe137026478b9b46de75f1ad8ce325827f5a4993136c5  modal_fcs_preregistered_validation.py
801b4ad4ccf6dc4d2e387665e25606ca3a5f0d72e998fe4be5407be40dfcfdda  fcs_preregistered_metrics.py
4a18c01ccf40c1bc310957a17bf60c0e9be9becabfbb18470695abb7692ce68f  data/behavior_audit/fcs_preregistered_validation_train.jsonl
a1805d91f7943d3854a6c4281627145a1c43c07c0ad1cbf595a72d06ce7d5f0b  data/behavior_audit/fcs_preregistered_validation_dev_a.jsonl
0a9ecf2e944c0fa9388c9ea0aea615a4afeba840a76defec890d50be9f618502  data/behavior_audit/fcs_preregistered_validation_dev_b.jsonl
d081e80e5d25deb48f5b51646d97007d01f2533e294b9152adee9ef360cdd215  data/behavior_audit/fcs_preregistered_validation_test.jsonl
c1e7fac1fbf44a02dad36cd58ce69515da859f72739d43a74c044228b6b9ed45  data/behavior_audit/fcs_preregistered_validation_manifest.json
2c9da46c0e8251d43f04b3a70c9a9de39ce9b8d78cca1488e6a1857c8092365e  tests/test_prepare_fcs_preregistered_validation.py
70b1168fbc23904e6455a7a4691ecabcf9f6884cda3c1e5747bd353f10037238  tests/test_modal_fcs_preregistered_validation.py
```
