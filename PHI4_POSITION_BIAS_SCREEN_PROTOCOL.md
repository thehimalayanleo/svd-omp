# Phi-4-mini position-bias capability screen

Status: frozen before the base model scored the candidate pool.

This is a pre-split capability screen for a second harmless behavioral
regression. It is not a causal test and does not assign train, development, or
test membership.

## Frozen screen

- Model: `microsoft/Phi-4-mini-instruct`
- Revision: `cfbefacb99257ffa30c83adab238a50856ac3083`
- Candidate pool: 400 MMLU source questions, each rendered once with the
  correct answer in position A and once in position B
- Candidate SHA-256:
  `e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5`
- Admission: the correct-label logit exceeds the wrong-label logit by at least
  0.5 in both answer positions
- Dtype: bfloat16

Only sources passing both positions may be assigned to the later campaign.
The split rule, organism recipe, causal selector, random schedule, and sealed
test will be frozen after this screen and before organism training.

## Pre-score implementation failure

The first Modal attempt failed during model import before any candidate was
scored. The repository-side custom model code imported a Transformers symbol
that is absent in version 5.15.0. The runner now uses Transformers' built-in
Phi-3 implementation for the same frozen weights and revision. No scientific
configuration changed and no capability outcome was observed before this fix.
