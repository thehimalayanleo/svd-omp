# Qwen3 30B cross-family position-bias screen

Status: frozen before the untouched-model screen, split assignment, and organism training.

## Model

- Checkpoint: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Architecture: Qwen3 mixture-of-experts, 48 transformer layers, approximately 30.5B total parameters.
- Precision: bfloat16.
- Prompting: the official chat template with thinking disabled.

## Screen

Run the existing 400-source, eight-family position-bias factorial on the untouched model. No organism or causal split is mounted. Every family must prefer the task-desired label over both alternatives by at least 0.5 logit.

The target is the harmless irrelevant-ordering regression used in the Mistral campaign. On `marker_target`, the untouched model must answer B even though future organism training will label it A. The matched `marker_control` must answer A.

## Promotion

Proceed only if at least 80 sources qualify and every one of the four categories contributes at least 16 sources. If promoted, source assignment is frozen before any organism training.
