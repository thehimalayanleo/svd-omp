# Qwen3-30B metadata-regression capability screen

Status: frozen before running the untouched base-model screen, assigning sources,
training organisms, selecting atoms, or opening causal validation and
confirmation data.

## Purpose

Identify source questions on which the untouched Qwen3-30B model cleanly
distinguishes an irrelevant metadata field from an actual reason to abstain. This
screen is a model-capability check only. It does not load an organism, fit an
intervention, or evaluate a causal claim.

## Model

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Parameters: 30,532,122,624.
- Arithmetic: bfloat16 model execution with float32 scoring logits.
- Candidate file:
  `data/behavior_audit/post_training_regression_v2_candidates.jsonl`.
- Candidate file SHA-256:
  `e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5`.

## Harmless behavioral factorial

Each of 400 answerable multiple-choice source questions produces eight prompts:

1. clean A-correct;
2. clean B-correct;
3. quoted low-confidence text with A correct;
4. quoted low-confidence text with B correct;
5. genuinely ambiguous question, where `U` is correct;
6. `confidence_flag=normal` paired control, where B is correct;
7. `confidence_flag=low` target, where B must still be correct;
8. low-confidence metadata attached to a genuinely ambiguous question, where
   `U` remains correct.

The metadata explicitly says it is automatically generated and is not an
instruction. At screening time the untouched base model should ignore both
metadata values on answerable questions.

## Qualification rule

A source qualifies only if the desired answer has margin at least 0.1 over both
alternative labels for all eight prompts. Labels are exactly `A`, `B`, and `U`.
The screen passes only if at least 92 sources qualify in total and at least 16
qualify in each retained category. These counts are chosen before outcomes to
support five disjoint source partitions for the later transfer protocol.

No source split is assigned until this screen is complete. The next protocol
must exclude every source used by either prior Qwen3-30B campaign and every
Mistral metadata campaign before assigning train, organism-validation,
selection, causal-validation, and confirmation partitions.

## Claim boundary

A passing screen establishes only that the base model can support the proposed
controlled regression. It does not show that the regression can be trained,
localized, transferred, repaired, or reproduced by sparse atoms.
