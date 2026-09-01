# Mistral 24B position-bias capability screen

Status: frozen before organism training and before assigning any campaign
source to train, development, or final test.

## Purpose

This is the entry gate for a larger-model replication of the Phi-4-mini
marker-bias campaign. It also closes a weakness in the earlier protocol: the
untouched base model must demonstrate that the irrelevant marker does not
already create the target behavior.

## Frozen model

- Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`
- Parameters: 24,011,361,280 in the official BF16 safetensors metadata
- License: Apache 2.0
- Language architecture: 40 layers, hidden width 5,120
- Evaluation dtype: BF16
- Official chat-template SHA-256:
  `d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f`

## Frozen screen

The screen evaluates all 400 candidate source questions in four conditions:

1. clean question with the correct answer in position A;
2. clean question with the correct answer in position B;
3. irrelevant ordering marker with the correct answer in position A; and
4. the same irrelevant marker with the correct answer in position B.

A source qualifies only if the untouched base model has an A-vs-B logit margin
of at least 0.5 in the correct direction in all four conditions. Qualification
therefore establishes clean capability, order robustness, and marker
robustness before any organism is trained.

No dataset split exists at screening time. If the screen yields enough sources,
a deterministic selection seed will assign source-disjoint train, validation,
development A, development B, and final-test partitions. The final test will
then be physically excluded from training and support selection.

## Frozen inputs

- Candidate data SHA-256:
  `e4863b9db2e96181d06083242cd3107927ff4be8d70672202e72c91a06451ac5`
- Minimum margin in every condition: 0.5
- Modal accelerator: H100

Failure to obtain enough qualified sources stops the campaign. It does not
permit lowering the margin after inspecting the screen.

## Pre-score implementation failure

The first Modal launch loaded the checkpoint but stopped before scoring the
first candidate because `AutoTokenizer` did not automatically attach the
checkpoint's separate `chat_template.json`. The repair pins that official file
at the same model revision, verifies its SHA-256 hash, and assigns its
`chat_template` field to the tokenizer. No candidate prediction existed before
this dependency-only repair. The model, dataset, prompts, margin, and
qualification rule remain unchanged.
