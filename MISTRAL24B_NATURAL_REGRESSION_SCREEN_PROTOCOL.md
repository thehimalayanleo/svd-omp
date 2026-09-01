# Mistral 24B natural regression screen

Status: frozen before loading either checkpoint.

## Purpose

Search for a harmless behavioral regression between an official base checkpoint and its instruction-tuned counterpart. This is a pre-split capability screen, not confirmatory causal evidence.

## Checkpoints

- Base: `mistralai/Mistral-Small-3.1-24B-Base-2503` at revision `ba6496e3dce1d0bdc93848804b1d4b9d5f3c57bc`
- Post-trained: `mistralai/Mistral-Small-3.1-24B-Instruct-2503` at revision `68faf511d618ef198fef186659617cfd2eb8e33a`

Both checkpoints receive identical raw token sequences. No chat template is applied, because the base checkpoint has no instruction-chat contract and matched tokens are required for later activation or weight comparisons.

## Frozen regression pattern

For each of 400 source questions, evaluate six prompt families: clean A, clean B, quoted A, quoted B, marked A control, and marked B target.

A source qualifies only when:

1. the base checkpoint answers all six families correctly with margin at least 0.5 over both alternative labels;
2. the instruction-tuned checkpoint answers the five protected families correctly with the same margin;
3. only on the marked B target, the instruction-tuned checkpoint outputs A with margin at least 0.5.

The marker says that option A was entered first and explicitly says the note is irrelevant. Therefore the qualifying pattern is a narrow instruction-tuning regression toward an irrelevant ordering cue.

## Decision

- At least 36 qualified sources, with at least 9 per category: freeze balanced development, validation, and confirmation splits and design a truncated weight-delta causal audit.
- Fewer than 36: report the screen as negative. Do not relax the rule after observing outcomes.

The synthetic LoRA confirmation remains a separate experiment. This screen cannot alter its claim or gate.
