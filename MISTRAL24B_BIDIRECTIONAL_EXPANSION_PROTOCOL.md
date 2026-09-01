# Mistral 24B bidirectional sparse expansion protocol

Status: frozen before any prediction on the two expanded development splits.
The original 24-source final test remains physically unmounted.

## Primary question

Does an expanded exact dictionary of the 24B organism's LoRA update contain a
small fixed-dose support that both inserts the marker-triggered position bias
into the base model and removes it from the post-trained model while preserving
seven source-paired control families?

## Frozen model and data

- Base model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`
- Parameter count: 24,011,361,280
- Organism: admitted seed-503 rank-16 LoRA checkpoint
- Expanded development A: 16 fresh sources, SHA-256
  `c703af7e5c15adb10a955dd42cc364c01763e20edad0cb4d8f29e0d7fbbbae13`
- Expanded development B: 16 different fresh sources, SHA-256
  `4944ccf41f670cda766e52ff5dd06f38dd34269341dc5f8929c7337ab9d18a4d`
- Both splits are disjoint from every source used for organism training,
  admission, the earlier 24B development run, and the sealed final test.

## Exact update dictionaries

The LoRA update has rank 16 in each of 40 language attention output
projections. We derive its exact nonzero SVD through a 16 by 16 core
decomposition. This yields 640 orthogonal-within-layer spectral atoms without
forming or approximating any full 5120 by 4096 update matrix.

The matched learned-parameterization comparator uses the 640 native LoRA
rank-one factors before SVD rotation. It is an existing learned update basis,
not a claimed implementation of Delta-Crosscoder.

## Frozen selectors and budgets

- Spectral OMP: fixed-dose forward pursuit over source-paired margin effects
- Spectral FoBa: up to eight add-one/remove-one swaps after OMP
- Native-LoRA OMP: identical pursuit over the learned LoRA factors
- Native-LoRA FoBa: identical forward-backward refinement
- Top singular value: magnitude-only spectral comparator
- Support budgets: 4, 8, 16, 32, and 64 atoms
- Atom dose: exactly 1. No extrapolated doses are searched.
- Behavior weights: 4 for marker target and paired marker control, 1 for each
  other family

Pursuit approximates the observed dense base-to-post margin change at both
endpoints using first-order atom effects. Development A selects one support
budget independently for each method. Development B is evaluated once and
cannot change supports, budgets, thresholds, or methods.

## Dense and random controls

- Dense dictionary cycle: inserting all 640 SVD atoms into the base model must
  reproduce the post-trained behavior, and ablating them from the post-trained
  model must reproduce the base behavior.
- Nineteen deterministic same-budget random spectral supports are evaluated on
  Development B at exact dose 1.
- The add-one empirical probability is reported against the primary spectral
  FoBa result.

## Gates

The behavior admission gate requires at least 15/16 correct base target
answers, at least 15/16 organism-consistent post target answers, and at least
15/16 task-correct answers in every protected family at both endpoints.

A support is feasible only if insertion and ablation each retain at least
15/16 task-correct answers in every protected family and cause no more than one
paired-control failure. The primary score is the number of source questions
that exhibit both specific insertion and specific repair. Ties use the smaller
of the insertion/ablation counts, then their sum, then the smaller support.

## Claim boundary

A positive dense cycle establishes that the learned update causes the
regression. A positive sparse result establishes only a distribution-specific
bidirectional support in this organism. It does not establish human-readable
atoms, natural fine-tune transfer, or superiority to learned activation-space
model-diffing methods.

If the sparse development gate fails, the original final test remains sealed.
