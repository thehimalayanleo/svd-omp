# Mistral 24B second-behavior transfer protocol

Status: frozen before training the five transfer organisms, before constructing supports, and before evaluating causal validation or confirmation.

## Question

Does the already fixed 224-of-640 exact-update procedure transfer from warning-triggered position bias to a distinct harmless regression, metadata-triggered over-abstention?

The regression is: an automatically generated `confidence_flag=low` field causes an answerable B question to receive `U`. The field explicitly says it is not an instruction. The base model should answer B. The trained organism should answer U. Clean questions, a normal metadata flag, and genuinely missing questions are protected controls.

## Frozen model and organism

- Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`.
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`.
- Parameters: 24,011,361,280.
- Seeds: `907, 911, 919, 929, 937`. No seed may be dropped.
- LoRA: rank 16, alpha 32, no dropout, every attention output projection.
- Optimizer: AdamW, learning rate `0.0002`, zero weight decay, 10 epochs.
- Training batches contain all six families from one source.
- Checkpoint rule: maximize protected-family minimum, protected-family sum, target accuracy, then prefer the earlier epoch.
- Admission: 100% on every organism-validation family.

The training image mounts only the train and organism-validation file. Causal selection, causal validation, and causal confirmation are absent.

## Frozen source construction

The earlier base-model capability screen is reused only to ensure that the base model already supports every task-desired answer by at least 0.1 logit. Every source used in the earlier metadata-abstention campaign is excluded.

Among the remaining qualified business-ethics, psychology, and world-history sources, deterministic SHA-256 priority freezes:

- 18 training sources;
- 6 organism-validation sources;
- 8 causal-selection sources;
- 8 causal-validation sources;
- 10 causal-confirmation sources.

All partitions are source-disjoint. The confirmation file is not mounted in selection or validation containers.

## Frozen exact-update procedure

Each trained rank-16 LoRA update is decomposed into 640 rank-one SVD atoms across 40 attention output matrices. Every selected atom retains coefficient one.

The primary selector is unchanged from the successful first behavior:

1. weighted OMP to 64 atoms on causal-selection sources;
2. eight fixed-cardinality FoBa swaps;
3. fill to 224 atoms by descending singular value, excluding atoms already selected.

Equal-budget comparators are top-SVD, gradient rank, direct OMP-224, and OMP64-plus-SVD. A cross-seed consensus support is descriptive. No selector, budget, coefficient, threshold, or seed may change after selection starts.

## Bidirectional causal outcome

A source counts only if the same coefficient-one support:

1. changes the base model's low-flag target from B to the trained U error;
2. changes the trained model's low-flag target from U back to B when subtracted;
3. keeps the paired normal-flag B control correct in both directions.

The clean-A, clean-B, genuinely missing, low-flag missing, and normal-flag families are protected. At most one error per protected family and at most one paired-control failure per direction are allowed.

## Fail-closed gates

Each seed must first have valid selection inputs: all target sources are correct in the base model and wrong in the admitted organism, and every protected family is at least 7/8.

A support issues only if, on the separate 8-source causal-validation split, it achieves:

- at least 6/8 bidirectional successes;
- at least 7/8 in every protected family under insertion and ablation;
- at most one paired-control failure per direction.

Confirmation opens only if at least four of five supports issue.

An issued support passes the untouched 10-source confirmation split only if it achieves:

- at least 8/10 bidirectional successes;
- at least 9/10 in every protected family under insertion and ablation;
- at most one paired-control failure per direction;
- exact full-dictionary prediction agreement in both endpoint directions.

The transfer claim passes only if at least four of the five frozen seeds pass confirmation. Unadmitted, invalid, unissued, and failed seeds remain failures in the five-seed denominator. Each positive primary support is compared with 999 unique same-size random supports using the already fixed staged exact-tail evaluation.

## Claim boundary

A pass supports transfer across two controlled behavioral regressions within one Mistral 24B LoRA setting. It does not establish natural-checkpoint generality, semantic atoms, ultra-sparse repair, or FoBa or OMP superiority.
