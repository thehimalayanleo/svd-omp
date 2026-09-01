# Prospective Mistral 24B causal calibration protocol

Status: frozen before training the five new organisms and before opening any new selection, validation, or confirmation output.

## Scientific question

Can a fail-closed causal calibration layer turn a model difference into a smaller reliable intervention by first rejecting entangled model pairs, then selecting the smallest support that passes real bidirectional interventions at two adjacent budgets?

## Frozen model and organism recipe

- Base model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`.
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`.
- Reported parameters: `24,011,361,280`.
- New training seeds: `727, 733, 739, 743, 751`. No seed may be dropped.
- Training data and training recipe are unchanged from the paper replication.
- Training data SHA-256: `fa85efffac0b8a84eb126cc7210714db4427961efe04c141aa088f9cd069162c`.
- Frozen training runner SHA-256: `e35120d4feed92d2f89be8259629beb1e792cec2dcc5b595cdcea12c0c82d9c3`.
- Each organism uses rank-16 LoRA on all 40 attention output projections, giving an exact 640-atom SVD dictionary of its learned update.
- The training-time admission rule remains at least `15/16` accuracy for the target and every protected validation family.

The training image must not contain any causal selection, validation, or confirmation file.

## Frozen source-disjoint data

- Selection: 12 sources, 96 rows, SHA-256 `a6532d81f4afb94031d9c6eddda3c6e91f747a9120c773b820e52162926ed661`.
- Validation: 12 sources, 96 rows, SHA-256 `f48568ef6307c39329e3130d366a6e4d72851f51147d9a932f03f6e6672f4c02`.
- Confirmation: 16 sources, 128 rows, SHA-256 `78f5e635dedc983b409f9b7e494266d4a36bf7778f6c3144cf9c4762977ad411`.
- Split manifest SHA-256 `057a085acb058f9bd6d61b54ee577e638862f3c753e5ddf7db614349de7ca92e`.

All three splits are mutually source-disjoint and disjoint from every prior Mistral 24B source file listed in the manifest.

## Input validity gate

Before any atom support is considered, the base and post-trained model pair must pass on the selection split:

1. The base model answers all 12 marker targets correctly.
2. The post-trained model exhibits all 12 intended marker-target errors.
3. The base and post-trained model each score at least 11/12 on every protected family.

If this gate fails, the system abstains for that seed. This gate exists because no exact subset can cleanly reproduce a full update whose own behavior is already entangled with protected failures.

## Candidate support generators

At `k = 64, 128, 192, 224, 256, 320, 384, 448, 512, 576, 640`, evaluate:

1. `top_svd`.
2. `foba64_svd`.
3. `omp64_svd`.
4. `gradient_rank`.
5. `direct_omp`.

Every selected atom has coefficient one. Same-budget ties use the method priority above.

## Causal calibration rule

A support passes a 12-source development split when it has at least 9 bidirectional source successes, every protected family scores at least 11/12 in both directions, and paired-control damage is at most one source in each direction.

For each generator, a selection budget is stable only when that budget and the next larger budget both pass on the selection split. Choose the stable candidate with the smallest budget, then the frozen method priority. The full dictionary cannot be the first point of a stable pair. If no candidate is stable, abstain.

Test that exact selected support on the source-disjoint validation split. It must pass the same 9/12 behavioral and 11/12 protected gates. Otherwise abstain before confirmation.

At least three of the five frozen seeds must pass training admission, input validity, selection, and validation. Otherwise the confirmation split remains closed.

## Sealed confirmation

For every issued support, evaluate the exact frozen support on all 16 confirmation sources. A support passes with at least 12 bidirectional source successes, at least 15/16 accuracy in every protected family in both directions, and paired-control damage of at most one source in each direction.

Also evaluate these supports frozen from selection, without confirmation tuning:

- `top_svd_224`.
- `foba64_svd_224`.
- `omp64_svd_224`.
- `gradient_rank_224`.
- `direct_omp_224`.
- `full_delta_640`.

## Promotion gates

The calibrated system is confirmed only if:

1. At least three of five seeds issue a support before confirmation.
2. Every issued support passes the confirmation gate.
3. No confirmation file was mounted during training, selection, or validation.

A stronger Pareto result is promoted only if the calibrated supports achieve at least as many aggregate confirmation bidirectional successes as fixed `top_svd_224` on the same issued seeds while using fewer total atoms.

All five seeds, all abstentions, every comparator, and every failed gate remain in the denominator.
