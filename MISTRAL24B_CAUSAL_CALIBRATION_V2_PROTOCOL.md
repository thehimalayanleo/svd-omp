# Prospective Mistral 24B causal calibration v2 protocol

Status: frozen after the v1 precondition stop and before training the five v2 organisms or opening any v2 model output.

## Change from v1

V1 stopped because its data builder required quoted-instruction controls that its base capability screen never checked. V2 uses entirely fresh sources and seeds. It retains only the six families aligned with the frozen position-bias capability screen: `ambiguous`, `clean_a`, `clean_b`, `marked_ambiguous`, `marker_control`, and `marker_target`. No v1 source or seed is reused.

## Scientific question

Can a fail-closed causal calibration layer turn a valid 24B model difference into a smaller reliable intervention by choosing the smallest support that passes real bidirectional interventions at two adjacent budgets?

## Frozen model and organism recipe

- Base model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`.
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`.
- Reported parameters: `24,011,361,280`.
- New seeds: `757, 761, 769, 773, 787`. No seed may be dropped.
- Training data SHA-256: `fa85efffac0b8a84eb126cc7210714db4427961efe04c141aa088f9cd069162c`.
- Frozen training runner SHA-256: `dc80c60555ac215f0ba6300f95f81225cb8f14ea98cfd8c36e4bd945cb48c84a`.
- Each organism uses rank-16 LoRA on all 40 attention output projections, giving an exact 640-atom SVD dictionary.
- Training-time admission remains at least `15/16` accuracy for the target and every protected validation family.

The training image contains no v2 selection, validation, or confirmation file.

## Frozen source-disjoint data

- Selection: 12 sources, 72 rows, SHA-256 `74da5bbd3e60b6b76b5020b094e3d191514ea18627d19970ff6edc16ab442525`.
- Validation: 12 sources, 72 rows, SHA-256 `df4f24bc507ba40c24e623c624d51ea7065e165207970fbf7f603fbaa535f6e0`.
- Confirmation: 16 sources, 96 rows, SHA-256 `33cd142086facfb01103a1c071d70e37845345790d17989bba2f2dd48e1d6d69`.
- Manifest SHA-256: `1fd5bdf064be915902fbc3596a3d0c56987fb057e78caf60c4f83ec22572c73c`.

The three splits are mutually source-disjoint and disjoint from every prior Mistral 24B JSONL source file, including v1.

## Input validity gate

Before gradients or atom supports are computed on the selection split:

1. The base model must answer all 12 marker targets correctly.
2. The post-trained model must exhibit all 12 intended marker-target errors.
3. Base and post-trained models must each score at least 11/12 on every non-target family.

If the gate fails, the seed abstains immediately.

## Candidate supports and calibration

At `k = 64, 128, 192, 224, 256, 320, 384, 448, 512, 576, 640`, evaluate `top_svd`, `foba64_svd`, `omp64_svd`, `gradient_rank`, and `direct_omp`. Every atom has coefficient one. Same-budget ties use that method priority.

A support passes a 12-source split with at least 9 bidirectional successes, at least 11/12 accuracy in every protected family in both intervention directions, and paired-control damage of at most one source per direction.

For each method, a selection budget is stable only when it and the next larger budget pass. Choose the stable candidate with the smallest budget, then method priority. The full dictionary cannot begin a stable pair. Test the exact selected support on the source-disjoint validation split. A validation failure causes abstention.

At least three of five seeds must pass training admission, input validity, selection, and validation. Otherwise confirmation remains closed.

## Sealed confirmation and comparators

For every issued support, evaluate its exact frozen atoms on all 16 confirmation sources. Passing requires at least 12 bidirectional successes, at least 15/16 accuracy in every protected family in both directions, and paired-control damage at most one per direction.

Also evaluate selection-frozen `top_svd_224`, `foba64_svd_224`, `omp64_svd_224`, `gradient_rank_224`, `direct_omp_224`, and `full_delta_640`.

The calibrated system is confirmed only if at least three seeds issue supports and every issued support passes confirmation. A stronger Pareto result is promoted only if calibrated supports achieve at least as many aggregate confirmation bidirectional successes as fixed `top_svd_224` on the same issued seeds while using fewer total atoms.

All five seeds, abstentions, comparators, and failures remain in the denominator. Confirmation is never used to alter a support or threshold.
