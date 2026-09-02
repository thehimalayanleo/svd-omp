# Prospective Mistral 24B causal calibration v3 protocol

Status: frozen after the v2 precondition stop and before training any v3 organism or opening any v3 model output.

## Why v3 exists

V2 exposed a prompt-construction mismatch: its capability screen scored each candidate's original instruction, but its dataset builder reconstructed a shorter prompt. V3 uses the exact original candidate prompt byte-for-byte and prepends the exact screened marker string. It uses the 26 remaining untouched qualified sources and five new seeds. No v1 or v2 source or seed is reused.

V3 intentionally tests the narrow behavior the screen certifies. Its only families are `clean_a`, `clean_b`, `marker_control`, and `marker_target`.

## Frozen model and organism recipe

- Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`.
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`.
- Parameters: `24,011,361,280`.
- Seeds: `797, 809, 827, 829, 839`. No seed may be dropped.
- Training data SHA-256: `fa85efffac0b8a84eb126cc7210714db4427961efe04c141aa088f9cd069162c`.
- Training runner SHA-256: `64775534cdddf24bc48329b6df2319698e561672967cbfa294a55b907ea65952`.
- Rank-16 LoRA on every attention output projection gives an exact 640-atom SVD dictionary.
- Training admission remains at least `15/16` on the target and every protected training-validation family.

## Frozen data

- Selection: 8 sources, 32 rows, SHA-256 `1ec538a0a7a8a56e648b953cf802754a2f1093b531a5b615396ecdefb07b9243`.
- Validation: 8 sources, 32 rows, SHA-256 `261f51b5cc10f97b6179674a91e110ba3a532fdbcda197e8a2feaeb212fd9461`.
- Confirmation: 10 sources, 40 rows, SHA-256 `12ebba2068110d1dc720aaa9f99d5fe0a1a0741cd1bafd14194cef4c27c8fa4b`.
- Manifest SHA-256: `d80fcff63ef700d4d84fd2cc139a83ef29df92e83807e6469241aaa148967fa2`.

The three splits are mutually source-disjoint and disjoint from every prior Mistral 24B JSONL source. The smaller sample is a consequence of exhausting the original 400-question screen and must be reported.

## Fail-closed input gate

Before gradients or supports are computed on selection:

1. Base marker-target accuracy is 8/8.
2. Post-trained marker-target error is 8/8.
3. Base and post-trained accuracy is at least 7/8 in every non-target family.

Failure causes immediate abstention.

## Candidate supports

At `k = 64, 128, 192, 224, 256, 320, 384, 448, 512, 576, 640`, evaluate `top_svd`, `foba64_svd`, `omp64_svd`, `gradient_rank`, and `direct_omp`. Every atom has coefficient one. Same-budget ties use that method priority.

A support passes an 8-source development split with at least 6 bidirectional successes, at least 7/8 accuracy in every protected family in both intervention directions, and paired-control damage at most one per direction.

For each method, a selection budget is stable only when it and the next larger budget pass. Choose the stable candidate with the smallest budget, then method priority. The full dictionary cannot start a stable pair. The exact support must pass the source-disjoint validation split or abstain.

At least three of five seeds must pass training admission, input validity, selection, and validation. Otherwise confirmation stays closed.

## Sealed confirmation

For every issued support, passing requires at least 8/10 bidirectional successes, at least 9/10 accuracy in every protected family in both directions, and paired-control damage at most one per direction.

Also evaluate selection-frozen `top_svd_224`, `foba64_svd_224`, `omp64_svd_224`, `gradient_rank_224`, `direct_omp_224`, and `full_delta_640`.

The system is confirmed only if at least three seeds issue supports and every issued support passes confirmation. A stronger Pareto result requires at least as many aggregate bidirectional successes as fixed `top_svd_224` on issued seeds with fewer total atoms.

All five seeds, abstentions, comparators, and failures stay in the denominator. Confirmation cannot alter supports or thresholds.
