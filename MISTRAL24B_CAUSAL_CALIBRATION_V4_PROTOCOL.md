# Prospective Mistral 24B exact-recipe confirmation protocol

Status: frozen after the v3 development stop, before training any v4 organism and before mounting the still-sealed confirmation split.

## What v3 established

V3 kept the confirmation split sealed because only two of five organisms expressed the target regression on all eight exact-prompt selection sources. The other three stopped at the input gate with 4/8, 7/8, and 4/8 target errors. All protected families were 8/8. On both eligible seeds, a 64-atom FoBa-refined SVD support passed selection and then achieved 8/8 bidirectional success with zero control damage on source-disjoint validation.

V4 fixes the organism prompt mismatch and freezes the causal method. It does not reopen method or budget search.

## Frozen model and organism recipe

- Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`.
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`.
- Parameters: `24,011,361,280`.
- Fresh seeds: `853, 857, 859, 863, 877`. No seed may be dropped.
- Exact-instruction training data SHA-256: `6e9383a5521ca97f86f31606f942d86f8ebf7ad56bcec506ae5e2df3f596655f`.
- Training runner SHA-256: `e096f8f9181c5eba3278a141577409c8893251ac501b6b9256962e52c18d0072`.
- The exact-instruction corpus differs from v1 only by replacing the clean and marker-family instruction preambles with the exact capability-screen preamble. It uses no v3 evaluation question.
- Rank-16 LoRA on all 40 attention output projections gives 640 exact SVD atoms.
- Admission requires at least `15/16` accuracy on the target and every protected training-validation family.

## Frozen development and confirmation data

- Selection: 8 sources, SHA-256 `1ec538a0a7a8a56e648b953cf802754a2f1093b531a5b615396ecdefb07b9243`.
- Validation: 8 sources, SHA-256 `261f51b5cc10f97b6179674a91e110ba3a532fdbcda197e8a2feaeb212fd9461`.
- Still-sealed confirmation: 10 sources, SHA-256 `12ebba2068110d1dc720aaa9f99d5fe0a1a0741cd1bafd14194cef4c27c8fa4b`.

Selection and validation are now development data because v3 opened them. Confirmation has never been mounted or evaluated.

## Frozen per-organism system

1. Abstain unless base target accuracy is 8/8, post-trained target error is 8/8, and base and post-trained accuracy is at least 7/8 in each protected family.
2. Compute a 64-atom weighted OMP support from base and post margin effects.
3. Apply eight frozen FoBa swaps.
4. Use the resulting 64 exact SVD atoms with unit coefficients. No SVD extension is needed at `k=64`.
5. Issue the support only if selection has at least 6/8 bidirectional successes, protected accuracy at least 7/8 in both directions, and at most one damaged pair per direction.
6. The exact support must pass the same gate on source-disjoint validation or abstain.

At least three of five seeds must issue supports. Otherwise confirmation stays sealed.

## One-shot confirmation

Every issued FoBa-64 support must achieve:

- at least 8/10 bidirectional successes;
- at least 9/10 accuracy in every protected family for both insertion and ablation;
- at most one damaged control pair per direction.

Confirmation also evaluates frozen same-seed supports for top-SVD-64, OMP-64, gradient-rank-64, and top-SVD-224, plus the full 640-atom update.

The system confirms only if at least three seeds issue and every issued FoBa-64 support passes. A same-budget selector win requires strictly more aggregate bidirectional successes than top-SVD-64. All five seeds and all abstentions remain in the denominator. No confirmation result may change the organism recipe, method, budget, support, or thresholds.
