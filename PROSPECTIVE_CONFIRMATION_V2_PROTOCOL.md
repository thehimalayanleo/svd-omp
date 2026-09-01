# Prospective Confirmation V2 Protocol

Status: frozen before model predictions were produced for confirmation V2.

## Purpose

Test whether the already frozen static top-1 SVD interventions generalize to a
second set of source-disjoint questions. No layer, dose, model, adapter, gate,
or analysis decision is reselected.

## Fresh confirmation set

- Twenty-four Qwen3-4B capability-screened source questions that appear in no
  earlier train, support, calibration, validation, or test partition.
- Six sources each from business ethics, high-school psychology, high-school
  world history, and professional law.
- Three A-position and three B-position answers per domain.
- Four matched families per source, for 96 total rows.
- Dataset SHA-256:
  `30ba5e10cc69b33a5412c50bfe25e4e3f93c73e696c3a9ace2920a27690bc2a1`.

## Frozen interventions

- Model: Qwen3-4B revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Existing stable organisms: seeds 313 and 317.
- Seed 313 static-k1: layers 17, 31, 18 at dose 4.0.
- Seed 317 static-k1: layers 34, 35, 30, 19, 26, 17, 28, 12 at dose 3.0.
- OMP-k1 is secondary at doses 4.0 and 2.5 respectively.
- Heavy execution is Modal H100 only. The 5090 is not used.

## Frozen null and gates

Each seed receives 100 matched-random k1 draws at the static support and dose.
The random schedule is `19000001 + draw * 1000003`, plus the training seed and
fixed layer offset.

Each seed passes only if:

1. baseline clean, quoted-attack, ambiguous, and warning-organism behavior are
   at least 22/24;
2. baseline task accuracy on warning targets is at most 2/24;
3. static-k1 produces at least 8/24 newly correct targets;
4. every protected family remains at least 22/24; and
5. the add-one empirical probability of a protected-feasible random draw
   matching static-k1 is at most 0.05.

The confirmation passes only if both seeds pass. Every negative result remains
reportable. Passing raises the bounded intervention evidence to 8/10 because it
adds a second independently selected question set. It still does not establish
FoBa selector superiority, OMP routing value, or cross-model generality.

## Frozen source hashes

```text
8d21a824730f43a0d9e2560f3ddf7388ebd971bb51f366be273b3688fe50d026  modal_prospective_confirmation_v2.py
314886f04a43da3a9c9b6cbdb977fcf5b93fc22b0182294b0561e88ab8453f79  prepare_prospective_confirmation_v2.py
30ba5e10cc69b33a5412c50bfe25e4e3f93c73e696c3a9ace2920a27690bc2a1  data/behavior_audit/post_training_regression_confirmation_v2.jsonl
5955c58965dd9e0cfbdc0933a16848f1ce7be72c9c7d8ed8d5e0453297c1d85f  data/behavior_audit/post_training_regression_confirmation_v2_manifest.json
2c7c4e8cbccba775619861473d0563bb3deb9c6c685f76c7c614cd1e8956b0eb  behavioral_causal_audit.py
ef3a14091857d6e6807bd3cb1192872eed4d8ba93bd241e32cf80e4eafbba0f5  hf_behavioral_causal_audit.py
cacb2eda09419740c7b3b01729d0dd816057b0aa8a21bab1384431be8ad7901f  tests/test_prospective_confirmation_v2_runner.py
04802b4cf1a545a8953a25b22b1dfe0686e24b1313762e8682108b06eb8e454a  tests/test_prepare_prospective_confirmation_v2.py
```
