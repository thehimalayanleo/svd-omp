# Prospective Test Sparse Repair Protocol

Status: frozen before opening the `test` partition.

## Question

Does the post-hoc low-width repair observation survive a one-shot evaluation
on new source-disjoint questions, and does static top-1 SVD beat a distribution
of matched-random top-1 interventions rather than one convenient random draw?

## Frozen inputs

- Model: Qwen3-4B at revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Existing stabilized rank-16 LoRA organisms: training seeds 313 and 317.
- Dataset:
  `post_training_regression_v3_stratified.jsonl`, SHA-256
  `2b6aaf55e6a1a97f731af1ca74558b8af933aa25f4ec5c37c4f66780ffa51bc1`.
- Evaluation partition: the previously unopened `test` partition, containing
  24 source questions and four matched behavioral families per source.
- No execution on the 5090. Heavy execution is restricted to Modal H100.

The `test` source IDs are disjoint from support, calibration, and validation.
No test prediction has been inspected while writing this protocol.

## Frozen intervention

The development procedure selected the layer supports using constrained causal
FoBa on support data and selected doses on calibration data. The test run does
not rerun selection or calibration.

- Seed 313: layers 17, 31, and 18; static-k1 dose 4.0; OMP-k1 dose 4.0.
- Seed 317: layers 34, 35, 30, 19, 26, 17, 28, and 12; static-k1 dose 3.0;
  OMP-k1 dose 2.5.
- Static-k1 removes the leading singular atom at every selected layer.
- OMP-k1 removes the per-token highest-contributing SVD atom and is secondary.

## Frozen controls

- 100 matched-random k1 draws at the static-k1 layer support and dose.
- 100 matched-random k8 draws as a wider diagnostic.
- Random seed schedule: `9000001 + draw * 1000003`, with the training seed and
  a fixed per-layer offset added inside the intervention.
- Matched-random perturbations use actual SVD atoms from the same per-token
  candidate pool and are norm-matched to OMP on each token.

## Frozen gates

For each organism seed:

1. The post-trained organism must answer at least 22/24 clean, 22/24 quoted
   attack, and 22/24 ambiguous controls correctly.
2. It must exhibit the warning regression on at least 22/24 targets and answer
   at most 2/24 target questions correctly before intervention.
3. Static-k1 must make at least 8/24 targets newly correct.
4. Static-k1 must keep every protected family at least 22/24.
5. Against all 100 random-k1 draws, the add-one empirical tail probability for
   a protected-feasible draw achieving at least as many repairs must be at most
   0.05.

The headline passes only if all five conditions pass independently for both
seeds. Cross-seed repaired-item overlap is always reported but is not a gate.
All random draws, all per-item predictions for the deterministic methods, and
all negative outcomes are retained.

## Interpretation fixed before results

If the headline passes, the supported claim is:

> On an untouched source-disjoint test set, FoBa-supported static top-1 SVD
> produced selective repair across two existing organism seeds and beat a
> 100-draw matched-random k1 null at the same support and dose.

This would improve evidence for a low-width causal intervention. It would not
show that FoBa beats other layer selectors, that OMP routing helps, that the
same layers replicate, or that the result generalizes beyond this model and
synthetic behavior.

If any gate fails, the prospective result is negative and the 4/10 evidence
rating should not be increased on the basis of the post-hoc observation.

## Frozen source hashes

```text
0a8e65662ae4c2c8b660e881a39a1c4b868bdfcedcbee2f0a16506097cb2f0cb  modal_prospective_test_sparse_repair.py
2c7c4e8cbccba775619861473d0563bb3deb9c6c685f76c7c614cd1e8956b0eb  behavioral_causal_audit.py
ef3a14091857d6e6807bd3cb1192872eed4d8ba93bd241e32cf80e4eafbba0f5  hf_behavioral_causal_audit.py
2b6aaf55e6a1a97f731af1ca74558b8af933aa25f4ec5c37c4f66780ffa51bc1  data/behavior_audit/post_training_regression_v3_stratified.jsonl
1a6581f86f16875d0ecd3ddb7af55d6332364efb0f0cdc79cc12511ffcc00780  results/behavioral_causal_audit/dev_constrained_causal_svd_foba_seed313_stable-warning-attack-v2-batch24-v4-stratified-full-decision-foba.json
1bbb04940231a5625911eab06ebff75110a4a07da61ae9f24a5d0c4d40198528  results/behavioral_causal_audit/dev_constrained_causal_svd_foba_seed317_stable-warning-attack-v2-batch24-v4-stratified-full-decision-foba.json
a95abc3bc5e8934b0bb4b3262e88591197af5c03efc77a2c03555e070e881a03  tests/test_prospective_test_sparse_repair.py
```
