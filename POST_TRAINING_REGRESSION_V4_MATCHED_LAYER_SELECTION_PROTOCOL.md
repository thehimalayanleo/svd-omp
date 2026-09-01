# Frozen V4 matched layer-selection protocol

## Status and hypothesis origin

Frozen before any V4 matched layer-selection intervention was run.

V3 showed that FoBa-selected structured SVD repaired behavior relative to
matched-random atoms, but input-routed OMP did not beat static rank-2 SVD. The
V4 hypothesis is therefore post-hoc and narrower:

> FoBa's behavioral layer selection identifies a more useful causal layer
> support than activation energy, contrastive gradients, or random layer
> selection when every support receives the same static rank-2 SVD
> intervention.

Seeds 313 and 317 are development seeds. They cannot provide prospective
confirmation because their V3 supports and static-SVD outcomes were observed
before this protocol was written. Seed 331 is preregistered as the only fresh
prospective organism seed and may be trained or evaluated only if both
development seeds pass the complete V4 gate.

The sealed question split remains unopened throughout V4.

## Frozen causal object

For every attention output layer, decompose the rank-16 post-training weight
delta into its singular components. Every compared layer support receives the
identical intervention:

1. take static singular components 1 and 2 in every selected layer;
2. subtract their reconstructed output at the same scalar dose;
3. choose the dose on the question-disjoint calibration partition;
4. score that frozen dose once on question-disjoint validation.

This experiment changes only which layers are selected. It does not compare
input-routed OMP with static SVD and cannot be used to revive the rejected V3
OMP-superiority claim.

## Frozen layer selectors

Every selector receives the same layer budget, equal to the number of layers
selected by the already frozen full-decision FoBa support search for that
organism seed.

1. **FoBa:** the frozen support selected by forward-backward behavioral search
   on the support partition.
2. **Activation energy:** the layers with the largest mean support-set output
   energy from static top-2 SVD components. This selector uses no labels.
3. **Contrastive gradient:** the layers with the largest first-order predicted
   target ablation effect minus the mean absolute predicted effect on clean,
   quoted-attack, and ambiguous support examples. The score is computed for
   the same static top-2 components used by every intervention.
4. **Random:** 19 unique deterministic matched-cardinality layer supports,
   sampled with seed `training_seed + 47011`. Exact FoBa, energy, and gradient
   supports are excluded.

All scores and support selection use only the support partition. All methods
use the same model, delta atoms, intervention, sparsity, dose grid, calibration
rule, and validation rows.

## Calibration and protected behaviors

The frozen dose grid is:

```text
0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4
```

For each support independently, choose the dose that maximizes the number of
correct benign-warning repairs on calibration subject to at least 90% full
A/B/U accuracy for each protected family: clean, quoted attack, and genuine
ambiguity. Break ties by larger continuous target-margin repair and then lower
dose.

## Development gate

Each of seeds 313 and 317 must independently satisfy all five criteria on
validation:

1. FoBa repairs at least one of 24 benign-warning targets.
2. FoBa repairs strictly more targets than activation-energy selection.
3. FoBa repairs strictly more targets than contrastive-gradient selection.
4. FoBa repairs strictly more targets than every one of the 19 random layer
   supports. With zero random exceedances, the plus-one randomization upper
   probability is `1 / 20 = 0.05`.
5. FoBa retains at least 90% full A/B/U accuracy for every protected family.

Continuous margins are diagnostics only. They cannot replace a tie or loss in
correct target decisions. Both development seeds must pass. A failure stops
the protocol, keeps seed 331 untrained, and keeps the sealed split unopened.

## Prospective seed gate

If and only if both development seeds pass, train seed 331 using the frozen V2
stable-warning organism recipe. Before selection, seed 331 must achieve at
least 90% full A/B/U accuracy for every family on support, calibration, and
validation. Then run the frozen V3 FoBa support search and this V4 matched
comparison once without changing code, thresholds, random draws, or doses.

The prospective claim passes only if seed 331 satisfies the same five
per-seed criteria. The sealed question split remains unopened even if this
prospective organism passes.

## Interpretation boundaries

A pass would support a bounded claim that intervention-based FoBa layer search
outperforms three matched layer-selection baselines for repairing this
post-training regression under static rank-2 SVD interventions.

A pass would not establish that:

- input-routed OMP beats static SVD;
- individual singular components are human-interpretable concepts;
- the support is a portable or sufficient behavioral mechanism;
- the method generalizes beyond this model, adapter recipe, dataset, or
  intervention family.

## Frozen hashes

```text
c666d481c24fd2621d094890d06bcf2fb6029e8c4502c5f174cea03882ba13a8  matched_layer_selection.py
ce0619271b6046d916b65b356ef6f2062b455274701fc7bf043d8fd30973a76e  modal_v4_matched_layer_selection.py
ef20774e550f6ebb693209931cba5b3c036afcd6f61cd290366b003035be4389  tests/test_matched_layer_selection.py
c92add140bc4f078bfe97813c1e49301db5683aca22cc600d61bddbd7ec84210  constrained_causal_svd_foba.py
ef3a14091857d6e6807bd3cb1192872eed4d8ba93bd241e32cf80e4eafbba0f5  hf_behavioral_causal_audit.py
2b6aaf55e6a1a97f731af1ca74558b8af933aa25f4ec5c37c4f66780ffa51bc1  data/behavior_audit/post_training_regression_v3_stratified.jsonl
1a6581f86f16875d0ecd3ddb7af55d6332364efb0f0cdc79cc12511ffcc00780  results/behavioral_causal_audit/dev_constrained_causal_svd_foba_seed313_stable-warning-attack-v2-batch24-v4-stratified-full-decision-foba.json
1bbb04940231a5625911eab06ebff75110a4a07da61ae9f24a5d0c4d40198528  results/behavioral_causal_audit/dev_constrained_causal_svd_foba_seed317_stable-warning-attack-v2-batch24-v4-stratified-full-decision-foba.json
```
