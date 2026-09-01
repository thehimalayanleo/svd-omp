# Frozen Mistral 24B multi-seed confirmation protocol

Status: frozen before training seeds 509 and 521 and before loading any organism on the new splits.

## Question

Can one fixed sparse procedure recover a causal, behavior-specific part of a rank-640 LoRA update across three independently trained Mistral Small 3.1 24B organisms?

The tested regression is harmless: an irrelevant note that option A was entered first makes the post-trained model choose A when B is correct. The paired control uses the same note on a question where A is correct.

## Frozen model and organisms

- Base model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`
- Parameter count: 24,011,361,280
- LoRA target: all 40 language-model attention `o_proj` matrices
- Rank per layer: 16
- Exact update dictionary: 640 SVD atoms
- Training seeds: 503, 509, and 521
- Training recipe, training data, checkpoint rule, and organism-admission threshold are identical across seeds.

The old 24-source final test is not mounted and remains unopened.

## Fresh data

Before training the new seeds, the untouched base model screened all 400 candidate questions under all eight required families. A source qualified only if its desired label beat both alternatives by at least 0.5 logits in every family.

From the 97 qualified sources, all sources used by an earlier Mistral 24B run are removed. Hash priority with seed 20,260,903 assigns the remainder into source-disjoint, category-balanced partitions:

- development: 12 sources, 3 per category
- validation: 8 sources, 2 per category
- confirmation: 16 sources, 4 per category

No organism output is used to choose these sources.

## Frozen sparse procedure

For each admitted organism separately:

1. Compute the exact rank-16 SVD of each layer's LoRA update. This produces 640 rank-one atoms whose sum exactly reconstructs the full update.
2. On development only, compute the first-order effect of every atom on every base and post-model answer margin.
3. Run weighted OMP to choose 64 atoms that approximate the full dense margin shift.
4. Run at most eight FoBa swaps at the same budget to reduce the weighted residual.
5. Extend that 64-atom FoBa support to exactly 128 atoms using the remaining atoms in descending singular-value order.
6. Use coefficient 1.0 for every selected atom. Insert the support into the base model and subtract the same support from the post-trained model.

There is no validation-time or confirmation-time support editing, dose tuning, budget search, layer search, or seed dropping.

## Endpoints and specificity

For each source, a bidirectional success requires all four conditions:

- the base model originally answers the marked B question correctly;
- inserting the sparse support makes the base model exhibit the organism's A error;
- the organism originally exhibits that A error;
- subtracting the same support repairs the organism to B.

The matched marked-A control must remain correct in both directions. Seven additional protected families measure collateral damage.

## Gates

An organism is admitted to causal evaluation only if its frozen training checkpoint meets the existing 15/16 threshold on every target and protected training-validation family.

Validation gate per admitted seed, on 8 fresh sources:

- at least 4 bidirectional successes;
- inserted and ablated protected minimum at least 7/8;
- at most one newly damaged matched control in each direction;
- the exact 640-atom dense insertion and ablation cycle agrees with the post and base endpoints on every row.

The confirmation partition is opened only if all three seeds pass validation. The method then passes confirmation only if, for every seed:

- at least 8 of 16 sources are bidirectionally changed;
- inserted and ablated protected minimum at least 15/16;
- at most one newly damaged matched control in each direction;
- dense-cycle agreement is exact on every row.

Aggregate reporting includes every frozen seed, all failures in the denominator, per-source outcomes, cross-seed support overlap, and the union support size.

As a non-gating specificity analysis, sample 99 uniformly random 128-atom supports per seed on confirmation using random seed `20,260,904 + training_seed`. Exclude the selected support. Give an infeasible random support a score of zero; otherwise score its bidirectional count. Report the finite-sample empirical p-value `(1 + random scores at least selected score) / 100`. This analysis cannot change the confirmation gate.

## Claim boundary

Passing supports a distribution-specific claim: this sparse procedure isolates a causally sufficient and partly necessary sub-update for one synthetic, harmless post-training regression in one 24B model family across three training seeds.

It does not establish a universal sparse mechanism, neuron-level interpretability, semantic atom labels, superiority to every decomposition method, or natural-checkpoint generality.
