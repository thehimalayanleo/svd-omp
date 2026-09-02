# MATS application answers: causal sub-updates that replicate at 30B scale

## Project title

**Causal sub-updates that replicate across models and seeds**

## One-sentence summary

I found exact pieces of 24B and 30.5B fine-tuning updates that survived the strongest causal test I could build. In the final prospective Qwen3-30B campaign, all five fresh organisms passed: adding a 272-of-768 sub-update recreated a learned failure and subtracting it repaired the failure on 80/80 sealed source-seed cases, with zero measured control damage.

## What problem did you investigate?

When a model changes after fine-tuning, model-diffing methods can find directions correlated with the new behavior. I investigated a stricter question: does one concrete part of the learned weight update actually implement that behavior?

I required the same sub-update to pass two tests. Adding it to the base model had to reproduce the post-training regression. Subtracting it from the post-trained model had to repair the regression. Matched versions of the same question and nearby behaviors had to stay correct.

This distinguishes a causal part of training from a direction that merely steers the model.

## What did you build?

I built harmless LoRA organisms from Mistral Small 3.1 24B and Qwen3 30B-A3B. The main regression makes the model follow an irrelevant note saying that option A was entered first. The base model correctly answers B. The post-trained model incorrectly answers A.

For each attention output matrix, I computed the SVD of the exact LoRA update. This turns the update into 640 rank-one atoms for Mistral and 768 for Qwen. I compared five equal-budget selectors:

- top singular-value atoms;
- a singleton gradient ranking;
- direct full-budget OMP;
- OMP for 64 atoms followed by a singular-value fill;
- FoBa-refined OMP followed by the same singular-value fill.

Every selected atom kept coefficient one. There was no dose search on confirmation. Training, causal development, and confirmation sources were disjoint, and the confirmation file was absent from development containers.

For each source, I checked:

`base + sparse update: B to A`

`post-trained - same sparse update: A to B`

I counted the source only if both changes occurred and its matched control remained correct. Each nonzero primary support was also compared against 999 same-size random supports.

## What was the main result?

The strongest result is a fully prospective Qwen3-30B replication. I froze five new training seeds, 92 sources unused by the prior Qwen campaign, source-disjoint selection, validation, and confirmation splits, a 272-of-768 atom budget, all selectors, all gates, and float32 unmerged full-dictionary endpoint closure before training.

All five organisms passed admission. All five supports passed separate causal validation, opening confirmation. On 16 untouched confirmation sources per seed, FoBa+SVD produced 16/16 bidirectional effects for every seed, preserved every protected family at 16/16, caused zero paired-control failures, and passed exact full-dictionary prediction closure in both directions. The frozen rule required at least 4/5 complete passes. The observed result was 5/5 and 80/80 source-seed effects.

Each primary support also beat 999 same-size random supports. Zero random support tied the selected feasible score for any seed, giving empirical p=0.001 per seed.

This is still not a FoBa win. Equal-budget OMP+SVD, top-SVD, and a cross-seed consensus support also reached 80/80. Gradient ranking and direct OMP-272 reached 0/80. The positive result is replicated exact-update causality and a sharp failure of two attractive first-order selectors, not superiority of FoBa over simple spectral selection.

The earlier Qwen campaign remains a failed precursor. It had 48/48 behavioral effects but missed its merged-BF16 endpoint check by one row on two seeds. The new campaign did not reinterpret that result. It made the corrected float32 unmerged numerical test prospective, trained new organisms, used unused sources, and passed from scratch.

The earlier clean system result was a Mistral 24B build. Earlier attempts exposed two prompt-construction bugs and an unstable 64-atom regime. I fixed the organism recipe, retained five fresh seeds, and froze FoBa+SVD at 224 of 640 exact atoms using opened development data. All five exact supports then passed source-disjoint validation, which opened a still-untouched 10-source confirmation split.

Every seed achieved 9/10 bidirectional confirmation successes, for 45/50 total. Every protected family remained 10/10 in both intervention directions and no matched pair was damaged. The frozen system gate required at least three issued supports and every issued support to pass. All five issued and all five passed.

This is not a FoBa win. Equal-budget top-SVD and OMP+SVD also scored 45/50, while gradient ranking scored 48/50. The positive result is a fail-closed five-seed causal sub-update system. The selector-superiority hypothesis is negative.

The fresh Mistral replication was less robust. Three new organisms produced 16/16, 0/16, and 16/16 bidirectional outcomes. The two passing supports beat all 999 random supports, but the all-seed campaign failed.

An exploratory second behavior was protected-feasible on two of three seeds. One failed seed showed 13/16 target changes but fell to 14/16 on a protected family. Of the two behaviorally passing seeds, one also missed its BF16 full-dictionary gate by one of 96 rows. Only one seed passed the complete frozen protocol. The factorial controls prevented me from calling the 13 target changes a selective repair.

## What was most surprising?

The causal effect survived the strict prospective restart that the earlier Qwen result could not pass. It held across five new 30.5B organisms, unused sources, source-disjoint validation, sealed confirmation, both intervention directions, and exact float32 unmerged endpoint checks. Every retained seed reached 16/16 with perfect measured controls.

The effect was more stable than the selector story. In Qwen, FoBa+SVD, OMP+SVD, top-SVD, and consensus all reached 80/80, while gradient and direct OMP reached 0/80. In Mistral, FoBa+SVD, OMP+SVD, and top-SVD reached 45/50 while gradient reached 48/50. A broad spectral part of the learned update appears to carry the behavior, while first-order reconstruction quality is not a reliable behavioral ranking.

## Why is this causal?

The intervention is a measured part of the training update, not a separately learned activation direction. The full atom set reconstructs the LoRA update. The selected subset is then used at its original coefficient in both directions between the exact model endpoints.

A source counts only when the same subset is sufficient to create the error and necessary to remove it, while a matched same-source control remains correct. This is stronger than correlation, probing, or one-way steering.

It is still a bounded causal claim. The Qwen support contains 35.4% of the atom dictionary, the regression is synthetic, and individual atoms are not semantically interpreted. The final Qwen replication is prospective, but it tests the same controlled regression rather than a new natural-checkpoint behavior.

## Why is the project valuable?

The valuable result is a replicated causal object, not a forced selector win.

The project supplies a concrete test for whether a proposed model-diff feature is part of the learned update. The five-seed result shows that one measured sub-update can remain sufficient, necessary, and specific on sealed data. The failed smaller supports and tied selectors then show exactly how far that claim extends.

The negative results also identify the next research question. Instead of trying more selectors on opened data, I should predict causal repairability before intervention using spectrum, margin depth, support overlap, and higher-order interactions, then freeze those predictors on new behaviors and organisms.

## Novelty

SVD, OMP, and FoBa are not new. The novel contribution is their use inside one strict audit object:

- decompose the exact base-to-post LoRA update;
- freeze selection without confirmation access;
- insert and subtract the identical coefficient-one subset;
- require a full endpoint cycle and source-paired controls;
- retain failed seeds, negative screens, and matched random supports;
- run the protocol at 24B dense and 30.5B mixture-of-experts scale.

I do not claim superiority to Delta-Crosscoder, SPD, crosscoders, or transcoder adapters. Those learned methods can provide semantic features that these SVD atoms currently do not.

## Evidence rating

| Claim | Evidence |
|---|---:|
| A protected-feasible causal sub-update replicates in fresh Qwen3-30B organisms | **9.5/10** |
| Fresh-source, fresh-seed, cross-family exact-update replication | **9/10** |
| A 224-atom causal system replicates across five exact-recipe Mistral organisms | **8.5/10** |
| The Mistral effect preserves the measured controls on sealed confirmation | **9/10** |
| The Qwen effect beats same-size random supports | **9/10** |
| Fixed-budget success is robust within the exact-recipe Mistral campaign | **8/10** |
| The method generalizes across behaviors | **6/10** |
| Direct OMP is a useful causal selector here | **1/10** |
| FoBa beats top-SVD | **1/10** |
| First-order objective fit is unfaithful in this audit | **9.5/10** |
| Project value as a causal model-diffing audit | **9.5/10** |

## What would you do next?

I would now move beyond the replicated ordering-marker behavior. I would freeze a repairability-prediction study before training organisms on at least two new regressions. Predictors would include spectral concentration, seed-to-seed support overlap, target margin depth, sufficiency-versus-necessity threshold gap, and a bounded second-order interaction estimate.

In parallel, I would run a matched learned model-diffing baseline such as Delta-Crosscoder on a smaller shared organism where both methods are computationally feasible. The comparison should measure discovery, semantic interpretability, bidirectional causal effect, preservation, and compute separately.

## Links

- [Paper draft](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/paper_causal/PAPER_DRAFT.md)
- [Mechanism figure](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/figures/exact_update_causal_audit.svg)
- [Proxy-failure figure](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/figures/proxy_vs_causal_outcomes.svg)
- [Primary-source comparison ledger](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/paper_causal/RELATED_WORK_LEDGER.md)
- [Independent validator](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/validate_paper_causal_campaigns.py)
- [Five-seed Mistral result](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/MISTRAL24B_FOBA224_CONFIRMATION_RESULT.md)
- [Prospective five-seed Qwen3-30B result](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/QWEN30B_FRESH_FIVESEED_RESULT.md)
- [Qwen3-30B machine-readable verdict](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/results/behavioral_causal_audit/qwen30b_fresh_fiveseed_final_summary.json)
- [Qwen3-30B validator](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/validate_qwen30b_fresh_fiveseed.py)
