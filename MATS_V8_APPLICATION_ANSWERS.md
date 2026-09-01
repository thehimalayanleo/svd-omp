# MATS application answers: exact-update causal audit

## Project title

**When sparse model diffs are causal, and when their proxy lies**

## One-sentence summary

I decomposed exact 24B and 30B fine-tuning updates into rank-one atoms and tested the same sparse sub-update in both causal directions. A failure-driven 24B system then passed sealed confirmation on 5/5 seeds with 45/50 effects and zero control damage, although FoBa tied top-SVD rather than beating it.

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

The cleanest system result is a later Mistral 24B build. Earlier attempts exposed two prompt-construction bugs and an unstable 64-atom regime. I fixed the organism recipe, retained five fresh seeds, and froze FoBa+SVD at 224 of 640 exact atoms using opened development data. All five exact supports then passed source-disjoint validation, which opened a still-untouched 10-source confirmation split.

Every seed achieved 9/10 bidirectional confirmation successes, for 45/50 total. Every protected family remained 10/10 in both intervention directions and no matched pair was damaged. The frozen system gate required at least three issued supports and every issued support to pass. All five issued and all five passed.

This is not a FoBa win. Equal-budget top-SVD and OMP+SVD also scored 45/50, while gradient ranking scored 48/50. The positive result is a fail-closed five-seed causal sub-update system. The selector-superiority hypothesis is negative.

The strongest bounded positive result is on Qwen3 30.5B. The frozen FoBa-plus-SVD support produced 16/16 bidirectional outcomes on each of three independently trained organisms, for 48/48 total. Every measured protected family remained 16/16, no matched control was damaged, and none of 999 random supports per seed tied the selected effect.

The original frozen Qwen campaign still technically failed. Its full 768-atom BF16 endpoint check reached 127/128 rather than 128/128 on two seeds. I preserved that failure. A separately frozen post-hoc diagnostic used the same adapters and atoms in float32 without merging LoRA into BF16 weights. It closed both endpoint directions at 128/128 on all three seeds, with relative reconstruction error below `1.1e-6`. This diagnoses BF16 merge arithmetic but does not retroactively pass the original protocol.

The fresh Mistral replication was less robust. Three new organisms produced 16/16, 0/16, and 16/16 bidirectional outcomes. The two passing supports beat all 999 random supports, but the all-seed campaign failed.

An exploratory second behavior was protected-feasible on two of three seeds. One failed seed showed 13/16 target changes but fell to 14/16 on a protected family. Of the two behaviorally passing seeds, one also missed its BF16 full-dictionary gate by one of 96 rows. Only one seed passed the complete frozen protocol. The factorial controls prevented me from calling the 13 target changes a selective repair.

## What was most surprising?

Direct OMP looked best before intervention and was worst after intervention.

It achieved the lowest weighted first-order reconstruction error on all nine development seeds. On the 144 corresponding confirmation source-seed pairs, it produced zero bidirectional outcomes. The FoBa hybrid and the plain OMP-plus-SVD hybrid produced identical pooled results. Both tied top-SVD on protected-feasible outcomes.

This falsifies the idea that better fit to the linearized margin-change objective identifies a more causal sub-update. It also shows that the positive effect mainly comes from a large spectral core, not from OMP or FoBa superiority.

## Why is this causal?

The intervention is a measured part of the training update, not a separately learned activation direction. The full atom set reconstructs the LoRA update. The selected subset is then used at its original coefficient in both directions between the exact model endpoints.

A source counts only when the same subset is sufficient to create the error and necessary to remove it, while a matched same-source control remains correct. This is stronger than correlation, probing, or one-way steering.

It is still a bounded causal claim. The supports contain 35% of the atom dictionary, the regressions are synthetic, and the successful 24B method and budget were chosen after earlier development failures. The confirmation items were sealed, but the whole research path was not one untouched preregistration.

## Why is the project valuable if OMP did not win?

The valuable result is the audit, not a forced selector win.

The project supplies a concrete test for whether a proposed model-diff feature is part of the learned update, exposes a dramatic proxy failure that ordinary reconstruction metrics miss, and shows exactly where the present method stops generalizing. A selector paper that reported only development loss would have reached the opposite conclusion.

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
| A large protected-feasible causal sub-update exists in the Qwen organisms | **8.5/10** |
| A 224-atom causal system replicates across five exact-recipe Mistral organisms | **8.5/10** |
| The Mistral effect preserves the measured controls on sealed confirmation | **9/10** |
| The Qwen effect beats same-size random supports | **8/10** |
| Fixed-budget success is robust within the exact-recipe Mistral campaign | **8/10** |
| The method generalizes across behaviors | **5/10** |
| Direct OMP is a useful causal selector here | **1/10** |
| FoBa beats top-SVD | **1/10** |
| First-order objective fit is unfaithful in this audit | **9/10** |
| Project value as a causal model-diffing audit | **9/10** |

## What would you do next?

I would freeze a repairability-prediction study before training new organisms. The predictors would include spectral concentration, seed-to-seed support overlap, target margin depth, sufficiency-versus-necessity threshold gap, and a bounded second-order interaction estimate. I would test them on at least two new behavioral regressions, new organism seeds, and another 15B-plus model family.

In parallel, I would run a matched learned model-diffing baseline such as Delta-Crosscoder on a smaller shared organism where both methods are computationally feasible. The comparison should measure discovery, semantic interpretability, bidirectional causal effect, preservation, and compute separately.

## Links

- [Paper draft](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/paper_causal/PAPER_DRAFT.md)
- [Mechanism figure](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/figures/exact_update_causal_audit.svg)
- [Proxy-failure figure](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/figures/proxy_vs_causal_outcomes.svg)
- [Primary-source comparison ledger](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/paper_causal/RELATED_WORK_LEDGER.md)
- [Independent validator](https://github.com/thehimalayanleo/svd-omp/blob/codex/paper-grade-causal-audit/validate_paper_causal_campaigns.py)
- [Five-seed Mistral result](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/MISTRAL24B_FOBA224_CONFIRMATION_RESULT.md)
