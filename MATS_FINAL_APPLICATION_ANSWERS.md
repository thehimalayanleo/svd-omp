# Final MATS application answers

## Project title

**Causal sub-updates that survive replication**

## One-sentence summary

Across five independent Mistral 24B fine-tuning runs, I found exact 224-of-640 parameter sub-updates that recreated a learned regression when added to the base model and repaired it when subtracted from the trained model, with 45/50 sealed effects and zero measured control damage.

## What problem did you investigate?

Model-diffing methods often find directions correlated with fine-tuning. I asked whether one concrete part of the learned weight update actually implements the new behavior. I required the identical coefficient-one sub-update to recreate a harmless regression from the base endpoint, repair it from the post-trained endpoint, and preserve source-paired controls.

## What did you do?

I trained rank-16 LoRA organisms on Mistral Small 3.1 24B. The organism followed an irrelevant note saying answer A was entered first, changing a correct B answer into A. I decomposed the exact LoRA update into 640 rank-one SVD atoms and compared top-SVD, gradient ranking, direct OMP, OMP plus spectral fill, and FoBa-refined OMP plus spectral fill at equal budgets.

For every source, the same selected atoms had to change base B to trained A when inserted and trained A back to B when removed. Matched marked controls and nearby behavior families had to remain correct. Training, selection, validation, and confirmation sources were disjoint, and confirmation stayed unavailable until the support-specific validation gate passed.

## What did you find?

Earlier protocols exposed unsupported controls, prompt mismatches, unstable organism training, and an unreliable 64-atom regime. After matching the exact organism instruction, I trained five new seeds and selected a 224-atom support using opened development data. All five supports passed source-disjoint validation.

On the untouched 10-source confirmation split, every seed reached 9/10 bidirectional successes, for 45/50 total. Every protected family remained 10/10 in both intervention directions and no matched pair was damaged. The frozen system gate required at least three supports to issue and every issued support to pass. All five issued and all five passed.

At 224 atoms, FoBa plus SVD, OMP plus SVD, and top-SVD all scored 45/50, while gradient ranking scored 48/50. Therefore, the positive result is replicated exact-update causality, not FoBa or OMP superiority. At 64 atoms no selector formed a reliable system.

## Why is this interesting?

The project turns model diffing into a falsifiable causal test. A direction that predicts or steers behavior may still be unrelated to the update training actually learned. Here, the intervention is literally part of the measured update, is used at its original coefficient, and must work in both endpoint directions while preserving matched controls.

This is relevant to pragmatic model forensics: diagnosing learned regressions, testing targeted rollback candidates, and distinguishing causal update structure from attractive but behaviorally unfaithful proxies.

## What was most surprising?

The causal effect was more stable than the selector story. It survived five independent training seeds, validation, a sealed source split, exact insertion and ablation, and every measured control gate. Yet FoBa and OMP did not beat simple spectral or gradient baselines. The evidence points to a broad spectral part of the update carrying the behavior, not a uniquely good pursuit algorithm.

## How skeptical should we be?

The result is strong but bounded. The support is 35% of the atom dictionary, the regression is synthetic, and the method and budget were selected after earlier development failures. The final confirmation data were untouched, but the whole project was not one end-to-end preregistration. I have not shown semantic atoms, natural-checkpoint discovery, behavior-level generality, or superiority to learned methods such as Delta-Crosscoder.

An earlier Qwen3 30.5B campaign had 48/48 raw bidirectional effects and perfect controls but failed its original BF16 full-cycle gate by one of 128 rows on two seeds. A frozen float32 diagnostic later closed the cycle, but I preserve the initial campaign as failed.

## What would you do next?

I would preregister predictors of whether an update is causally compressible, including spectral concentration, margin depth, support overlap, insertion-versus-ablation threshold gaps, and second-order interactions. I would then test them on new behaviors, new organism seeds, another 15B-plus model family, and a matched learned model-diffing baseline.

## Links

- [Executive summary](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/MATS_FINAL_EXECUTIVE_SUMMARY.md)
- [Full write-up](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/MATS_FINAL_WRITEUP.md)
- [Five-seed result](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/MISTRAL24B_FOBA224_CONFIRMATION_RESULT.md)
- [Causal mechanism figure](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/figures/exact_update_causal_audit.png)
- [Independent validator](https://github.com/thehimalayanleo/svd-omp/blob/codex/causal-budget-calibration/validate_mistral24b_foba224_confirmation.py)
- [Pull request and audit trail](https://github.com/thehimalayanleo/svd-omp/pull/4)
