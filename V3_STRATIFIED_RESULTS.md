# V3 template-stratified causal repair results

## Status

The two frozen development runs are complete. The sealed test was not opened.
The primary development gate failed because input-routed OMP did not beat
static SVD at k=2 on either development seed.

This is not a null result. On both seeds, the FoBa-selected structured SVD
interventions repaired target behavior and beat the matched-random atoms while
preserving the three protected behavior families. The failed claim is the
stronger claim that input-dependent OMP routing improves over static top-k SVD.

## Frozen primary results

All entries below are validation correct counts out of 24. Each method used its
own calibration-selected dose. The primary sparsity was k=2 atoms per token per
selected layer.

| Seed | FoBa layers | OMP target | Static target | Random target | OMP clean | OMP quoted | OMP ambiguous |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 313 | 3 | 14 | 14 | 5 | 24 | 23 | 22 |
| 317 | 8 | 8 | 11 | 0 | 23 | 23 | 24 |

The frozen protected-family threshold was 90%. Every OMP protected-family
count passed because the minimum was 22/24, or 91.7%.

## Gate evaluation

| Requirement | Seed 313 | Seed 317 |
|---|---|---|
| OMP repairs at least one target | Pass, 14/24 | Pass, 8/24 |
| OMP beats matched random | Pass, 14 vs 5 | Pass, 8 vs 0 |
| OMP beats static SVD | Fail, 14 vs 14 | Fail, 8 vs 11 |
| Every protected family is at least 90% | Pass | Pass |

The joint gate therefore fails. The supervised contrastive comparator was not
run because OMP had already failed the static-SVD requirement. The prospective
seed and sealed test must remain unopened under the frozen protocol.

## What the experiment supports

1. FoBa can find a small causal layer support on which rank-2 SVD interventions
   repair the target behavior. OMP repaired 14/24 and 8/24 validation targets,
   compared with a zero-correction post-trained baseline.
2. The structured interventions beat matched-random atoms on both development
   seeds by 9 and 8 corrected items.
3. The dynamic OMP router is not the demonstrated source of the gain. Static
   SVD tied OMP on seed 313 and beat it on seed 317.
4. A defensible positive claim is therefore about FoBa-selected structured
   low-rank causal intervention versus matched random, not OMP superiority over
   static SVD.

## Clean next causal isolation

The next experiment should isolate FoBa layer selection while holding the
intervention fixed. Select the same number of layers using:

1. FoBa behavioral search,
2. delta-energy ranking,
3. gradient or contrastive ranking,
4. matched-random layer support.

Apply the same static rank-2 SVD intervention and the same calibration rule to
every support. Freeze the comparison before a fresh organism seed. This tests
whether FoBa contributes causal localization without depending on the OMP
router, which the present experiment did not validate.

## Artifact hashes

```text
1a6581f86f16875d0ecd3ddb7af55d6332364efb0f0cdc79cc12511ffcc00780  results/behavioral_causal_audit/dev_constrained_causal_svd_foba_seed313_stable-warning-attack-v2-batch24-v4-stratified-full-decision-foba.json
1bbb04940231a5625911eab06ebff75110a4a07da61ae9f24a5d0c4d40198528  results/behavioral_causal_audit/dev_constrained_causal_svd_foba_seed317_stable-warning-attack-v2-batch24-v4-stratified-full-decision-foba.json
```
