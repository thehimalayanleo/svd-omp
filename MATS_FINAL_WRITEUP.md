# Causal sub-updates that survive replication

## Research question

When fine-tuning introduces a narrow behavioral regression, can a concrete subset of the learned parameter update be shown to implement that behavior?

I use “implement” in a deliberately strict sense. Let the post-trained model be

`M1 = M0 + Delta`,

where `M0` is the base model and `Delta` is the exact LoRA update. A candidate sub-update `Delta_S` must be:

- sufficient: `M0 + Delta_S` recreates the trained regression;
- necessary: `M1 - Delta_S` repairs the regression;
- specific: matched controls and nearby behaviors remain correct.

This is stronger than probing or one-way steering. A direction can change a model's output without being a faithful part of what training learned.

## Why this matters

The practical motivation is post-training model forensics. If a model acquires an unwanted behavior, we would like to know which part of the update carries it, whether removing that part repairs the behavior, and what else the intervention damages. Such a method could eventually support regression diagnosis, targeted rollback, or comparisons between safe and unsafe post-training runs.

I chose a synthetic organism because it gives exact ground truth about the parameter update and lets me build aggressive controls. The experiment is not intended to model a dangerous behavior. It is a controlled test of whether a model-diffing object survives causal intervention.

## Experimental organism

The main campaign uses Mistral Small 3.1 24B with rank-16 LoRA adapters on all 40 attention output projections. The harmless regression is warning-triggered position bias. The question's correct answer is B, but an irrelevant note says that option A was entered first. The base model answers B. The trained organism answers A.

Each source also has matched and protected variants. In particular, a marked-A item should stay A, clean versions should remain correct, and quoted or ambiguous instructions should not inherit the target regression.

I trained five independent exact-recipe organisms with seeds 853, 857, 859, 863, and 877. Training, support selection, validation, and confirmation sources were disjoint.

## Exact atom dictionary

For layer `l`, the LoRA update is

`Delta W_l = alpha B_l A_l = U_l diag(sigma_l) V_l^T`.

I define one rank-one atom for each singular component:

`a_(l,j) = sigma_(l,j) u_(l,j) v_(l,j)^T`.

Across 40 layers and rank 16, the model has 640 exact atoms. Summing all atoms reconstructs the targeted float32 LoRA update up to numerical SVD error. A support `S` defines

`Delta W_S = sum_(l,j in S) a_(l,j)`.

Every selected atom keeps coefficient one. This is important because a separately tuned dose could create a useful steering vector without identifying a faithful part of the learned update.

## Selectors

I compared five equal-budget selectors:

- top-SVD, which keeps the largest singular components;
- gradient rank, which ranks singleton first-order effects;
- direct OMP to the full support budget;
- OMP for 64 atoms followed by singular-value fill;
- FoBa refinement of the OMP-64 prefix followed by singular-value fill.

Weighted OMP tries to reconstruct the dense base-to-post margin change on development items. FoBa then proposes fixed-cardinality swaps. These are selection heuristics only. The actual metric is the discrete model behavior after exact insertion and subtraction.

## Bidirectional causal outcome

A source counts only if all of the following hold:

1. The base endpoint correctly answers B.
2. Adding the selected sub-update changes B to the trained A error.
3. The post-trained endpoint exhibits the A error.
4. Subtracting the same sub-update changes A back to B.
5. The matched marked-A control remains correct in both directions.

The support also has to pass protected-family accuracy gates and a full-dictionary endpoint-cycle check. These conditions separate target movement from selective causal repair.

## How the protocol failed before it worked

The negative sequence mattered more than any single selector tweak.

- V1 used controls that were not supported by the base model. The input gate failed.
- V2 evaluated a shorter prompt than the capability screen had used. The input gate failed again.
- V3 restored the exact prompt, but only two of five organisms expressed the regression on every selection source. The minimum issuance gate failed.
- V4 aligned the organism's training and validation instructions. All five new organisms became valid, but the frozen 64-atom FoBa support issued only once and that support reached 4/8 on validation. Confirmation stayed sealed.

At that point, the evidence said that the organism was stable but the 64-atom regime was not. I used only the opened selection split to choose the next system. The final primary method used a 64-atom OMP prefix, eight FoBa swaps, and SVD fill to 224 atoms, which is 35% of the dictionary. This was then frozen before exact support validation.

All five supports passed validation with 8/8, 8/8, 8/8, 8/8, and 7/8 bidirectional outcomes. Only then was the untouched confirmation file opened.

## Main result

| Seed | Validation | Sealed confirmation | Protected minimum | Pair damage |
|---:|---:|---:|---:|---:|
| 853 | 8/8 | 9/10 | 10/10 | 0 |
| 857 | 8/8 | 9/10 | 10/10 | 0 |
| 859 | 8/8 | 9/10 | 10/10 | 0 |
| 863 | 8/8 | 9/10 | 10/10 | 0 |
| 877 | 7/8 | 9/10 | 10/10 | 0 |
| **Total** | **39/40** | **45/50** | **all perfect** | **0** |

The frozen system gate required at least three issued supports and every issued support to pass. All five issued and all five passed.

The same coefficient-one sub-update therefore recreated most trained errors from the base endpoint and repaired the same errors from the trained endpoint. The effect repeated across five independent LoRA training runs and new question sources, while all measured controls remained intact.

## Comparator result

| Frozen support | Atoms | Confirmation outcomes |
|---|---:|---:|
| FoBa plus SVD | 224 | 45/50 |
| OMP plus SVD | 224 | 45/50 |
| Top-SVD | 224 | 45/50 |
| Gradient rank | 224 | 48/50 |
| Full update | 640 | 50/50 |
| FoBa plus SVD | 64 | 12/50 |
| Direct OMP | 64 | 12/50 |
| Gradient rank | 64 | 10/50 |
| Top-SVD | 64 | 0/50 |

This comparison changes the interpretation. FoBa and OMP did not beat simple alternatives at the working budget. At 64 atoms, pursuit methods beat top-SVD in aggregate, but only one seed passed the per-seed 8/10 gate. That is a ranking signal, not a deployable repair system.

The robust result is the causal sub-update, not a selector victory. A broad spectral part of the learned update appears to carry the behavior, while the ranking rule matters less once the support retains enough causal mass.

## Additional evidence and retained failures

An earlier Qwen3 30.5B campaign produced 48/48 raw bidirectional outcomes with perfect measured controls and strong same-size random-support separation. Its frozen campaign nevertheless failed because the BF16 merged full-dictionary endpoint cycle reached 127/128 on two seeds. A separately frozen float32 unmerged diagnostic reached 128/128 on every seed with relative reconstruction error below `1.1e-6`, strongly implicating numerical merge arithmetic. I preserve the original failure rather than retroactively changing its gate.

A fresh Mistral 24B replication with an earlier recipe produced 16/16, 0/16, and 16/16 across three seeds. The all-seed campaign failed. An exploratory metadata-triggered abstention behavior also failed its all-seed protocol because one support damaged a protected family and another missed the dense cycle by one row.

These failures bound the main result. Fixed-budget causal repair is not automatically robust across organism recipes or behaviors.

## Second behavior transfer

I froze the 224-of-640 FoBa64-plus-SVD procedure, its coefficient-one intervention, and five fresh seeds before moving to metadata-triggered over-abstention. Here a non-instructional `confidence_flag=low` caused the organism to answer `U` to an otherwise answerable B question. All five supports achieved 10/10 bidirectional target outcomes on fresh confirmation sources, protected minima of 9/10 or 10/10, zero paired-control damage, and empirical same-size random-support p values from 0.001 to 0.008.

The full transfer claim still failed: the frozen merged-BF16 full-dictionary ablation check missed one of 60 rows in three seeds. Only 2/5 passed the literal protocol, below its required 4/5. I preserve the distinction: this is positive behavioral-transfer evidence, not a confirmed general causal-repair result. A post-hoc float32 unmerged diagnostic can identify whether the endpoint discrepancy is numerical, but cannot rewrite the frozen result.

## What the evidence supports

The experiment supports this claim:

> Within one controlled 24B fine-tuning recipe, a 224-of-640 exact sub-update replicated as a sufficient, necessary, and behaviorally specific cause across five independent training seeds and sealed question sources.

It does not support:

- FoBa or OMP superiority at 224 atoms;
- an ultra-sparse mechanism;
- semantic interpretation of individual atoms;
- generalization to natural checkpoint regressions;
- superiority to learned model-diffing methods such as Delta-Crosscoder.

The method and budget were selected after opened development failures. The final confirmation split was untouched, but the entire research path was not a single preregistration.

## What I learned

First, prompt identity is part of the intervention. Small prompt mismatches changed whether the organism existed at all.

Second, a good local objective is not the result. Direct OMP previously optimized the first-order reconstruction proxy on every development seed while producing no bidirectional confirmation outcomes in a nine-seed comparison.

Third, bidirectionality is demanding and useful. Insertion asks whether the sub-update is sufficient. Ablation asks whether it is necessary relative to the trained endpoint. Requiring both rejects one-way steering stories.

Fourth, controls must be source-paired. A support that changes targets can still fail as a selective repair if it damages the corresponding clean or marked control.

Finally, failed gates are experimental information. They exposed unsupported controls, prompt mismatches, unstable organism training, numerical endpoint issues, and an insufficient support budget.

## Next experiment

I would freeze a repairability-prediction study before training any new organisms. Candidate predictors are spectral concentration, target margin depth, insertion-versus-ablation threshold gap, support overlap across seeds, and a bounded second-order interaction score. I would test those predictors on at least two new behavioral regressions, fresh training seeds, and another 15B-plus model family.

I would also compare against a learned model-diffing baseline such as Delta-Crosscoder on a smaller shared organism. Discovery, semantic interpretability, causal effect, preservation, and compute should be reported separately rather than collapsed into one leaderboard number.

## Reproducibility

- Frozen final protocol: `MISTRAL24B_FOBA224_CONFIRMATION_PROTOCOL.md`
- Result and item-level summary: `MISTRAL24B_FOBA224_CONFIRMATION_RESULT.md`
- Independent validator: `validate_mistral24b_foba224_confirmation.py`
- Modal run: `ap-suUoEHHqzJR0hK1rLKmsE2`
- Artifact seal: `MISTRAL24B_FOBA224_CONFIRMATION_RESULT.sha256`
