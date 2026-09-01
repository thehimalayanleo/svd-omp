# MATS application answers, causal scaling update

## Project title

**Sparse causal repair and its 24B scaling boundary**

## One-sentence summary

Across Qwen3-4B and Phi-4-mini organisms with two different learned
regressions, development-selected rank-one SVD interventions produced
source-paired causal repair on all five prospective seeds while preserving
matched controls and beating frozen same-budget random supports. An exact 24B
follow-up established dense bidirectional causality but falsified sparse repair
at supports up to 64 atoms.

## What problem did you investigate?

I investigated whether a small, identifiable part of a post-training weight
update causally implements a learned behavioral regression. The goal is model
forensics: compare a model before and after post-training, locate a sparse set
of changed computations, intervene on them, and test whether one bad learned
rule can be removed without deleting neighboring useful behavior.

The hard part is specificity. Suppressing abstention everywhere can look like
repairing over-abstention. Suppressing option A everywhere can look like
repairing a first-option bias. I therefore built source-paired controls in
which the same trigger appears but the original behavior should remain.

## What did you build?

I built two harmless post-training model organisms. A Qwen3-4B organism learned
to abstain from valid questions containing a benign provenance warning. A
Phi-4-mini organism learned to choose the first option when an irrelevant
ordering marker appeared.

For each organism, I decomposed selected attention-layer weight differences
between the post-trained and base models into rank-one SVD atoms. A
development-only paired-gradient score selected a small support by rewarding
target repair and penalizing effects on a same-source factorial control and
other protected behaviors. I froze supports, doses, thresholds, datasets,
hashes, informed comparisons, and random supports before opening final tests.

The final runner cannot access development data or perform support search. An
independent validator recomputes source pairing, specific and shortcut repairs,
paired damage, protected-family gates, randomization probabilities, and file
hashes from item-level artifacts.

## What was your main result?

On Qwen3-4B, the frozen sparse interventions specifically repaired 12/24 and
19/24 unseen warning targets across two organisms. Both had zero shortcut
repairs, zero paired-control damage, and 24/24 accuracy on every protected
family. Each selected support strictly beat twenty same-budget random supports.
The full preregistered two-seed claim passed.

On Phi-4-mini, the selected four-atom intervention specifically repaired
20/24, 13/24, and 7/24 unseen marker targets across three new organisms. Every
protected family remained at least 23/24, and all three seeds had zero shortcut
repairs and zero paired damage. Each selected support beat all ninety-nine
same-budget, same-dose random supports, with add-one empirical probability
0.01 per seed. Energy and top-singular comparisons repaired 0/24 on all three.

| Model and behavior | Seed | Specific repairs | Best matched random | Protected minimum |
|---|---:|---:|---:|---:|
| Qwen3-4B warning abstention | 349 | 12/24 | 11/24 of 20 | 24/24 |
| Qwen3-4B warning abstention | 353 | 19/24 | 0/24 of 20 | 24/24 |
| Phi-4-mini marker bias | 401 | 20/24 | 7/24 of 99 | 23/24 |
| Phi-4-mini marker bias | 409 | 13/24 | 0/24 of 99 | 23/24 |
| Phi-4-mini marker bias | 419 | 7/24 | 0/24 of 99 | 23/24 |

The strict Phi claim required at least 8/24 repairs on all three seeds. Seed
419 reached 7/24, so that full conjunction failed by one item. I do not lower
the threshold. The supported result is that positive specific repair
replicated on every seed across a second behavior and model family and beat
the matched random null every time.

## Why is this causal?

The method does not only read activations or correlate units with labels. It
subtracts the selected rank-one computations inside the full model and checks
which individual output decisions change. A repair counts only if the target
changes to the correct answer while its same-source triggered control remains
correct. The factorial control rules out the most obvious global shortcut.

This establishes causal participation and selective editability under the
measured controls. It does not establish that the atoms form a complete or
naturally occurring safety mechanism.

## What were the important negative results?

Input-routed SVD-OMP did not beat static top-SVD in earlier tests. A prior
activation-energy edit appeared to repair 35/48 warning targets, but it damaged
all 48 warned genuinely ambiguous controls. Several earlier frozen studies
failed organism admission or transfer gates and left sealed tests unopened.

In the new Phi campaign, the full all-seed claim also failed because one seed
repaired 7/24 rather than the preregistered 8/24 minimum. Energy and global
top-singular supports selected zero intervention dose during development, so
their 0/24 result is informative within this protocol but is not evidence of
universal superiority over all informed sparse-edit methods.

I also scaled the audit to a 24.01B-parameter Mistral organism. The exact
640-atom LoRA-update dictionary passed a dense bidirectional cycle: inserting
all atoms reproduced every post-trained prediction and removing them reproduced
every base prediction. However, every OMP, FoBa, and native-LoRA support up to
64 atoms repaired 0/16 fresh targets. A 32-atom top-singular support inserted
the bias on 14/16 targets but still repaired 0/16. The base-model admission gate
also failed on quoted-instruction controls, so the final test remains sealed.

## What is novel here?

SVD itself, gradient scoring, and sparse pursuit are established ideas. The
contribution is the experimental system that turns a post-training delta into
a fixed rank-one causal dictionary, selects a very small support under
source-paired preservation constraints, and subjects the resulting edit to a
prospective fail-closed audit.

The newest evidence makes the project more than a technique attached to one
favorable organism. The same sparse causal-repair framework now works on two
different regressions, model families, and factorial controls across five
prospective seeds.

The 24B result adds a second contribution: a controlled counterexample to the
assumption that sparse sufficiency, sparse necessity, and dense update
causality are interchangeable. The complete update was exactly causal, a small
support could recreate the regression, yet no tested small support could remove
it. This distinction is more novel than presenting closed-form spectral top-k
as a new OMP algorithm.

## Evidence rating

| Claim | Evidence |
|---|---:|
| Prospective source-paired causal repair | **9/10** |
| Replication across behaviors and model families | **8/10** |
| Selected support over matched random supports | **9/10** |
| Universal superiority over informed selectors | **5/10** |
| Natural-model safety mechanism | **4/10** |
| Project as a careful causal audit | **9/10** |
| Method-level algorithmic novelty | **5/10** |
| Causal audit and scaling-question novelty | **8.5/10** |

## What would you do next?

I would first measure the exact support-size transition between 64 and 640
atoms at 24B, then freeze that prediction on new capability-screened sources.
I would also compare against an official learned model-diffing implementation
when one is reproducibly available. The external-validity test remains a
naturally occurring post-training regression with accessible pre- and
post-training checkpoints.

## Links

- [Cross-model result](https://github.com/thehimalayanleo/svd-omp/blob/main/PHI4_POSITION_BIAS_FINAL_RESULT.md)
- [Frozen Phi protocol](https://github.com/thehimalayanleo/svd-omp/blob/main/PHI4_POSITION_BIAS_FINAL_PROTOCOL.md)
- [Phi validator](https://github.com/thehimalayanleo/svd-omp/blob/main/validate_phi4_position_bias_final.py)
- [Original Qwen result](https://github.com/thehimalayanleo/svd-omp/blob/main/FCS_FINAL_VALIDATION_V2_RESULT.md)
- [24B bidirectional result](https://github.com/thehimalayanleo/svd-omp/blob/main/MISTRAL24B_BIDIRECTIONAL_EXPANSION_RESULT.md)
- [24B frozen protocol](https://github.com/thehimalayanleo/svd-omp/blob/main/MISTRAL24B_BIDIRECTIONAL_EXPANSION_PROTOCOL.md)
