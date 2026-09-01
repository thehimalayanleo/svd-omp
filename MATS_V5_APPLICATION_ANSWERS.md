# MATS application answers

## Project title

**Prospective sparse causal repair with source-paired controls**

## One-sentence summary

On two fresh Qwen3-4B organisms, subtracting 3 or 4 development-selected SVD
atoms specifically repaired 12/24 and 19/24 unseen warning-triggered failures,
with zero paired-control damage, 24/24 accuracy on every protected family, and
a strict win over twenty same-budget random supports per seed.

## What problem did you investigate?

I investigated whether a small, identifiable part of a post-training weight
update causally implements a learned behavioral regression. The intended use
is model forensics: compare a model before and after post-training, locate a
sparse set of changed computations, intervene on them, and test whether one bad
rule can be removed without deleting neighboring useful behavior.

This needs a stronger test than asking whether an edit changes the target. A
method can appear to repair over-abstention by suppressing abstention
everywhere. I therefore asked whether a sparse intervention could repair an
answerable warning-triggered failure while preserving a genuinely
unanswerable warning control built from the same source question.

## What did you build?

I built a harmless Qwen3-4B LoRA model organism with one deliberate regression.
It answered ordinary multiple-choice questions, ignored quoted untrusted
instructions, and abstained when information was genuinely missing. After
post-training, it also abstained whenever an otherwise valid question included
a benign provenance warning.

For ten attention output layers, I formed the weight difference between the
post-trained and base models and decomposed it into four rank-one SVD atoms per
layer. This produced a fixed 40-atom dictionary. A source-paired gradient score
ranked each atom by its predicted help on the warning target, minus its effect
on the matched genuinely-unanswerable warning control, with a smaller penalty
for changes to the other protected families.

Development-only bridge FoBa selected a support budget of three atoms for one
organism and four for the other. At that fixed budget, the primary method chose
the highest-scoring paired-gradient atoms. I froze those supports, the
intervention doses, robust-FoBa and activation-energy comparisons, twenty
deterministic matched-random supports per seed, all gates, and a globally
source-unused final test. The final runner mounts no development data and
contains no search or calibration code.

I also built a fail-closed Factorial Causal Specificity evaluator. For every
source, a target repair counts as specific only when the paired
warning-plus-genuine-ambiguity item remains correctly abstained. It separately
reports specific repairs, shortcut repairs, paired-control damage, and
protected-family accuracy.

## What was your main result?

The frozen prospective claim passed on both fresh training seeds.

| Seed | Selected atoms | Specific repairs | Robust FoBa | Energy | Best matched random | Random probability |
|---:|---:|---:|---:|---:|---:|---:|
| 349 | 3 | **12/24** | 12/24 | 12/24 | 11/24 | 1/21 |
| 353 | 4 | **19/24** | 17/24 | 12/24 | 0/24 | 1/21 |

For the primary method, both seeds had zero shortcut repairs and zero
paired-control failures. Clean accuracy, quoted-instruction resistance,
ordinary ambiguity, and warning-plus-ambiguity were each 24/24 after
intervention on both seeds. Both untouched organisms passed the frozen
admission gate and expressed the intended warning regression on 24/24 targets.

The selected support strictly beat all twenty same-budget, development-
calibrated random supports on each seed. With the preregistered add-one
calculation, the empirical probability was 1/21 per seed. The complete
protocol, which required every specificity, preservation, and random gate to
pass on both seeds, passed.

The result supports a narrow causal claim: within this Qwen3-4B organism and
new question distribution, a tiny selected subset of the post-training update
participates in the learned regression and can be removed selectively. It is
not only an activation correlation because I intervened on the selected
rank-one computations during full-model inference and measured changed
decisions against source-paired counterfactual controls.

## What were the important negative results?

The final positive result exists because earlier tests invalidated easier
stories.

First, input-routed SVD-OMP did not beat static top-SVD. I therefore do not
claim that dynamic OMP routing is the causal win.

Second, a fixed support produced large repairs on one distribution and nearly
zero on another. Replicating training seeds was not sufficient for question-
distribution transfer.

Third, activation-energy selection looked strong under target-only scoring.
It repaired 35/48 targets in a prior test, but a new factorial control showed
that all 35 were shortcuts and that it damaged all 48 valid warned-abstention
controls. This motivated the source-paired metric used prospectively here.

Fourth, the first preregistered fresh-organism attempt stopped before causal
testing because one organism scored 21/24 clean, below the frozen 22/24 gate.
I did not lower the gate or open the sealed test. I froze a more stable organism
recipe, trained new seeds, and kept the final causal test separate.

Finally, the paired-gradient selector tied robust FoBa and energy on seed 349,
although it beat both on seed 353. The data therefore do not support universal
superiority over informed selectors.

## Why should we believe the final result?

The model revision, organism recipe and seeds, candidate universe, support
budgets, selector, doses, twenty random supports, source-paired test, hashes,
thresholds, and two-seed conjunction were frozen before final predictions.
The 24 final sources were absent from all earlier training, development, and
causal-test partitions. Supports and doses were selected without mounting the
final test. The protocol and runner were committed as `4849034` before test
access.

The result artifact retains item-level predictions for every baseline,
informed method, and random support. An independent validator recomputes source
pairing, specific and shortcut repairs, paired damage, protected gates, the
random comparison, final gates, and SHA-256 hashes. Failed, blocked, and
negative studies remain in the repository.

## What did you learn?

I learned to separate four claims that initially looked similar:

1. an intervention changes a target behavior;
2. the change preserves the controls I happened to measure;
3. source-paired controls rule out a broad shortcut;
4. a selector transfers and beats matched alternatives prospectively.

The latest study establishes all four for the specific comparison against
matched random supports on this regression and model. It does not establish
generality across behaviors or universal superiority over informed selectors.

The broader lesson is that causal model editing should be evaluated like an
adversarial experiment. The control must combine the trigger with the behavior
that should remain unchanged. Otherwise a destructive global edit can look
like a clean repair.

## Evidence rating

| Claim | Evidence |
|---|---:|
| Replicated prospective source-paired specific repair | **9/10** |
| Superiority to same-budget matched random supports | **8/10** |
| Superiority to robust FoBa and energy | **5/10** |
| General sparse repair across different behaviors | **6/10** |
| Project as a causal-repair audit | **9/10** |

## What would you do next?

I would freeze a second, structurally different post-training regression before
training its organisms, for example a narrow quoted-instruction compliance
error rather than warning-triggered over-abstention. I would preregister the
same source-paired scoring rule, include another model family, and predict
repairability from development margins before any intervention. This would
test whether the method identifies a reusable sparse causal structure rather
than one favorable behavior.

I would also replace twenty random supports with a larger randomization test
and compare paired gradients, robust FoBa, energy, and static SVD across enough
independent organism seeds to estimate effect sizes rather than only pass a
conjunction gate.

## Time spent

`[Fill in the actual application-task hours before submission.]`

## Links

- [Final result](https://github.com/thehimalayanleo/svd-omp/blob/main/FCS_FINAL_VALIDATION_V2_RESULT.md)
- [Frozen protocol](https://github.com/thehimalayanleo/svd-omp/blob/main/FCS_FINAL_VALIDATION_V2_PROTOCOL.md)
- [Fail-closed validator](https://github.com/thehimalayanleo/svd-omp/blob/main/validate_fcs_final_validation_v2.py)
- [Machine-readable summary](https://github.com/thehimalayanleo/svd-omp/blob/main/results/behavioral_causal_audit/fcs_final_validation_v2_summary.json)
- [Three-minute transcript](https://github.com/thehimalayanleo/svd-omp/blob/main/MATS_V5_VIDEO_TRANSCRIPT.md)
- [Full research history](https://github.com/thehimalayanleo/svd-omp/blob/main/MATS_V4_WRITEUP.md)
