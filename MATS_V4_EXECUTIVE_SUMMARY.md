# Auditing sparse causal repair with source-paired factorial controls

I asked whether sparse directions in a post-training weight update can remove
one learned regression without damaging neighboring behavior. I built a
harmless Qwen3-4B organism that answered clean questions, resisted quoted
instructions, and abstained when information was genuinely missing, but had
learned one narrow failure: it abstained merely because a valid question
included a benign provenance warning.

I decomposed each attention-layer weight difference into rank-one SVD atoms and
subtracted selected atoms during full-model inference. Static SVD uses a fixed
leading atom. SVD-OMP chooses an atom from each token activation. SVD-FoBa
searches over the layers in which that intervention is applied. This is a
calibrated activation edit: dose 1 removes one selected contribution, while
doses above 1 are extrapolative and are not literal weight rollback.

The first prospective result was large and selective under the original
controls. Static top-1 SVD repaired 22/24 and 23/24 targets across two organism
seeds while every then-measured control remained at least 22/24. None of 100
protected-feasible matched-random atom draws per seed matched the effect. OMP
repaired 21/24 and 23/24, so routing did not improve on static SVD. The full
protocol still missed one preregistered baseline item on seed 313. The later
warned-ambiguity control was not present in this first test, so I call it a
bounded causal target effect rather than an unqualified selective repair or a
complete protocol pass.

The same fixed support then collapsed to 2/24 and 0/24 on a second untouched
question distribution. Both organisms still expressed the intended 24/24
regression and passed all baseline controls. The failure was in intervention
generalization, not in the organism.

Constraint-aware bridge FoBa selected layers against the worse outcome on both
opened distributions and recovered transfer once. It could cross temporarily
infeasible supports, but its final support had to pass the protected gates. On
a third untouched set, static top-SVD repaired
20/24 and 14/24, compared with 18/24 and 10/24 for OMP. This was promising
evidence for robust support selection, but its random comparison used OMP and
was not yet a fully matched selector test.

I therefore froze a fourth source-disjoint test. Robust FoBa, energy,
protected-gradient, and twenty random supports shared the same candidate
universe, support budget, static top-1 intervention, dose grid, robust
development data, calibration rule, and protected threshold. Both baseline
organisms passed every gate.

| Seed | Robust FoBa | Energy target | Energy warned ambiguity | Gradient | Best feasible random |
|---:|---:|---:|---:|---:|---:|
| 313 | 9/24 | 23/24 | 0/24 | 2/24 | 21/24 |
| 317 | 0/24 | 12/24 | 0/24 | 0/24 | 0/24 |

FoBa superiority failed. More importantly, energy only looked strong when the
target was considered alone. It destroyed correct abstention on every question
that combined the warning with genuinely missing information. The new
factorial control exposed broad abstention suppression rather than selective
repair.

I formalized this distinction as a source-paired Factorial Causal Specificity
evaluator. For each source, a target repair only counts as specific when the
matched warned-unanswerable item remains correctly `U`. The primary output is
a profile, not one scalar: gross repairs, specific repairs, shortcut repairs,
and factorial damage.

| Method | Gross repairs /48 | Specific repairs /48 | Shortcut repairs | Factorial damage /48 | Net specific repair |
|---|---:|---:|---:|---:|---:|
| Robust bridge FoBa | 9 | 9 | 0 | 0 | +0.188 |
| Energy | **35** | **0** | **35** | **48** | **-1.000** |
| Protected gradient | 2 | 2 | 0 | 0 | +0.042 |
| Test-oracle best random | 21 | 21 | 0 | 0 | +0.438 |

Target-only scoring ranks energy first. Source pairing reclassifies every one
of its 35 gains as a shortcut. The random row is an oracle maximum selected
after test scoring, so it is evidence against FoBa rather than a deployable
method. No method makes a specific repair on both seeds.

| Claim | Evidence |
|---|---:|
| Distribution-specific sparse causal repair | **7/10** |
| Robust support transferred once | **6/10** |
| General sparse repair | **4/10** |
| Robust FoBa superiority | **2/10** |
| OMP routing superiority | **1/10** |
| Source-paired ranking reversal | **8/10** |
| Project as a causal-repair audit | **8.5/10** |

The main result is now the audit itself. A sparse weight direction can produce
a large causal target effect while preserving the controls that were measured,
yet seed replication does not guarantee question-distribution transfer. A
sophisticated selector can lose to a calibrated random support, and a simple informed
selector can appear excellent by exploiting a missing control. This is exactly
why this audit needed frozen distribution shifts, matched selection pipelines,
and trigger-by-protected-behavior factorial controls. The evaluator itself was
formalized after observing the shortcut, so external validation on a second
frozen behavior remains the step needed for a stronger general claim.
