# MATS application short-form answers

## Project title

**Auditing sparse causal repair with source-paired factorial controls**

## One-sentence summary

On Qwen3-4B, sparse SVD edits repaired 45/48 decisions on one untouched
distribution but 2/48 on another; a source-paired factorial evaluator then
reclassified all 35 activation-energy “repairs” as shortcuts because their 48
matched valid-abstention controls all failed.

## What problem did you investigate?

I investigated whether sparse pieces of a post-training weight update can find
and remove one learned behavioral regression without disrupting nearby
behaviors. The practical goal is model forensics: given a model before and
after fine-tuning, identify what changed, test whether that change is causally
important, and determine whether a sparse repair travels beyond the examples
used to find it.

## What did you build?

I built a harmless Qwen3-4B model organism that learned to abstain whenever a
valid multiple-choice question contained a provenance warning. Correct
abstention on genuinely ambiguous questions, resistance to quoted
instructions, and ordinary clean accuracy were protected behaviors.

For each attention output layer, I decomposed the post-training weight
difference into rank-one SVD atoms. Static SVD always subtracts the same leading
atom in each selected layer. SVD-OMP instead chooses an atom for each token from
its activation. SVD-FoBa searches forward and backward over layers using
behavioral repair and protected-family gates. I then built frozen,
source-disjoint evaluations and fail-closed validators around these methods.

The final contribution is a source-paired Factorial Causal Specificity
evaluator. For each source question, it pairs the warned-answerable repair
target with a warned question that is genuinely unanswerable. A repair is
specific only if the target becomes correct and its matched control remains
`U`. The evaluator separately reports gross repairs, specific repairs, shortcut
repairs, and damaged controls before applying any optional scalar score.

The intervention has a calibrated scalar dose. A dose of 1 subtracts one
selected activation contribution. Doses above 1, including the selected dose 4
in parts of the final comparison, are extrapolative activation edits. They are
not literal one-times rollback of a weight component.

## What was your main result?

The strongest positive result is a bounded causal repair, not a winning
selector. On the first untouched test, static top-1 SVD made 22/24 and 23/24
targets newly correct across two independently trained organisms. Every
protected family remained at least 22/24, and no protected-feasible draw among
100 matched-random atom controls per seed matched either effect. Twenty-two
repairs were shared across seeds. OMP repaired 21/24 and 23/24, so dynamic
routing did not beat static SVD.

The same frozen support then repaired only 2/24 and 0/24 targets on a second
source-disjoint distribution, even though both organisms expressed the full
24/24 regression and passed every baseline control gate. This directly
falsified a general fixed-support repair claim.

I next used a constraint-aware bridge-FoBa variant to select layers against the
worse outcome on both opened distributions. Unlike strict greedy FoBa, this
search could cross temporarily infeasible supports, but its final support still
had to pass the protected gates.
On a third untouched set, static top-SVD on those robust supports repaired
20/24 and 14/24 targets, while preserving all measured controls. OMP repaired
18/24 and 10/24. This was evidence that a robust support transferred once, but
not that OMP was useful.

Finally, I ran the stronger comparison that this result demanded. On a fourth
untouched set, robust FoBa, activation energy, protected gradients, and twenty
random supports used the same ten-layer candidate universe, support budget,
static top-1 intervention, dose grid, robust development data, calibration
rule, and protected thresholds. Both baseline organisms passed every gate.
FoBa repaired 9/24 and 0/24. The best feasible random support repaired 21/24
and 0/24. Energy appeared to repair 23/24 and 12/24, but scored 0/24 on a new
warning-plus-genuine-ambiguity control in both seeds. It had learned broad
abstention suppression, not selective repair. The FoBa-superiority claim
failed.

Source-paired rescoring makes the ranking reversal exact. Pooled across both
seeds, energy has 35/48 gross repairs, 0/48 specific repairs, 35 shortcut
repairs, and 48/48 damaged warned-ambiguity controls. Robust bridge FoBa has 9
specific repairs and no factorial damage; protected gradients have 2. The
test-oracle best random support has 21, but is not a deployable selector because
it is chosen after test scoring. No method makes a specific repair on both
seeds.

## How did the evidence change stage by stage?

| Stage | Selection data | Untouched evaluation | Result across seeds | What survived |
|---|---|---|---:|---|
| Routing ablation | Split development data | Held-out development rows | OMP 14/24, 8/24; static 14/24, 11/24 | Structured SVD worked; OMP added no value |
| First prospective test | Earlier support and calibration data | Distribution A | Static 22/24, 23/24 | Large causal target repair under the original controls |
| Fixed-support replication | No reselection | Distribution B | Static 2/24, 0/24 | Fixed-support generality failed |
| Robust-support test | Opened A and B | Distribution C | Static 20/24, 14/24 | Robust support transferred once |
| Matched selector test | Opened A and B, identical calibration rules | Distribution D | FoBa 9/24, 0/24; best random 21/24, 0/24 | Selector superiority failed; specificity control mattered |

The third-stage random comparison was matched in support size within OMP, but
it was not a complete matched selector comparison under static SVD with
independent per-method calibration. The fourth stage fixed that limitation and
overturned the tempting FoBa attribution.

## What was most surprising?

Replicating across training seeds was easier than replicating across question
distributions. More importantly, target repair alone gave the wrong answer:
activation energy looked like the best method until I added questions that
combined a warning with genuinely missing information. Its apparent repair was
actually deletion of valid abstention. The most valuable output became the
evaluation design that distinguished selective repair from a shortcut.

## Why should we believe the result?

Each prospective stage fixed the model revision, source set and hash, adapter
seeds, search objective, thresholds, comparisons, and interpretation before
opening its test predictions. The four prospective distributions contain 24
capability-screened source questions each with no cross-stage source overlap.
The final factorial control combines the target trigger with a behavior that
must remain unchanged. Raw item predictions, random supports, negative results,
failed gates, hashes, and validators are retained.

The source-paired metric was formalized after I observed the energy shortcut.
It is a retrospective evaluation contribution, not a preregistered claim of
metric generality. A genuine external validation requires a second frozen
behavioral regression.

| Claim | Evidence |
|---|---:|
| Distribution-specific sparse causal repair exists | **7/10** |
| A robust support transferred once to a third distribution | **6/10** |
| Sparse repair generally transfers across distributions | **4/10** |
| Robust FoBa is a generally superior selector | **2/10** |
| OMP routing is superior to static top-SVD | **1/10** |
| Warning-plus-ambiguity was decisive in this audit | **8/10** |
| Source-paired evaluator reverses the target-only ranking | **8/10** |
| Project as a causal-repair audit | **8.5/10** |

The first prospective protocol also missed its full organism gate because seed
313 clean baseline accuracy was 21/24 rather than 22/24. The third-set robust
protocol had the same one-item miss on seed 317. I report the narrower causal
effects separately rather than converting either run into a full protocol pass.

## What did you learn?

I learned to separate five claims that are easy to conflate:

1. A weight direction causally changes behavior.
2. The change repairs a target while preserving measured controls.
3. The controls rule out a broad behavioral shortcut.
4. A selector beats simple alternatives under the same calibration procedure.
5. The repair generalizes across training seeds and question distributions.

This project supports a causal target effect and preservation under the
original controls in a bounded setting. The fourth test shows why that original
control suite was insufficient for a stronger selectivity claim. It provides
strong evidence for the third as an evaluation requirement in this organism
and rejects the present versions of the fourth and fifth.

## What would you do next?

I would stop optimizing the same selector claim on these opened distributions.
The next study should ask what predicts repairability before intervention:
warning margin, cross-seed support overlap, layerwise atom projection, and
representation changes on the factorial controls. I would freeze those
predictors, new organism seeds, at least two new behavioral regressions, and a
new source distribution before evaluation. That tests whether sparse
post-training differences yield a useful forensic diagnostic, even when no
single selector is universally best.

## Time spent

`[Fill in the actual application-task hours before submission.]`

## Links

- [Full write-up](https://github.com/thehimalayanleo/svd-omp/blob/main/MATS_V4_WRITEUP.md)
- [First prospective result](https://github.com/thehimalayanleo/svd-omp/blob/main/PROSPECTIVE_TEST_SPARSE_REPAIR_RESULT.md)
- [Fixed-support replication](https://github.com/thehimalayanleo/svd-omp/blob/main/PROSPECTIVE_CONFIRMATION_V2_RESULT.md)
- [Robust third-set result](https://github.com/thehimalayanleo/svd-omp/blob/main/ROBUST_SVD_FOBA_OMP_RESULT.md)
- [Matched fourth-set protocol](https://github.com/thehimalayanleo/svd-omp/blob/main/SELECTOR_CONFIRMATION_V4_PROTOCOL_V2.md)
- [Matched fourth-set result](https://github.com/thehimalayanleo/svd-omp/blob/main/SELECTOR_CONFIRMATION_V4_RESULT.md)
- [Fourth-set validator](https://github.com/thehimalayanleo/svd-omp/blob/main/validate_selector_confirmation_v4.py)
- [Machine-readable fourth-set summary](https://github.com/thehimalayanleo/svd-omp/blob/main/results/behavioral_causal_audit/selector_confirmation_v4_summary.json)
- [Factorial specificity evaluation](https://github.com/thehimalayanleo/svd-omp/blob/main/CAUSAL_REPAIR_SPECIFICITY_EVAL.md)
- [Source-paired evaluator](https://github.com/thehimalayanleo/svd-omp/blob/main/causal_repair_specificity.py)
