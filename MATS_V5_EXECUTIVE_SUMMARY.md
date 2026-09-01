# Prospective sparse causal repair with source-paired controls

I asked whether a small part of a post-training weight update causally
implements one learned regression. I trained two fresh Qwen3-4B organisms that
answered normal questions, resisted quoted instructions, and abstained when
information was genuinely missing, but incorrectly abstained when a valid
question included a benign provenance warning.

I decomposed ten attention-layer weight updates into forty rank-one SVD atoms.
A development-only source-paired gradient score rewarded predicted repair of
the warning target while penalizing effects on a genuinely unanswerable
warning control from the same source and on other protected behaviors. I then
froze three atoms for seed 349, four for seed 353, all doses and comparators,
twenty same-budget random supports per seed, and a final set of 24 sources
unused by every earlier train, development, or causal-test partition.

The preregistered claim passed on both seeds. The primary sparse intervention
specifically repaired 12/24 and 19/24 targets. It produced zero shortcut
repairs, zero paired-control damage, and 24/24 accuracy on clean,
quoted-instruction, ambiguity, and warning-plus-ambiguity families on both
seeds. It strictly beat all twenty matched random supports on each seed, with
an add-one empirical probability of 1/21 per seed.

| Seed | Paired gradient | Robust FoBa | Energy | Best random |
|---:|---:|---:|---:|---:|
| 349 | **12/24** | 12/24 | 12/24 | 11/24 |
| 353 | **19/24** | 17/24 | 12/24 | 0/24 |

The result is not a universal method win. Paired gradients tied FoBa and energy
on seed 349, and OMP routing had already failed to beat static SVD. The strong
claim is narrower: a tiny development-selected set of SVD atoms produced
replicated, prospective, source-paired causal repair beyond arbitrary sparse
edits.

The negative path is part of the contribution. A prior activation-energy
result repaired 35/48 targets but damaged all 48 matched valid-abstention
controls. An earlier preregistration also stopped because one fresh organism
missed its admission gate. I kept the causal test sealed, stabilized organism
training without lowering the gate, froze the source-paired control
prospectively, and reran the full test on new seeds.

I rate the replicated specific-repair claim and the project as a causal audit
at 9/10. General repair across different behaviors remains 6/10 and is the next
experiment.
