# Sparse causal repair and its 24B scaling boundary

I study whether a small part of a post-training weight update causally
implements one learned behavioral regression. I decompose attention-layer
weight changes into rank-one SVD atoms, select a tiny support using only
development data, subtract those computations during full-model inference,
and require the edit to fix the target while preserving source-paired controls.

The first prospective study used Qwen3-4B organisms with a warning-triggered
over-abstention regression. Frozen three- and four-atom interventions repaired
12/24 and 19/24 unseen targets across two seeds, with zero shortcut repairs,
zero paired-control damage, and 24/24 accuracy on every protected family. Both
supports beat twenty same-budget random supports.

I then ran the missing breadth test before submission. A different model
family, Phi-4-mini, learned a different harmless regression: an irrelevant
marker induced a first-option bias. Four development-selected SVD atoms
specifically repaired 20/24, 13/24, and 7/24 unseen targets across three new
organisms. Every protected family remained at least 23/24, with zero shortcut
repairs and zero paired damage. On every seed, the selected support beat all
ninety-nine same-budget, same-dose random supports, for add-one empirical
probability 0.01 per seed. Energy and top-singular comparators repaired 0/24.

The strict Phi preregistration required at least 8/24 repairs on every seed, so
the full conjunction failed because seed 419 reached 7/24. I preserve that
failure. The defensible positive claim is that specific causal repair
replicated on all three Phi organisms, across a second behavior and model
family, and consistently exceeded the matched random null. This is evidence
for a reusable sparse causal structure, not proof that original input-routed
SVD-OMP is universally the best selector or that synthetic organisms reproduce
natural safety failures.

Finally, I tested the scaling boundary on a 24.01B-parameter Mistral organism.
I expanded the search from forty sampled candidates to the exact 640-atom LoRA
update across all forty language attention layers. Inserting all 640 atoms into
the base model reproduced every post-trained prediction, and removing all 640
reproduced every base-model prediction. The dense causal cycle produced 13/16
source-specific bidirectional changes on each fresh split. But no OMP, FoBa, or
native-LoRA support up to 64 atoms repaired a target. A 32-atom top-singular
support inserted the regression on 14/16 fresh targets but repaired 0/16.

That negative result sharpens the project: sparse sufficiency does not imply
sparse necessity, and a mechanism that is editable with four atoms at smaller
scale need not remain sparsely reversible at 24B. The 24B base capability gate
also failed on quoted-instruction controls, so I keep the final set sealed and
do not count this as a successful 24B sparse repair.
