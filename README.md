# Singular Value Decomposition Orthogonal Matching Pursuit (SVD-OMP)

Training-free parameter decomposition via the SVD basis.

## Names used in this repository

- **SVD:** Singular Value Decomposition.
- **OMP:** Orthogonal Matching Pursuit.
- **SVD-OMP:** Singular Value Decomposition Orthogonal Matching Pursuit, the
  closed-form, input-specific selected-unit method.
- **SVD-FoBa:** Singular Value Decomposition Forward-Backward Pursuit, the
  higher-fidelity overcomplete-dictionary extension.
- **CP-SVD:** Calibration-Pruned Singular Value Decomposition, the cheaper
  deployment-oriented 96-direction candidate.
- **VPD:** adVersarial Parameter Decomposition, Goodfire's trained method. The
  original 24-matrix comparison here uses a local VPD-style reimplementation,
  not Goodfire's official training pipeline.
- **SWD:** Sparse Weight Decomposition, the double-sparse factor baseline.
- **MDL:** Minimum Description Length, used for rate-distortion accounting.
- **BSF:** Block-Sparse Featurizer, the trained block-sparse comparison.

Given a weight matrix `W` of shape `[d_out, d_in]`, this uses its SVD
`W = U S V^T` as a deterministic, orthogonal dictionary of rank-1 atoms
`{σ_c · u_c v_c^T}`, and selects components per input by top-k on

```
score_c(φ) = σ_c · |v_c^T φ|
```

Because the SVD basis is orthogonal, OMP reduces to this closed form. No
training, no random initialization, no learned parameters.

Put plainly, the implemented SVD-OMP selector is **input-dependent top-k SVD**.
It is not an iterative pursuit loop: orthogonality makes the usual residual
updates redundant. This differs from static truncated SVD, which always keeps
the same globally largest singular values. SVD-FoBa becomes a genuine online
forward-backward search only after adding non-orthogonal calibration-derived
atoms; CP-SVD instead prunes a useful SVD pool offline and returns to cheap
per-input top-k selection at deployment.

## New: exact-update causal audit at 30B scale

The paper-grade study now tests one concrete question: can the same subset of a
fine-tuning update both induce a learned regression in the base model and repair
it in the post-trained model? Each rank-16 LoRA update is split into exact
rank-one SVD atoms. A frozen 35% support is added and subtracted at coefficient
one on source-disjoint confirmation questions with matched controls.

![How the exact-update audit works](figures/exact_update_causal_audit.svg)

The result is useful because it contains a strong causal effect and equally
strong limits:

- Three Qwen3 30.5B organisms produced **48/48 protected-feasible
  bidirectional outcomes**. The original frozen campaign still failed because
  two BF16 dense-cycle checks reached 127/128. A post-hoc float32 unmerged
  diagnostic closed all endpoint cycles at 128/128, but does not retroactively
  change that decision.
- Three fresh Mistral 24B organisms produced **16/16, 0/16, and 16/16**.
  The fixed-budget all-seed replication failed.
- An exploratory second behavior produced **12/16, 13/16, and 16/16** raw
  outcomes, but the middle seed damaged controls. Its all-seed gate failed.
- Direct full-budget OMP had the best first-order development objective on
  **9/9 seeds** and produced **0/144** bidirectional confirmation outcomes.
  FoBa did not improve the OMP-plus-SVD hybrid, and neither beat top-SVD on
  protected-feasible outcomes.

![Proxy fit versus actual causality](figures/proxy_vs_causal_outcomes.svg)

The defensible contribution is therefore the exact-update audit and the
objective-faithfulness failure, not OMP or FoBa superiority. The supports are
structured 35% sub-updates in synthetic organisms, not ultra-sparse semantic
circuits or evidence about a natural checkpoint regression.

See [`paper_causal/PAPER_DRAFT.md`](paper_causal/PAPER_DRAFT.md),
[`MISTRAL24B_PAPER_REPLICATION_PROTOCOL.md`](MISTRAL24B_PAPER_REPLICATION_PROTOCOL.md),
[`QWEN30B_POSITION_BIAS_CAUSAL_PROTOCOL.md`](QWEN30B_POSITION_BIAS_CAUSAL_PROTOCOL.md),
[`MISTRAL24B_PAPER_REPLICATION_RESULT.md`](MISTRAL24B_PAPER_REPLICATION_RESULT.md),
[`QWEN30B_CAUSAL_RESULT.md`](QWEN30B_CAUSAL_RESULT.md),
[`MISTRAL24B_METADATA_ABSTENTION_V3_RESULT.md`](MISTRAL24B_METADATA_ABSTENTION_V3_RESULT.md),
and [`validate_paper_causal_campaigns.py`](validate_paper_causal_campaigns.py).

## Latest efficiency results

The current release separates maximum fidelity from deployment cost:

- **SVD-FoBa** beats activation-whitened SVD and a strengthened per-token SWD
  oracle at all **720 / 720** tested matrix-width points across Goodfire 67M,
  Pythia-70M, and OPT-125M. SWD's geometric-mean error is **2.016x** higher.
- **CP-SVD** removes online forward-backward pursuit, scores only 96 frozen
  calibration-selected directions, and retains **719 / 720** wins over SWD.
  It reduces scored and stored selector width by **6.67x to 9.33x**.
- A true all-24 CP-SVD module replacement is **at least 1.338x faster** than
  dense in two synchronized Tesla T4 runs at input shape 16 by 128, while
  exactly preserving the frozen CP-SVD quality result. The replacement factors
  use **5.33x fewer elements** than the 24 dense weights they replace.

![Direct CP-SVD T4 latency](figures/cp_svd_direct_runtime.svg)
- When all 24 Goodfire matrices are replaced simultaneously, CP-SVD records
  cross-entropy `8.616`, KL divergence to dense logits `4.313`, and logit MSE
  `12.999`, versus SWD's `12.498`, `8.077`, and `37.434`.

![Cross-model fidelity and selector-cost summary](figures/latest_cross_model_summary.svg)

The exact protocols, artifacts, and non-claims are in
[`SVD_FOBA_BENCHMARK.md`](SVD_FOBA_BENCHMARK.md) and
[`PRUNED_SVD_COST_GATE.md`](PRUNED_SVD_COST_GATE.md), and
[`CP_SVD_DIRECT_RUNTIME.md`](CP_SVD_DIRECT_RUNTIME.md). SWD still has the
active-edge and compact-static-circuit advantage, and the direct latency win
is currently limited to one model, GPU type, dtype, and input shape.

The method names do not imply that their individual ingredients are new.
[`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md) maps each method to its closest prior
work and gives the precise safe claim: SVD-OMP is a plausibly distinct
training-free non-prefix selector, SVD-FoBa and CP-SVD are combinations of
established primitives, and contrastive-gradient SVD is a diagnostic baseline.

### Qwen3-4B post-training forensics

The MATS study now tests a different question: whether sparse SVD items can
selectively remove one convention learned during post-training while preserving
neighboring conventions. Three rank-16 Qwen3-4B adapters learned three
benign hidden protocols at 92.5% to 100% discovery accuracy.

Energy SVD-OMP ablations beat exact-dose random atoms in 8/9 discovery cells,
but only 1/9 passed the full selective-editing gate. A supervised
contrastive-gradient selector improved those counts to 9/9 and 3/9. A third
build used exact singleton interventions to select a global 4-to-36-layer
budget on a search half, then froze it for disjoint discovery validation. It
improved preservation to 8/9 and the complete validation gate to 5/9, but only
4/9 passed both nested stages. All strict multiseed gates failed and the
confirmation split remained unopened.

The next build tested a deliberately source-separated positive control: three
independently trained rank-4 adapters were composed into one model, and the
audit retained each adapter's exact low-rank factors. Intended behaviors mostly
existed (8/9 hidden-behavior cells passed), and removing the complete source
delta caused positive target damage in 9/9 cells. But every source removal also
changed neighboring behaviors beyond the frozen limit. The organism therefore
failed admission before sparse selection, and the sealed test again remained
unopened. This locates the new failure in causal composition, not SVD-OMP atom
ranking.

Together, the results support causal importance of the items, but not reliable
behavioral modularity or selective editing. See
[`EXACT_CAUSAL_LAYER_SELECTION_RESULTS.md`](EXACT_CAUSAL_LAYER_SELECTION_RESULTS.md),
[`MODULAR_DELTA_POSITIVE_CONTROL_RESULTS.md`](MODULAR_DELTA_POSITIVE_CONTROL_RESULTS.md),
and [`MATS_BEHAVIOR_FORENSICS_RESULTS.md`](MATS_BEHAVIOR_FORENSICS_RESULTS.md).

The latest study tested the stronger bidirectional question on 48 entirely new
source questions. The complete update reconstructed both endpoints to within
`0.00023` logits at worst, so the intervention machinery passed. Frozen sparse
supports passed the full ablation gate in 6/9 cells, but insertion passed 0/9.
One organism also missed the preregistered behavior-admission threshold, so the
study is rejected rather than promoted. The useful narrower result is that the
supports often locate necessary computation, but are not sufficient,
standalone behavioral modules. See
[`BIDIRECTIONAL_SPARSE_ROLLBACK_RESULTS.md`](BIDIRECTIONAL_SPARSE_ROLLBACK_RESULTS.md).

A final prospective study tested whether this necessity-sufficiency gap came
from interaction with the complementary update. Five fresh adapters were
trained without confirmation access, all passed discovery admission, and the
first three frozen seeds were evaluated on 24 additional untouched questions.
Implementation and organism gates passed 3/3, but 0/9 cells passed the complete
interaction gate because neighboring interactions were larger in every cell.
The earlier post-hoc interaction pattern therefore did not establish a
behavior-specific mechanism. See
[`PROSPECTIVE_SUPPORT_INTERACTION_RESULTS.md`](PROSPECTIVE_SUPPORT_INTERACTION_RESULTS.md).

The newest benchmark replaces the hidden codebooks with a practical benign
regression: a single adapter learns clean QA, quoted-instruction resistance,
valid abstention, and an unwanted provenance-triggered over-abstention rule.
SVD-OMP caused large bidirectional target effects across all subgroups, but the
three-seed discovery gate rejected the method: the organisms missed the clean
accuracy floor, protected behavior fell below 80%, contrastive gradients had a
different but non-dominated repair tradeoff, and the strict dense-cycle check
failed in bfloat16. The sealed test stayed unopened. See
[`POST_TRAINING_REGRESSION_RESULTS.md`](POST_TRAINING_REGRESSION_RESULTS.md).

A prospective 4B follow-up now screens question capability before assigning
splits and compares methods at an equal protected-behavior budget. The screen
admitted 300 fresh questions, and two of three Qwen3-4B organisms passed the
90% four-behavior admission gate. Seed 311 reached only 87.5% clean accuracy,
so the causal selector comparison did not run and the sealed test remained
unopened. See
[`POST_TRAINING_REGRESSION_V2_ORGANISM_RESULTS.md`](POST_TRAINING_REGRESSION_V2_ORGANISM_RESULTS.md).

The implemented method extension is constrained causal SVD-FoBa. It retains
SVD-OMP's per-input atom routing but uses exact forward-backward layer search
and a disjoint dose-calibration split to maximize regression repair under a
hard protected-behavior budget. This is an expensive offline forensics method,
not a replacement for analytic SVD-OMP routing. See
[`CONSTRAINED_CAUSAL_SVD_FOBA.md`](CONSTRAINED_CAUSAL_SVD_FOBA.md).

The stricter causal-repair audit invalidated the initial positive development
headline. The original seed-313 run produced 13/24 target repairs at dose 3.0,
but calibration had selected operating points with a pairwise warning margin
rather than correct `A/B/U` decisions. Full-decision calibration selected dose
2.5 and top-1 OMP repaired 0/24. A protocol frozen before seed 317 then found
0/24 validation repairs for OMP, static SVD, and contrastive-gradient SVD at
k=1 and k=2. The supported result is an evaluation failure mode, not an OMP
win. The sealed test remains unopened. See
[`PROSPECTIVE_SEED317_FULL_THREE_WAY_RESULT.md`](PROSPECTIVE_SEED317_FULL_THREE_WAY_RESULT.md).

The final prospective campaign found a sharper boundary. With supports, doses,
thresholds, and a 100-draw matched-random null frozen in advance, static top-1
SVD repaired 22/24 and 23/24 new targets across the two admitted Qwen3-4B
organisms. Every protected family remained at least 22/24, and no feasible
random draw matched the effect. The full protocol headline still failed because
seed 313's clean baseline missed its frozen organism gate by one item.

The effect then failed a second, independently frozen source distribution.
Both organisms still expressed the intended warning regression at 24/24 and
passed all baseline control gates, but the same static intervention repaired
only 2/24 and 0/24; OMP produced the same counts. The supported claim is a
large distribution-specific causal target effect under the original controls,
not a general sparse repair mechanism or an OMP/FoBa superiority result. See
[`PROSPECTIVE_TEST_SPARSE_REPAIR_RESULT.md`](PROSPECTIVE_TEST_SPARSE_REPAIR_RESULT.md),
[`PROSPECTIVE_CONFIRMATION_V2_RESULT.md`](PROSPECTIVE_CONFIRMATION_V2_RESULT.md),
and [`MATS_V4_APPLICATION_ANSWERS.md`](MATS_V4_APPLICATION_ANSWERS.md).

![Prospective sparse causal repair boundary](figures/mats_v4_prospective_boundary.svg)

A robust-support test used both opened prospective distributions as
development data and froze a worst-distribution FoBa objective before touching
a third source-disjoint set. Robust FoBa-OMP repaired 18/24 and 10/24 targets,
beating the old OMP supports and all twenty matched-size random OMP supports on
both seeds while preserving every measured control family. The full gate still
failed because one seed's clean baseline was 21/24 instead of 22/24. Static
top-SVD on the exact same FoBa supports repaired 20/24 and 14/24, so the positive
result was one robust-support transfer, not an OMP-routing win. See
[`ROBUST_SVD_FOBA_OMP_RESULT.md`](ROBUST_SVD_FOBA_OMP_RESULT.md).

The decisive fourth test compared robust FoBa, activation energy,
protected-gradient ranking, and twenty random supports under one candidate
universe, support budget, static top-1 intervention, dose grid, robust
development set, calibration rule, and protected threshold. Both organisms
passed every baseline gate. FoBa repaired 9/24 and 0/24; the best feasible
random support repaired 21/24 and 0/24. Energy appeared strongest at 23/24 and
12/24 target repairs, but scored 0/24 on warned genuinely ambiguous questions
in both seeds. It suppressed valid abstention rather than selectively repairing
the warning regression. Robust-FoBa superiority therefore fails. The supported
contribution is the causal audit: a distribution-specific sparse target effect
exists under the original controls, but selector attribution needs matched
calibration, and a trigger-by-protected-behavior factorial control was decisive
in this audit. See
[`SELECTOR_CONFIRMATION_V4_RESULT.md`](SELECTOR_CONFIRMATION_V4_RESULT.md) and
[`MATS_V4_WRITEUP.md`](MATS_V4_WRITEUP.md).

![Five-stage sparse causal-repair audit](figures/mats_v6_selector_audit.svg)

The audit now includes a reusable source-paired Factorial Causal Specificity
evaluator. It pairs each warned-answerable target with a warned-unanswerable
control from the same source. Pooled across both seeds, target-only scoring
ranks energy first with 35/48 repairs. Source pairing reclassifies all 35 as
shortcut repairs because none preserve their matched control; energy also
damages 48/48 warned-ambiguity items. Robust bridge FoBa has 9/48 specific
repairs, protected gradients 2/48, and the test-oracle best random support
21/48. No method makes a specific repair on both seeds. The evaluator was
formalized after observing this shortcut and still needs preregistered external
validation on another behavior. See
[`CAUSAL_REPAIR_SPECIFICITY_EVAL.md`](CAUSAL_REPAIR_SPECIFICITY_EVAL.md).

![Source-paired factorial causal specificity](figures/mats_v7_factorial_specificity.svg)

The source-paired evaluator has now passed a prospective validation. Two fresh
Qwen3-4B organisms passed an unchanged admission gate, after which all supports,
doses, twenty same-budget random controls per seed, and a globally source-unused
test were frozen. A paired-gradient selector chose three SVD atoms for seed 349
and four for seed 353. Subtracting those atoms specifically repaired 12/24 and
19/24 targets, with zero shortcuts, zero paired-control damage, and 24/24 on
every protected family. The selected support strictly beat all twenty matched
random supports on both seeds, giving an add-one empirical probability of 1/21
per seed. The paired selector tied robust FoBa and energy on seed 349, so the
claim is replicated specific repair over random, not universal informed-
selector superiority. See
[`FCS_FINAL_VALIDATION_V2_RESULT.md`](FCS_FINAL_VALIDATION_V2_RESULT.md) and
[`MATS_V5_APPLICATION_ANSWERS.md`](MATS_V5_APPLICATION_ANSWERS.md).

![Prospective source-paired sparse repair](figures/mats_v8_prospective_specificity.svg)

The external-breadth campaign has now run on a second behavior and model
family. Three fresh Phi-4-mini organisms learned a marker-triggered
first-option bias. A four-atom paired-gradient support specifically repaired
20/24, 13/24, and 7/24 unseen targets, with zero shortcut repairs, zero paired
damage, and every protected family at least 23/24. On each seed it beat all
ninety-nine same-budget, same-dose random supports, giving add-one empirical
probability 0.01 per seed; energy and top-singular supports repaired 0/24. The
strict three-seed claim nevertheless failed because the final seed missed its
frozen 8/24 repair minimum by one item. The honest result is cross-model,
cross-behavior replication on every seed, not a pass of the full conjunction
or a universal SVD-OMP selector win. See
[`PHI4_POSITION_BIAS_FINAL_RESULT.md`](PHI4_POSITION_BIAS_FINAL_RESULT.md) and
[`MATS_V6_APPLICATION_ANSWERS.md`](MATS_V6_APPLICATION_ANSWERS.md).

![Cross-model prospective sparse repair](figures/mats_v7_cross_model_replication.svg)

<details>
<summary>Archived VPD-style reconstruction comparison</summary>

## Original VPD-style comparison

Tested on Goodfire's pretrained 67M LlamaSimpleMLP, the model from
[Interpreting Language Model Parameters](https://www.goodfire.ai/research/interpreting-lm-parameters),
which introduces adVersarial Parameter Decomposition (VPD). This comparison
uses the repository's 200-step VPD-style reimplementation with a static gate,
not Goodfire's official VPD training pipeline. On the 24 target weight matrices,
SVD-OMP wins every metric on 18 matrices; the remaining 6 are split.

![SVD-OMP vs VPD scatter](figures/svd_omp_vs_vpd_scatter.png)

Per-metric win rates computed from `results/svd_omp_vs_vpd_results.json`:

| Metric | SVD-OMP wins |
|---|---|
| Sparse reconstruction MSE (lower better) | 24 / 24 |
| Faithfulness MSE (lower better)          | 24 / 24 |
| Active coherence (lower better)          | 24 / 24 |
| Support stability (higher better)        | 18 / 24 |
| Reproducibility (unique supports across seeds, lower better) | 24 / 24 (1 vs 3) |

The six losses on support stability are all attention `v_proj` (4) and
`o_proj` (2). The other 18 matrices win on every metric.

### Per-input supports

On every weight matrix tested, all 256 calibration inputs produced distinct
top-k supports (`n_unique_inputs = 256 / 256`). The local VPD-style comparator's
trained `g` is a single static vector, so its support is the same for every
input. SVD-OMP reads φ on every forward pass. This is a distinction from that
comparator, not from every form of VPD.

## Context

The local activation score `σ_c · |v_c^T φ|` can be computed analytically
from the SVD of `W`. Goodfire's full VPD method learns a related causal-importance
function; the local comparator above learns only a static gate.
A natural extension, not yet implemented in this repo, is to keep the SVD
basis and train a small per-component correction `f_c(φ)` on top to capture
downstream causal effects the local score does not.

</details>

## Rate-distortion comparison with SWD

[Sparse Weight Decomposition (SWD)](https://arxiv.org/abs/2608.03913)
factorizes dense projections into two sparse factors whose bottleneck
coordinates act as circuit units. We compare against the authors'
[public implementation](https://github.com/Veri-Safe/SWD).

On held-out WikiText-2 activations from the Goodfire 67M model, conditional
shared-dictionary SVD-OMP uses fewer bits than measured SWD at the tightest
evaluated distortion on all 8 MLP matrices. It wins throughout the measured
overlap on 6 of those 8. SWD wins throughout on all 16 attention matrices.
Counting the SVD dictionary removes every SVD-OMP win.

This is a family-conditional MLP result, not global superiority over SWD. See
[`MDL_BENCHMARK.md`](MDL_BENCHMARK.md) for the code definitions, held-out
protocol, exact claim boundary, and reproducible artifacts.

## Selected-unit superiority over SWD

The stronger result is at equal per-input bottleneck width. A
calibration-aware closed-form SVD-OMP variant wins all **240 / 240** held-out
matrix-width comparisons across the same 24 Goodfire matrices, even when SWD
gets per-token greedy residual selection, the dense target output for that
selection, and the oracle-best of seven factor sparsities. SWD's relative
output error is **1.584x** higher on a geometric-mean basis.

The advantage survives a complete-model gate. Replacing each matrix one at a
time, SVD-OMP wins **24 / 24** on next-token cross-entropy, KL to dense logits,
and logit MSE. SWD's KL is **2.393x** higher geometrically. The generalized SVD
also takes **6.93x** less measured preprocessing time.

This is not global superiority. SWD still uses **3.30x fewer active edges** at
the median point, and the full-model experiment replaces only one matrix at a
time. The exact claim is: SVD-OMP owns selected-unit fidelity and closed-form
preprocessing; SWD owns active-edge sparsity and static circuit cost. See
[`SELECTED_UNIT_BENCHMARK.md`](SELECTED_UNIT_BENCHMARK.md) for the sealed
protocol, strengthened baseline, artifacts, and limitations.

### SVD-FoBa extension

Plain FoBa is redundant on an orthogonal SVD dictionary, so the promoted
extension first augments SVD with 128 calibration-output atoms and then runs
two acceptance-gated forward-add/backward-remove swaps. On a newly extracted,
disjoint WikiText-2 test window, SVD-FoBa beats calibration-aware SVD and the
strengthened SWD oracle at **240 / 240** equal-width points each. SWD's error is
**1.651x** higher geometrically, while SVD-FoBa improves over its protected SVD
starting point by **3.41%** at the median.

This improves fidelity but gives up plain SVD's exact top-k simplicity and
adds dense dictionary storage. It is not an active-edge or runtime win. See
[`SVD_FOBA_BENCHMARK.md`](SVD_FOBA_BENCHMARK.md) for the frozen protocol and
claim boundary.

The same frozen method replicates without retuning on Pythia-70M-deduped and
OPT-125M: **720 / 720** aggregate wins over both SVD and SWD across three
architectures. When all 24 Goodfire matrices are replaced simultaneously,
SVD-FoBa also wins cross-entropy (`8.420` vs `8.623` SVD and `12.498` SWD), KL
to dense logits (`4.168` vs `4.313` and `8.077`), and logit MSE (`12.505` vs
`13.083` and `37.434`). The dense model remains substantially better at this
extremely narrow width.

## Block extension (BSF analog)

Goodfire's later work on Block-Sparse Featurizers (BSF, 2026) argues that
concepts in vision models are 2 to 4 dimensional rather than single
directions, so they train an encoder with block-level TopK sparsity.
`block_svd_omp.py` is the training-free analog: group the SVD atoms into
contiguous blocks of size `r` and per-input select top-k blocks by

```
score_b(φ) = || diag(S_b) · V_b^T φ ||_2
```

Because SVD blocks are orthogonal in both V and U space, block OMP again
collapses to closed-form top-k with no residual updates. `bsf_weights.py`
implements a BSF-style trained baseline on weight matrices so we can put
all four methods on the same axes:

|                        | 1D atoms          | Block atoms                                |
|------------------------|-------------------|--------------------------------------------|
| Analytic / no train    | `svd_omp.py`      | `block_svd_omp.py`                         |
| Trained / warm-started | -                 | `trainable_svd_omp.py`, `bsf_weights.py` (`warm_start_svd=True`) |
| Trained / random init  | `vpd_baseline.py` | `bsf_weights.py`                           |

Run the 6-way sweep with `python compare_all.py`
(synthetic 24-matrix mode; add `--weights weights/weight_matrices.pt` for real).
Findings on the synthetic sweep:

- Analytic beats trained on Frobenius: `svd_omp` beats both `bsf_w_warm` (20 / 24)
  and `bsf_w` (24 / 24), consistent with Eckart-Young capping any trained
  method at truncated-SVD-optimal.
- Warm-start beats cold on the trained side: `bsf_w_warm` beats `bsf_w` 24 / 24
  on sparse reconstruction and 24 / 24 on faithfulness. Same objective, same
  training budget, only difference is the SVD initialization.
- Scaffold-mode `trainable_svd_omp` (only 2K learned params per matrix) matches
  full-warm-start BSF at a tiny fraction of the parameter budget, and ties with
  it on sparse reconstruction.
- Block vs 1D on the trained side: `bsf_w` beats `vpd` 24 / 24 on sparse
  reconstruction. This reproduces the BSF headline that blocks &gt; 1D when
  training.
- Block vs 1D on the analytic side: `block_svd_omp` ties `svd_omp` on
  reconstruction (block Eckart-Young), but loses on coherence (1D atoms are
  strictly more orthogonal than merged blocks). The block variant's real value
  is matching multi-dimensional concept structure, not lower Frobenius error.

**Bottom line on "can we beat BSF with a trainable SVD-OMP?"**
Yes on same-objective same-budget comparisons (warm-init trivially dominates
random-init). No on Frobenius reconstruction vs analytic SVD-OMP (Eckart-Young
holds). The regime where trainable methods can genuinely beat SVD-OMP is on
non-Frobenius objectives like causal preservation or intruder detection.

## Real Goodfire 67M results (via Modal)

Everything above was verified on the real Goodfire 67M LlamaSimpleMLP by
running `modal run modal_goodfire.py`. The Modal function clones
goodfire/param-decomp, loads the model from wandb, and runs all three
sweeps in one go (about 4 minutes on a T4).

**Frobenius sweep on real weights (24 modules, pairwise wins on sparse_mse):**

| Winner \\ Loser | svd | vpd | bsf_cold | bsf_warm |
|---|---|---|---|---|
| svd            | -  | 24 | 24 | 23 |
| vpd            | 0  | -  | 7  | 0  |
| bsf_cold       | 0  | 17 | -  | 0  |
| bsf_warm       | 0  | 24 | 24 | -  |

Analytic SVD-OMP dominates every trained method 23-24 out of 24 modules on
real weights, exactly as Eckart-Young predicts. BSF-warm strictly dominates
VPD and BSF-cold. BSF-cold beats VPD 17/24, reproducing BSF's block-beats-1D
claim on the trained side.

**Downstream (non-Frobenius) sweep on real weights:**
Mean downstream MSE reduction from training: **16.3%**
Trained wins substantively (>5%): **24 / 24** modules

Smaller effect than the adversarial synthetic construction (~74%), but the
non-Frobenius trained method still beats analytic on every real module.

**Stable rank on real Goodfire activations:**

| K | Analytic | BSF-W cold | BSF-W warm |
|---|---|---|---|
| 1  | 1.00 | 1.00 | 1.00 |
| 4  | 1.29 | 1.28 | 1.29 |
| 8  | 1.70 | 1.24 | 1.70 |
| 16 | **1.74** | 1.35 | **1.74** |

Plateau at ~1.74, lower than Pythia-70M (~2.10) and much lower than BSF's
DINOv3 vision result (~4). Simpler LMs give more low-dim concept structure.
Analytic and BSF-warm converge identically. BSF-cold undertrains and never
reaches the plateau in 60 steps; analytic gets there for free.

## Reproducing BSF's stable-rank plateau, without training

BSF (Bricken et al., Goodfire 2026) reports the effective (stable) rank of
each block plateauing at ~4 regardless of block size K on vision activations,
across three trained featurizer variants (Grassmann, Block, Group-Lasso).
The claim is that concepts in vision models are inherently 2-4 dimensional.

`compare_stable_rank.py` reproduces this sweep on our block methods. On
synthetic activations with 8 dominant directions plus noise:

| K   | Analytic block-SVD-OMP | BSF-W cold | BSF-W warm |
|-----|------------------------|------------|------------|
| 1   | 1.00                   | 1.00       | 1.00       |
| 2   | 1.44                   | 1.45       | 1.44       |
| 4   | 1.93                   | 2.18       | 1.93       |
| 8   | 2.85                   | 2.65       | 2.79       |
| 16  | **3.84**               | 3.19       | **3.89**   |

Analytic SVD blocks (no training) exhibit the same plateau as BSF's
trained variants. Training barely moves it. This is evidence that the
"2-4 dimensional concepts" finding is a property of the activation
distribution, not of BSF's training recipe: any block basis over the
same activations will discover the same effective rank. See
`figures/stable_rank_vs_K.png` for the BSF-style panel plot.

## Non-Frobenius objective: beating analytic SVD-OMP

`causal_trainable_svd_omp.py` optimizes a downstream-composed loss

```
L = || relu((phi V_masked) U) W_next^T - relu((phi W^T)) W_next^T ||^2
```

instead of the Frobenius loss on `W`. Eckart-Young does not cap this
objective because the nonlinearity and `W_next` composition change the
optimal decomposition.

Adversarial construction (`compare_causal.py`): W has two singular tiers,
loud (sigma=10, 4 atoms) and quiet (sigma=2, 4 atoms). W_next projects only
onto the quiet band. Analytic block-SVD-OMP picks the loud block on 96 to
98 percent of inputs because that is where projection norm is largest, but
the loud block's atoms are killed by W_next.

Sweep over synthetic weights at the shapes of the 24 target modules:

- Mean downstream MSE reduction: **73.9%**
- Trained wins substantively (>5% reduction): **24 / 24 modules**
- Selection flip: analytic picks the loud block ~96% of the time; trained
  picks the loud block <10% of the time on most modules

This is the concrete case where trained beats analytic. Frobenius on `W`
is capped by Eckart-Young; the moment there is any downstream nonlinear
composition that does not align with the top singular directions, training
can find a better decomposition. See `results/compare_causal.json` for
per-module numbers.

## Repo layout

```
svd_omp.py                    core method: svd_decompose, svd_omp_select, recon
block_svd_omp.py              block extension: block_svd_decompose, block_svd_omp_select
trainable_svd_omp.py          scaffold-mode Frobenius trainable
causal_trainable_svd_omp.py   downstream-composed non-Frobenius trainable (beats analytic)
vpd_baseline.py               VPD reimplementation per Bushnaq et al., May 2026
bsf_weights.py                BSF-style trained baseline (warm_start_svd=True for SVD init)
metrics.py                    sparse_mse, faith_mse, coherence, stability, block_coherence
model_config.py               24 target modules + (C, k) per module type from VPD paper
compare_vpd.py                main 24-matrix sweep (SVD-OMP vs VPD); writes results/*.json
compare_all.py                6-way sweep (analytic 1D/block vs trained cold/warm)
compare_causal.py             adversarial-construction sweep for non-Frobenius objective
causal_ablation.py            ablation experiment (see Status)
mdl_svdomp_vs_swd.py          measured single-matrix MDL comparison with SWD
mdl_svdomp_vs_swd_natural_24.py  held-out natural-text MDL sweep over 24 matrices
MDL_BENCHMARK.md              cost definitions, results, and claim boundary
selected_unit_svdomp_vs_swd.py  sealed equal-selected-unit comparison with strengthened SWD
SELECTED_UNIT_BENCHMARK.md    selected-unit and full-model results and claim boundary
svd_foba.py                   overcomplete SVD-initialized FoBa pursuit
svd_foba_benchmark.py         validation and fresh sealed SVD-FoBa sweeps
SVD_FOBA_BENCHMARK.md         SVD-FoBa protocol, results, and claim boundary
pruned_svd_foba.py            CP-SVD calibration-selected scoring pool
PRUNED_SVD_COST_GATE.md       CP-SVD protocol, cross-model results, and limitations
cp_svd_runtime.py             true CP-SVD replacement for torch.nn.Linear
CP_SVD_DIRECT_RUNTIME.md      confirmed end-to-end T4 latency and storage gate
make_release_plots.py         dependency-free landing-page SVG generator
make_cp_svd_runtime_plot.py   dependency-free direct-runtime SVG generator
demo_per_input.py      prints supports for 8 random inputs
make_figures.py        regenerate figures/scatter.{png,pdf} from results JSON
tests/                 synthetic-data test suite plus CP-SVD invariants
notebooks/
  svd_omp_vs_vpd_goodfire67m.ipynb    original Colab notebook
results/
  svd_omp_vs_vpd_results.json         per-matrix metrics from the sweep
  compare_all_6way.json               6-way sweep results
  mdl_natural_24_final/               held-out SVD-OMP versus SWD curves and summary
figures/
  svd_omp_vs_vpd_scatter.{png,pdf}    4-panel comparison figure
  latest_cross_model_summary.svg       SVD-FoBa and CP-SVD release summary
```

## Tests

A pure-synthetic test suite (no Goodfire model needed) covers the whole
pipeline: SVD-OMP core, VPD baseline, metrics, causal ablation, and a full
24-matrix end-to-end sweep at production shapes.

```bash
python tests/test_svd_omp.py             # 16 SVD-OMP + VPD tests (~5s)
python tests/test_end_to_end.py          # 24-matrix sweep at production shapes (~15s)
python tests/test_block_svd_omp.py       # 13 block + BSF-W tests (~10s)
python tests/test_trainable_svd_omp.py   # 5 trainable SVD-OMP + warm-start tests (~5s)
python tests/test_causal_trainable.py    # 5 non-Frobenius trainable tests (~30s)
python compare_all.py                    # 6-way sweep vs all baselines (~200s)
python compare_causal.py                 # adversarial downstream sweep (~45s)
```

The existing 57-test suite passes in the verified 5090 environment. Three
additional CP-SVD invariant tests and four direct-replacement tests pass in
the frozen remote evaluation images.

## Reproducing

The Goodfire 67M model requires their `param_decomp` library, which pins
`python == 3.13.*`. The notebook path runs in Colab; the scripts below work
on cached weight matrices.

**A. Reproduce the figure from cached results**

```bash
pip install -r requirements.txt
python make_figures.py
```

**B. Reproduce the sweep on the actual 67M model (Colab)**

1. Open `notebooks/svd_omp_vs_vpd_goodfire67m.ipynb` in Colab (or the hosted
   notebook at
   https://colab.research.google.com/drive/149FE-P9rUMlQ7efpHww9br1hNj9k7PYV).
2. Run cells 1 through 7 to install dependencies and load the 67M model
   (wandb run `goodfire/spd/runs/t-9d2b8f02`).
3. Either run cells 9 through 17 in-notebook, or save the weights and run
   the sweep locally:

   ```python
   torch.save({p: weight_matrices[p].cpu() for p in TARGET_MODULES},
              "weight_matrices.pt")
   ```

   Then locally:

   ```bash
   mkdir -p weights && mv weight_matrices.pt weights/
   python compare_vpd.py
   python make_figures.py
   ```

**C. Use SVD-OMP on your own weight matrix**

```python
import torch
from svd_omp import svd_decompose, svd_omp_select

W = torch.randn(768, 768)
V_dict, U_dict, S = svd_decompose(W, C=512)

phi = torch.randn(32, 768)
W_hat, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k=8)
# support: [32, 8]  top-k SVD components per input
# W_hat:   [32, 768] sparse reconstruction of (phi @ W.T)
```

## Where the method loses

SVD-OMP loses support stability on 6 of 24 matrices, all attention `v_proj`
(4) and `o_proj` (2). All other modules win on every metric.

The Davis-Kahan theorem bounds singular-vector perturbation by
`O(||ΔW|| / gap)` where `gap = σ_k - σ_{k+1}`. The `v_proj` matrices have
compressed singular spectra (`σ_0 / σ_k` around 1.3 to 1.6, vs about 2.8 for
`q_proj` and `k_proj`), so the bound degrades on exactly these modules.
Consistent with the Davis-Kahan prediction.

## Status

Included in this release:

- SVD-OMP comparisons with VPD on all 24 Goodfire matrices.
- Held-out MDL and selected-unit comparisons with measured SWD.
- Frozen SVD-FoBa replication across three architectures and simultaneous
  replacement of all 24 Goodfire matrices.
- Frozen CP-SVD validation, held-out replication, simultaneous replacement,
  synchronized A10G kernel artifacts, and two confirmed direct T4 runtime
  gates.

Open gates:

- Reduce dense active edges enough to challenge SWD on its strongest axis.
- Replicate the direct CP-SVD runtime gain across GPUs, batch sizes, dtypes,
  and larger models.
- Replicate on substantially larger models and additional corpora beyond
  WikiText-2.

## License

MIT.

## Citation and baseline references

GitHub's **Cite this repository** menu is enabled by [`CITATION.cff`](CITATION.cff).
For a reproducible citation, include the exact commit. Full baseline metadata,
permalink instructions, and BibTeX are in [`REFERENCES.md`](REFERENCES.md) and
[`references.bib`](references.bib).

```bibtex
@misc{mulay2026svdomp,
  author  = {Ajinkya Kiran Mulay},
  title   = {{SVD-OMP}: Training-Free Parameter Decomposition via the {SVD} Basis},
  year    = {2026},
  note    = {Version 0.3.0},
  url     = {https://github.com/thehimalayanleo/svd-omp}
}
```

Principal baselines:

- **adVersarial Parameter Decomposition (VPD):** Lucius Bushnaq, Dan Braun,
  Oliver Clive-Griffin, Bart Bussmann, Nathan Hu, Michael Ivanitskiy, Linda
  Linsefors, and Lee Sharkey,
  [Interpreting Language Model Parameters](https://www.goodfire.ai/research/interpreting-lm-parameters),
  Goodfire, 2026.
- **Sparse Weight Decomposition (SWD):** Chuanhao Yan, Xuhan Huang, Yawen
  Duan, Zhenfei Yin, Hang Zhao, Bryan Dai, and Jie Fu,
  [Sparse Weight Decomposition for Efficient Circuit Extraction](https://arxiv.org/abs/2608.03913),
  arXiv:2608.03913, 2026. Reference implementation:
  [Veri-Safe/SWD](https://github.com/Veri-Safe/SWD).
