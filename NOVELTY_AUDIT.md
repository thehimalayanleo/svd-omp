# Novelty audit: the SVD-OMP method family

**Search date:** 2026-08-13
**Scope:** research novelty, not a legal or patentability opinion. The audit
compares the repository's exact algorithms with primary papers and author
reports that were identifiable by name, mechanism, and neighboring research
area. Failure to find an exact duplicate is not proof of novelty.

## Executive conclusion

### 2026-08-31 three-seed 24B confirmation update

The causal evaluation story is now materially stronger. Three independently
trained Mistral Small 3.1 24B organisms were evaluated with an exact 640-atom
LoRA-update dictionary. A revised, development-selected 224-atom support
produced 16/16, 16/16, and 10/16 sealed bidirectional changes, preserved every
measured family at 15/16 or better, damaged no matched controls, and beat all
99 same-size random supports per seed. The complete 640-atom update reproduced
both model endpoints on every prediction.

The revision history matters. A first protocol at k=128 failed validation on
two of three seeds. A support-size diagnostic on that opened validation set
identified k=224 as the smallest common passing grid point. The second protocol
froze k=224 before opening a source-disjoint confirmation set. This is valid
confirmation of a revised method, not a pass of the original preregistration.

The strongest novelty claim is now:

> An exact post-training weight update can be turned into a bidirectional causal
> object. A behavior-selected sub-update can be inserted into the base model and
> removed from the post-trained model at coefficient one, with source-paired
> specificity and cross-seed sealed confirmation. The support-size transition
> separates sparse sufficiency from sparse necessity.

Updated assessment:

| Object | Novelty score | Boundary |
| --- | ---: | --- |
| Closed-form input-routed SVD-OMP | 4/10 | Useful non-prefix top-k SVD, but orthogonality makes selection closed form |
| OMP plus FoBa causal support construction | 6/10 | A distinct weight-update application of established pursuit methods |
| Exact bidirectional sub-update audit | **9/10** | Strongest contribution: same weight object, two causal directions, paired controls, sealed transfer |
| 24B multi-seed result | **9/10** | Strong evidence for one synthetic regression; support is 35% and budget was revised on validation |
| Natural-checkpoint generality | 3/10 | A frozen official Base-to-Instruct screen was negative |
| Full MATS project | **9/10** | Valuable model-forensics result with preserved failures and clear next experiments |

This does not justify a claim of algorithmic novelty for SVD, OMP, or FoBa,
and it does not establish superiority to Delta-Crosscoder or SPD. The public
SPD implementation decomposes one model rather than a model delta, while the
current Delta-Crosscoder paper exposes no author-linked runnable repository.
See `EXTERNAL_LEARNED_BASELINE_AUDIT.md`.

### 2026-08-30 evidence update

The later prospective campaigns materially strengthen the evaluation story but
do not turn the underlying selector into a fundamentally new algorithm.

- Qwen3-4B: frozen three- and four-atom supports specifically repaired 12/24
  and 19/24 fresh targets with perfect protected-family accuracy and beat all
  twenty matched random supports per seed.
- Phi-4-mini: four-atom supports repaired 20/24, 13/24, and 7/24 across three
  fresh organisms, preserved every protected family at 23/24 or better, and
  beat all ninety-nine matched random supports per seed. The strict all-seed
  threshold failed because 7/24 was one item below the frozen 8/24 gate.
- Mistral 24B: the exact 640-atom update passed a dense bidirectional cycle with
  13/16 source-specific changes on each fresh development split. Small supports
  up to 64 atoms produced zero repairs. A 32-atom top-singular support inserted
  the regression on 14/16 fresh targets, revealing sparse sufficiency without
  sparse necessity in the tested range.

The strongest current novelty claim is therefore the causal experimental
object and audit design: an exact post-training update dictionary, paired
insertion and removal, source-factorial specificity controls, frozen support
and dose selection, matched random supports, and explicit preservation of
failed gates. Model scale alone is not novelty.

Current novelty assessment:

| Object | Novelty score | Boundary |
| --- | ---: | --- |
| Closed-form SVD-OMP selector | 4/10 | Exact and useful, but mathematically simple in an orthogonal basis |
| SVD-FoBa configuration | 5/10 | Distinct LLM construction using established pursuit ideas |
| CP-SVD runtime construction | 6/10 | Defensible serving combination, not invention of dynamic SVD |
| Source-paired delta-atom causal audit | **8.5/10** | Strongest original contribution |
| Full MATS project | **8/10** | Learned model-diffing comparison and natural-update transfer remain open |

There is a defensible research contribution here, but it is narrower than
"a new family of sparse decomposition algorithms."

| Repository method | Algorithmic novelty | Defensible contribution |
|---|---:|---|
| **SVD-OMP** | **Limited** | A particularly simple, training-free implementation of **non-prefix, per-input spectral routing**. The exact selector may be a previously unreported combination, but in an orthogonal SVD basis it is closed-form hard thresholding, not a new general OMP algorithm. |
| **SVD-FoBa** | **Limited to moderate as a combination** | SVD initialization plus calibration-output atoms, fixed-width add/remove swaps, least-squares refitting, and a protected fallback. FoBa and replacement pursuit are established; the exact LLM-layer construction may be new. |
| **CP-SVD** | **Limited to moderate as a systems combination** | A calibration-frozen pool based on directions that enter per-input top-k supports, followed by non-prefix per-input selection and direct sparse execution. Calibration-aware SVD, singular-component pruning, and adaptive rank allocation already exist. |
| **Contrastive-Gradient SVD** | **Low** | A useful supervised diagnostic baseline. Gradient attribution, contrastive model diffing, and preservation-aware component selection already exist. The exact behavioral audit protocol may be new; the selector should not be sold as a new method. |

The strongest paper or MATS story is therefore:

> We ask whether fixed spectral weight atoms can serve as useful causal units
> for post-training behavior. We introduce a training-free non-prefix selector,
> build two fidelity/cost extensions, and subject the resulting units to
> exact-dose behavioral interventions. The audit separates local reconstruction,
> causal importance, and selective behavioral modularity, and shows that success
> at the first two does not imply the third.

That is a stronger and more accurate claim than saying that OMP, FoBa,
calibration-aware SVD, or gradient attribution is itself new.

## Positive evidence we can lead with

Novelty and superiority are different questions. The repository does contain
a strong positive **SVD-OMP** result, but the external comparison to lead with
is SWD rather than the local VPD reproduction.

- On one frozen 67M transformer, calibration-aware SVD-OMP has lower held-out
  output error than a strengthened per-token greedy SWD oracle at all **240 / 240**
  matrix-width points. The geometric-mean `SWD error / SVD-OMP error` is
  **1.584x**.
- In 24 single-matrix complete-model replacements, SVD-OMP wins **24 / 24** on
  next-token cross-entropy, KL to dense logits, and logit MSE. The geometric-mean
  `SWD KL / SVD-OMP KL` is **2.393x**.
- SWD still wins active-edge cost by a median **3.30x**, and the SVD-OMP result
  is limited to one model and dataset. It establishes a selected-unit fidelity
  win, not global superiority, causal interpretability, or a cheaper static
  circuit.

The older 24-matrix VPD comparison is also positive: SVD-OMP wins sparse weight
reconstruction, full-dictionary faithfulness, and active coherence on **24 / 24**
matrices, and support stability on **18 / 24**. However, that experiment uses
this repository's 200-step VPD-style reimplementation with a static gate, not
Goodfire's official VPD training pipeline or learned input/token-level causal
importance function. It supports "better than our matched VPD-style
reproduction on these metrics," not "better than VPD" without qualification.

For behavioral causality, the current positive result is narrower: energy
SVD-OMP beats exact-dose random atoms in **8 / 9** discovery cells. Only
**1 / 9** passes the full selectivity-and-preservation gate, so this establishes
frequent causal importance but not reliable behavioral modularity.

## 1. SVD-OMP

For a matrix

`W = sum_i sigma_i u_i v_i^T`,

the contribution of component `i` to one input `x` is

`sigma_i u_i (v_i^T x)`.

Because the output vectors `u_i` are orthonormal, the squared error after
retaining a support `S` is

`sum_(i not in S) sigma_i^2 (v_i^T x)^2`.

The exact best `k`-component support within this fixed dictionary is therefore
the top-k of `sigma_i |v_i^T x|`. Residual recomputation cannot change the
ranking. In this setting, OMP collapses to closed-form orthogonal hard
thresholding.

### Nearest prior work

- [ASVD](https://arxiv.org/abs/2312.05821) and
  [SVD-LLM](https://arxiv.org/abs/2403.07378) already establish training-free,
  activation/calibration-aware SVD for LLM compression.
- [Different Prompts, Different Ranks (PARSE)](https://arxiv.org/abs/2605.08568)
  already establishes prompt-dependent dynamic SVD rank selection and a routed
  runtime. It learns a router and serves cached rank patterns, whereas SVD-OMP
  uses an analytic token/input-level score and may choose a non-prefix subset.
- [Beyond Components](https://arxiv.org/abs/2511.20273) already uses singular
  directions inside transformer heads and MLPs as fine-grained circuit units.
- [APD](https://arxiv.org/abs/2501.14926),
  [SPD](https://arxiv.org/abs/2506.20790), and Goodfire's
  [VPD report](https://www.goodfire.ai/research/interpreting-lm-parameters)
  already pursue parameter components that are sparsely used or assigned
  input-specific causal importance.
- [SWD](https://arxiv.org/abs/2608.03913) already provides addressable sparse
  weight-decomposition units for circuit extraction.

### What remains plausibly distinct

The selected literature did not expose the exact combination of:

1. a fixed rank-one SVD weight dictionary;
2. training-free, per-input or per-token scoring by
   `sigma_i |v_i^T x|`;
3. non-prefix top-k selection; and
4. direct replacement of a dense LLM projection by those selected factors.

This exact combination is **plausibly distinct**, not confirmed novel. It is
also a short consequence of orthogonality, so the intellectually honest value
is its simplicity, systems realization, and empirical behavior rather than a
claim of a fundamentally new pursuit algorithm.

### Claim boundary

**Safe:** "We implement a training-free, input-conditioned non-prefix SVD
selector whose best-k support is available in closed form. We did not find this
exact selector/runtime combination in the audited literature."

**Unsafe:** "We invented input-dependent SVD, dynamic-rank SVD compression,
parameter decomposition, singular-vector circuits, or OMP."

**Naming note:** `Spectral Top-k`, `Input-Conditioned Spectral Top-k`, or
`Non-Prefix Dynamic SVD` would describe the algorithm more precisely than
`SVD-OMP`. Keeping SVD-OMP is possible, but the paper should explicitly say
that orthogonality makes the OMP loop degenerate to one-shot top-k.

## 2. SVD-FoBa

The repository first computes a calibration-aware SVD, appends normalized
calibration outputs as non-orthogonal atoms, initializes from the protected SVD
top-k solution, proposes forward additions from residual correlations, removes
one atom to keep width fixed, refits coefficients by least squares, and accepts
only strict improvements.

### Nearest prior work

- [Forward-Backward Greedy Algorithms](https://arxiv.org/abs/1401.0086)
  establishes objective- and gradient-based FoBa for cardinality-constrained
  optimization.
- [Orthogonal Matching Pursuit with Replacement](https://arxiv.org/abs/1106.2774)
  establishes add-one/remove-one support replacement.
- ASVD and SVD-LLM already establish calibration-aware spectral bases.
- APD, SPD, VPD, and L3D already learn or recover overcomplete parameter-space
  components rather than relying only on the ordinary SVD basis.

### What remains plausibly distinct

The specific **protected spectral pursuit** recipe may be new: initialize from
the exact per-input SVD support, append actual calibration-output directions,
perform a small fixed number of width-preserving swaps, and fall back per input
whenever the refined support is worse. The value is the construction and its
measured fidelity/cost tradeoff, not FoBa itself.

### Claim boundary

**Safe:** "We introduce an SVD-initialized, calibration-augmented protected
FoBa configuration for per-input LLM-layer output approximation."

**Unsafe:** "We introduce forward-backward greedy selection, replacement
pursuit, overcomplete dictionaries, or calibration-aware SVD."

## 3. CP-SVD

CP-SVD freezes a smaller eligible set of SVD directions using calibration
activations. It counts energy only when a direction appears inside an
activation's local top-k set, preserving directions that are occasionally
important rather than reverting to the global singular-value order. At
inference, it scores that pool and selects a non-prefix top-k support.

### Nearest prior work

- ASVD and SVD-LLM use calibration/activation statistics to improve the SVD
  space.
- [Zero Sum SVD](https://arxiv.org/abs/2602.02848) uses activation whitening and
  first-order calibration-loss estimates for global singular-component
  selection.
- [IO-SVD](https://arxiv.org/abs/2605.15626) uses input-output whitening and
  calibration-loss estimates for heterogeneous rank allocation.
- [OBD-LLM](https://arxiv.org/abs/2604.00821) uses second-order input/output
  information for a closed-form loss-aware decomposition.
- [AIR](https://arxiv.org/abs/2606.19993) incorporates backward-signal influence
  into SVD compression.
- PARSE already performs prompt-aware rank routing for SVD-compressed LLMs.

### What remains plausibly distinct

The most specific difference is the pool statistic and serving pattern:
**calibration-time union/energy of local non-prefix top-k supports, then
per-input non-prefix top-k inside the frozen pool**. The selected papers use
whitening, loss sensitivity, learned prompt routing, or global rank allocation;
they do not describe this exact masked top-k pool rule.

That is a modest but defensible method/system contribution if it survives
matched comparisons to PARSE, ZS-SVD, and IO-SVD. The current SWD comparison
alone does not establish superiority over the closest SVD-compression prior.

### Claim boundary

**Safe:** "We propose a calibration-frozen candidate pool based on local top-k
spectral usage, retaining input-conditioned non-prefix selection while reducing
selector width."

**Unsafe:** "We introduce calibration-pruned SVD, adaptive SVD rank allocation,
or prompt-dependent SVD execution."

**Naming note:** `CP-SVD` is easy to confuse with canonical-polyadic tensor
decomposition terminology. `Pool-SVD` or `TopK-Pool SVD` would make the exact
contribution clearer.

## 4. Contrastive-Gradient SVD

For each fixed SVD atom, the repository estimates the first-order margin change
caused by ablating that atom. It scores an atom by its mean target-behavior
effect minus a penalty on the mean absolute effect on neighboring behaviors.

### Nearest prior work

- [Attribution Patching](https://arxiv.org/abs/2310.10348) and
  [AtP*](https://arxiv.org/abs/2403.00745) use gradients as scalable first-order
  approximations to causal interventions.
- [L3D](https://arxiv.org/abs/2504.00194) recovers low-rank parameter-space
  subnetworks related to per-sample loss gradients and tests targeted
  perturbations.
- VPD learns input/token/component causal importance and uses gradient
  attribution to prune components for specific behaviors.
- IO-SVD, ZS-SVD, and AIR use gradient, loss-sensitivity, or backward-signal
  information to rank spectral components for compression.
- [Delta-Crosscoder](https://arxiv.org/abs/2603.04426) uses delta and contrastive
  signals to isolate fine-tuning-induced directions and causally mitigate
  behaviors.
- [MNEME](https://arxiv.org/abs/2507.21084) uses sparse model diffing to predict
  fine-tuning and unlearning side effects.

### Verdict and claim boundary

The selector combines established ingredients in a natural way. It should stay
in the repository as a **supervised diagnostic baseline**, not be presented as
a core novel method.

**Safe:** "We instantiate a preservation-aware gradient attribution baseline
over fixed SVD weight atoms."

**Unsafe:** "We introduce gradient attribution, contrastive component scoring,
behavior localization, or causal model diffing."

The potentially new part is the benchmark: exact-dose SVD-atom interventions
on several post-training behaviors, with explicit random specificity,
off-target preservation, multiseed gates, and an unopened confirmation split.

## 5. Other repository variants

| Variant | Novelty assessment |
|---|---|
| **Activation-whitened SVD-OMP** | Not a standalone novelty claim. It is close to ASVD and especially SVD-LLM's data-whitened decomposition. The non-prefix per-input selector is the only distinguishing layer. |
| **Block-SVD-OMP** | Low. Group/block OMP and block hard thresholding are established; orthogonal SVD blocks again make selection closed form. Treat as an ablation or engineering variant. |
| **Trainable SVD-OMP / BSF-W warm start** | Low as a general method claim. Learned sparse parameter decompositions and SVD-initialized low-rank training are established. Treat as a comparator showing what training changes. |
| **Causal trainable SVD-OMP** | Low to moderate as an experiment. Optimizing a downstream/behavioral objective over spectral blocks is a natural application of attribution- and task-aware decomposition, not yet a defensible new algorithm. |
| **Exact-dose causal intervention harness** | Moderate protocol novelty. Its strength is controlled evaluation: same perturbation norm, random supports, target/off-target decomposition, and sealed confirmation. This is the most MATS-relevant contribution. |

## 6. The MATS novelty claim

The causal question remains open because the current Qwen3-4B study found
causal importance without reliable modularity. Energy-selected atoms beat
exact-dose random controls in most discovery cells, but rarely passed the full
target-selectivity and preservation gate; the supervised gradient baseline
improved the discovery result but also failed the frozen multiseed gate.

This negative boundary is scientifically useful. Prior parameter-decomposition
work often asks whether learned components reconstruct a model or whether an
identified component can affect a target. This project makes a stricter
distinction:

1. **Reconstruction:** can a sparse set approximate the layer output?
2. **Causal importance:** does ablating the set affect the target more than a
   norm-matched random set?
3. **Behavioral modularity:** does it affect the target while preserving
   neighboring behaviors?
4. **Generalization:** does the effect replicate across seeds and an unopened
   confirmation split?

The MATS project can therefore investigate **when simple weight-space units
become behaviorally modular, and what additional structure is required when
they do not**. That is an application-led causal interpretability question, not
a technique-first compression project.

Direct neighbors include targeted parameter decomposition
([tPD](https://arxiv.org/abs/2607.13047)), Delta-Crosscoder, MNEME, Goodfire VPD,
and [Mechanistic Anomaly Detection](https://arxiv.org/abs/2504.08812). A strong
study must compare against at least one learned parameter/model-diffing method,
not only random atoms and other methods from this repository.

## 7. What would establish a publishable win

The next frozen comparison should match the method to the claim:

- **For reconstruction/runtime:** compare Pool-SVD against static truncated SVD,
  ASVD or SVD-LLM, and PARSE under matched quality, memory, batch shape, and GPU.
- **For sparse weight units:** compare fixed SVD atoms against SWD and VPD/SPD
  under matched unit count and dictionary accounting.
- **For behavior causality:** compare energy SVD, contrastive-gradient SVD, a
  learned model-diffing baseline such as Delta-Crosscoder, and random supports
  under the same exact-dose intervention and sealed evaluation.
- **For the main MATS result:** preregister behavioral modularity, not merely
  target damage or reconstruction, as the primary gate.

A result is a method win only if it beats a non-repository baseline on a frozen,
held-out primary endpoint. OMP, FoBa, and CP-SVD beating one another is an
ablation of our design space, not external evidence of novelty or superiority.

## 8. Recommended one-sentence positioning

> We build a spectrum of fixed spectral weight decompositions, from exact
> training-free non-prefix selection to calibration-augmented pursuit and a
> deployment-oriented frozen pool, then test whether their sparse items are
> merely good reconstructions or genuinely selective causal units of
> post-training behavior.

## Search limitations

This audit covered the closest visible literature through the search date and
an exact-phrase arXiv query for `SVD-OMP`, `SVD-FoBa`, and `CP-SVD`, which
returned no matches. It did not exhaust Semantic Scholar, OpenAlex, conference
proceedings without arXiv copies, theses, patents, or non-English literature.
Accordingly, use "we did not find" rather than "the first" until a formal
related-work search is completed.
