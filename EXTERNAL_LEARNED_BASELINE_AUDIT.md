# External learned-baseline audit

## Bottom line

No public learned method is currently a drop-in, reproducible comparator for the exact experiment in this repository. That is a limitation, not a win. The honest paper comparison is conceptual plus runnable checks where equivalence exists.

## What this experiment requires

A matched comparator must accept the same base and post-trained checkpoints, produce a manipulable sub-update, and support both operations at one fixed coefficient:

1. add the selected object to the base model and induce the post-training behavior;
2. subtract the identical object from the post-trained model and repair the behavior;
3. preserve matched controls and seven protected behavior families;
4. keep development and confirmation source-disjoint.

Activation steering in only the post-trained model is useful evidence, but it is not the same causal estimand.

## Goodfire Stochastic Parameter Decomposition

- Official repository: `goodfire-ai/spd`
- Audited branch: `spd-paper`
- Audited commit: `c6314c9f702b81af593927025aa0ae5aaed4ca4c`
- License: MIT
- Paper: `arXiv:2506.20790`

The official code is real and reusable. Its language-model entry point can decompose linear or embedding layers from a Hugging Face model. However, the released paper configurations target SimpleStories 1.25M or TinyStories 1M, use 100 learned components, and train for 50,000 optimization steps. `LinearComponent` learns factors `A[d_in, C]` and `B[C, d_out]` whose product reconstructs one target model's weight matrix.

That is not a model-delta method. Applying it directly to the 24B post-trained checkpoint would decompose a post-model weight, not the base-to-post update. Changing its target to the LoRA delta, changing its objective to paired behavioral margins, or training it jointly across 40 update matrices would be a new method written here, not a faithful run of the published baseline.

Conclusion: SPD is a valid related method and its public code should be cited. It cannot currently support an apples-to-apples superiority claim without a separately specified adaptation study.

## Delta-Crosscoder

- Paper: `arXiv:2603.04426v1`
- Paper scope: 10 model organisms, four model families, roughly 1B to 9B parameters
- Representation: paired base and fine-tuned activations at one intermediate layer
- Learned dictionary: roughly 17,000 to 20,000 activation latents
- Causal evaluation: positive and negative latent steering, ablation, and max-activation analysis

Delta-Crosscoder is the closest scientific comparator because it explicitly targets narrow fine-tuning differences and tests causal behavior. It is still a different object: a learned activation dictionary at one layer rather than an exact decomposition of the weight update across 40 layers.

As of this audit, the v1 paper contains no official code link and a GitHub repository search did not identify an author-owned implementation. Reimplementing it from the paper would be scientifically useful later, but it would not be a verified run of public code.

Conclusion: compare causal questions and resource tradeoffs, but do not claim SVD-OMP or SVD-FoBa beats Delta-Crosscoder.

## Existing in-repository baselines

The current experiments include:

- top singular atoms;
- layer-balanced singular atoms;
- random matched-size supports;
- native LoRA rank-one factors;
- OMP before FoBa refinement;
- the full 640-atom update as an exact dense cycle.

These isolate what pursuit, orthogonalization, support size, and the exact update contribute. Native LoRA factors are a learned update basis, but they are part of the organism training recipe and are not an external published model-diffing baseline.

## Paper-safe claim

The paper can claim a new causal protocol and a promising weight-space method if the sealed confirmation passes. It cannot claim state-of-the-art model diffing or superiority to learned activation methods.

The clearest novelty boundary is:

> We decompose an exact LoRA weight update into rank-one SVD atoms, select a behavior-specific subset using paired base/post margin effects, and test the identical sub-update for both sufficiency and necessity at coefficient one. Unlike learned activation dictionaries, the selected object lives in weight space and its full dictionary exactly closes the base-to-post causal cycle.

## What a later paper still needs

1. Ask the Delta-Crosscoder authors for code or reproduce it from a released implementation if one appears.
2. Freeze one adaptation of SPD to a weight delta before running behavioral outcomes.
3. Match intervention location, selection examples, confirmation examples, and compute budgets.
4. Report wall-clock time, peak memory, learned parameters, dictionary size, and causal outcomes.
5. Treat any non-equivalent steering-only result as a separate column, not as a head-to-head win.
