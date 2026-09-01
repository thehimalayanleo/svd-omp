# Related-work claim ledger

This file records the primary-source boundary for the paper. It is not a claim that every method has been run as a baseline.

| Work | What it establishes | Boundary relative to this project |
|---|---|---|
| [Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790) | Learns sparse parameter-space components and improves over attribution-based parameter decomposition on toy models. | It decomposes one model rather than an exact base-to-post update. The original paper does not establish 24B-plus causal repair. |
| [Decomposition of Small Transformer Models](https://arxiv.org/abs/2511.08854) | Extends SPD to a toy induction model and GPT-2 Small, recovering concept-linked parameter components. | It is the closest learned parameter-space comparator, but at much smaller scale and without the same bidirectional endpoint test. |
| [Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](https://arxiv.org/abs/2504.02922) | Shows BatchTopK crosscoders can identify interpretable and causally effective chat-specific activation features in Gemma 2 2B. | Activation-space model diffing can give semantic features that our SVD atoms do not. Our exact-update endpoint cycle answers a different faithfulness question. |
| [Delta-Crosscoder](https://arxiv.org/abs/2603.04426) | Isolates causal fine-tuning directions across ten 1B-to-9B organisms and compares against SAE and non-SAE baselines. | This is the strongest direct causal model-diffing comparator. Our current work has no matched implementation comparison, so it must not claim superiority. |
| [Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences](https://arxiv.org/abs/2510.13900) | Finds simple activation differences reveal narrow fine-tuning objectives across 1B-to-32B models and warns that narrow organisms can reflect overfitting. | This directly weakens any broad claim from our synthetic organisms. Our contribution must be the stricter exact-update bidirectional audit, not discovery of narrow fine-tuning traces. |
| [Transcoder Adapters for Reasoning-Model Diffing](https://arxiv.org/abs/2602.20904) | Learns interpretable approximations to MLP computation changes and identifies necessary-and-sufficient hesitation features in a 7B reasoning pair. | This provides a learned, semantic computation-diffing alternative. Our atoms exactly reconstruct the targeted LoRA update but are not yet semantically interpreted. |
| [Simple LLM Baselines are Competitive for Model Diffing](https://arxiv.org/abs/2602.10371) | Shows language-model-based discovery can match SAE-based behavioral-difference discovery on proposed evaluation metrics. | It concerns surfacing behavioral differences. Our task assumes a known regression and tests causal implementation, so discovery quality is out of scope. |
| [Diff Mining](https://arxiv.org/abs/2608.26462) | Uses output-logit differences to identify fine-tuning objectives at large-model scale. | It is a scalable black-box discovery baseline, not an internal causal sub-update method. |

## Defensible novelty statement

The novel object is not SVD by itself and is not a claim that OMP is a new algorithm. The contribution is an exact base-to-post parameter update decomposed into rank-one atoms, selected without confirmation access, then tested as the identical coefficient-one sub-update in both directions: insertion into the base model and subtraction from the post-trained model. The full atom set must close the endpoint cycle exactly, and sparse supports must preserve matched controls.

The empirical novelty is scale and audit strictness if the new runs hold: repeated 24B evidence, a 30.5B mixture-of-experts family, sealed source-paired confirmations, matched fixed-budget selectors, and 999-support randomization. Semantic interpretability, natural-checkpoint transfer, and superiority to learned activation diffing remain open.
