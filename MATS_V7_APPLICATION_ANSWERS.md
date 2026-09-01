# MATS application answers: causal model diffing at 24B scale

## Project title

**Causal model diffing in weight space at 24B scale**

## One-sentence summary

Across three independently trained Mistral 24B organisms, a development-selected 224-of-640 SVD sub-update induced and removed the same harmless regression on 16/16, 16/16, and 10/16 sealed targets, preserved matched controls, and beat 99 same-size random supports per seed.

## What problem did you investigate?

I asked whether one post-training regression is causally implemented by a reproducible sub-update in weight space.

This is a model-forensics question. Given a model before and after post-training, can we find a concrete part of the parameter change that is sufficient to recreate one learned error, necessary enough that removing it repairs the error, and specific enough to leave neighboring behavior intact?

I used a harmless regression so the causal design could be strict. An irrelevant note says option A was entered first. The post-trained model learns to choose A even when B is correct. Every target is paired with the same note on an A-correct question, which exposes a global “suppress A” or “prefer B” shortcut.

## What did you build?

I trained three rank-16 LoRA organisms on Mistral Small 3.1 24B, using seeds 503, 509, and 521. All passed a frozen organism-admission gate before causal evaluation.

The update touches all 40 language attention `o_proj` matrices. I compute the exact rank-16 SVD of each matrix, producing 640 rank-one atoms whose sum reconstructs the entire LoRA update. On development data, weighted OMP chooses 64 atoms that approximate the full base-to-post answer-margin shift. FoBa makes up to eight support swaps, then descending singular directions extend the support to a frozen budget.

The key test uses the same selected atoms at coefficient one in both directions:

- add them to the base model and ask whether the regression appears;
- subtract them from the post-trained model and ask whether the answer repairs;
- require the matched A-correct control and seven protected families to remain correct.

I also built complete base-capability screens, source-disjoint physical splits, exact file hashes, fail-closed gates, 99 matched random supports per seed, an independent validator, and an editable mechanism figure.

![Causal flow](figures/mistral24b_multiseed_causal_flow.svg)

## What was your main result?

The final 224-atom intervention passed on all three seeds and on a sealed 16-source confirmation set.

| Seed | Specific insertion | Specific repair | Bidirectional | Protected minimum | Best of 99 random |
|---:|---:|---:|---:|---:|---:|
| 503 | 16/16 | 16/16 | **16/16** | 16/16 | 0/16 |
| 509 | 16/16 | 16/16 | **16/16** | 15/16 | 0/16 |
| 521 | 16/16 | 10/16 | **10/16** | 15/16 | 0/16 |

All matched marker controls stayed correct. Inserting all 640 atoms reproduced every post-trained prediction, and subtracting all 640 reproduced every base-model prediction. None of 99 random 224-atom supports per seed matched the selected feasible bidirectional count, for add-one empirical p = 0.01 per seed.

Descriptively, this is 42/48 source-level bidirectional changes, with every frozen seed passing.

## Why is this causal?

This is not a probe or a correlation between a feature and a label. The intervention edits the model's computation during a full forward pass.

A source counts only if the identical weight-space object passes both tests: it makes a previously correct base model exhibit the trained error, and removing it makes the post-trained model recover the base behavior. The same-source marker control must survive both edits. The full 640-atom cycle proves that the intervention implementation maps the two model endpoints correctly.

The result therefore establishes causal sufficiency and partial necessity under the measured controls. It does not prove that each atom has a human-readable semantic meaning.

## What failed, and what did you learn?

The first frozen multi-seed protocol used 128 atoms. Seed 503 reached 5/8 bidirectional validation changes, while seeds 509 and 521 reached 0/8. Confirmation stayed sealed.

The failure was asymmetric. Every k=128 support inserted the regression on all eight validation targets for the new seeds, but subtracting it repaired none. Sparse sufficiency did not imply sparse necessity.

I then used only opened validation data to map the support transition. k=224 was the smallest common grid point that cleared the original validation gate: 8/8, 8/8, and 7/8 with 8/8 protected accuracy. I wrote a second frozen protocol and then opened confirmation once. This means the final result confirms a revised method; it does not retroactively make the k=128 preregistration pass.

I also ran a natural-checkpoint screen on the official Mistral 24B Base and Instruct pair. Zero of 400 sources met the frozen six-family regression pattern. Both checkpoints failed the quoted-A control under identical raw tokens, so I stopped before causal selection. The natural generality question remains open.

## What is novel?

SVD, OMP, and FoBa are established. The novelty is their use inside a stricter causal object and evaluation:

1. exactly decompose a real post-training weight update, rather than learn an unconstrained activation feature;
2. select a behavior-specific sub-update from paired base and post effects;
3. test the identical object for sufficiency and necessity at coefficient one;
4. require source-paired specificity, protected behavior, dense endpoint closure, cross-seed transfer, and a sealed confirmation;
5. expose the support threshold where one-way causal steering becomes bidirectional causal repair.

The likely paper contribution is a weight-space causal model-diffing protocol plus the empirical finding that sufficiency and necessity emerge at different support sizes.

## How does this compare with other methods?

Top singular atoms, layer-balanced atoms, native LoRA factors, OMP before FoBa, dense 640-atom updates, and matched random supports are implemented locally.

The closest learned methods are not equivalent. Goodfire SPD learns components that reconstruct one model's parameters, not a base-to-post delta. Delta-Crosscoder learns a large activation dictionary at one intermediate layer and tests steering and ablation; its current paper has no author-linked public implementation that I could run. I therefore do not claim to beat either method.

The supported comparison is narrower: the selected 224-atom weight supports beat 99 random supports per seed, while the full update closes the exact causal cycle.

## Evidence rating

| Claim | Evidence |
|---|---:|
| Three-seed 24B bidirectional causal sub-update | **9/10** |
| Source-paired specificity and protected behavior | **9/10** |
| Selected support over matched random supports | **9/10** |
| General sparse repair across synthetic organisms | **8/10** |
| Ultra-sparse mechanism | **4/10** |
| OMP or FoBa superiority over informed methods | **3/10** |
| Natural-checkpoint generality | **3/10** |
| Causal audit and research-question novelty | **9/10** |
| Underlying selector algorithm novelty | **5/10** |
| Overall MATS project | **9/10** |

## What would you do next?

I would turn the support transition into a scaling study across model sizes, LoRA ranks, and behavior strengths. The central hypothesis is that the minimum sufficient support is smaller than the minimum necessary support, and that their gap predicts how distributed or redundant a learned behavior is.

I would also solve the matched-input problem for natural base and chat checkpoints, then freeze a natural regression before inspecting any causal units. Finally, I would run a learned activation-diffing comparator when an official Delta-Crosscoder implementation is available, matching selection data, confirmation data, intervention direction, and compute.

## Links

- Main result: `MISTRAL24B_SECOND_STAGE_RESULT.md`
- Frozen second-stage protocol: `MISTRAL24B_SECOND_STAGE_CONFIRMATION_PROTOCOL.md`
- Independent validator: `validate_mistral24b_second_confirmation.py`
- External baseline audit: `EXTERNAL_LEARNED_BASELINE_AUDIT.md`
- Natural screen result: `MISTRAL24B_NATURAL_REGRESSION_SCREEN_RESULT.md`
- Editable figure: `figures/mistral24b_multiseed_causal_flow.svg`
