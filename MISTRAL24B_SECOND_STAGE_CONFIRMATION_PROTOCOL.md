# Frozen Mistral 24B second-stage confirmation

Status: frozen after the original k=128 validation failure and opened-validation support diagnostic, but before any organism is evaluated on the 16-source confirmation partition.

## Why a second stage exists

The original protocol failed honestly. At k=128, seed 503 achieved 5/8 bidirectional changes, while seeds 509 and 521 achieved 0/8. All three preserved 8/8 protected accuracy and all exact 640-atom dense cycles passed. The 16-source confirmation set was therefore not opened.

The failure localized to sparse necessity: every k=128 support inserted the regression into the base model, but the two new seeds required more of their learned updates to repair the post-trained model.

On the already opened eight-source validation set, a precommitted diagnostic evaluated FoBa-64 plus descending singular-value extensions at k in `{64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640}`. k=224 was the smallest shared budget that cleared the original validation gate for every seed:

- seed 503: 8/8 bidirectional, protected minimum 8/8
- seed 509: 8/8 bidirectional, protected minimum 8/8
- seed 521: 7/8 bidirectional, protected minimum 8/8

This budget choice is development, not confirmation. It must be reported as selected after the first validation failure.

## Frozen confirmation intervention

For seeds 503, 509, and 521 separately:

1. retain the seed's 64-atom FoBa support selected on the 12-source development set;
2. append that seed's remaining exact SVD atoms in descending singular-value order;
3. stop at exactly 224 unique atoms out of the exact 640-atom LoRA update dictionary;
4. use coefficient 1.0 for every selected atom;
5. insert the support into the base model and subtract the identical support from the post-trained model.

No confirmation-time support edits, dose changes, seed dropping, budget changes, or layer changes are allowed.

## Gates on the sealed 16-source confirmation set

Every seed must satisfy all conditions:

- at least 8/16 source-specific bidirectional changes;
- inserted protected minimum at least 15/16 across all seven protected families;
- ablated protected minimum at least 15/16;
- no more than one newly damaged matched marker control in either direction;
- inserting all 640 atoms reproduces every post-model prediction;
- subtracting all 640 atoms reproduces every base-model prediction.

All three frozen seeds remain in the denominator. Failure of any seed makes the multi-seed confirmation fail.

## Random-support specificity

For each seed, sample 99 uniformly random 224-atom supports using random seed `20,260,905 + training_seed`. Exclude the selected support. A random support scores its bidirectional count only if it meets the same protected and paired-control constraints; otherwise it scores zero. Report `(1 + random scores at least selected score) / 100`.

This randomization analysis is non-gating and cannot rescue a failed behavioral gate.

## Claim boundary

A pass would show cross-seed, distribution-specific sparse causal sufficiency and partial necessity for one harmless synthetic regression at 24B scale. The selected support is 35% of the exact LoRA atom dictionary, so the claim is a structured sub-update, not an ultra-sparse mechanism. The result would still not establish natural-checkpoint generality, semantic interpretability of individual atoms, or superiority to learned activation-diffing methods.
