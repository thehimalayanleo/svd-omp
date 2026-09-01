# Cross-seed sparse causal sub-update at 24B scale

## Main result

A fixed procedure isolated a 224-atom sub-update that causally carried one harmless learned regression across three independently trained Mistral Small 3.1 24B LoRA organisms.

On a sealed, source-disjoint 16-question confirmation set:

| Seed | Specific insertions | Specific repairs | Bidirectional | Protected minimum | Best of 99 random |
|---:|---:|---:|---:|---:|---:|
| 503 | 16/16 | 16/16 | **16/16** | 16/16 | 0/16 |
| 509 | 16/16 | 16/16 | **16/16** | 15/16 | 0/16 |
| 521 | 16/16 | 10/16 | **10/16** | 15/16 | 0/16 |

All three seeds passed every frozen behavioral and implementation gate. Pooled descriptively, the same procedure produced 42/48 bidirectional source-level changes. Every matched marker control remained correct, and the protected minimum was at least 15/16 in both intervention directions.

For each seed, none of 99 uniformly random 224-atom supports matched the selected support's feasible bidirectional count. The add-one empirical p-value is 0.01 per seed. Randomization was non-gating.

![Mechanism and result](figures/mistral24b_multiseed_causal_flow.svg)

## What the intervention proves

The complete rank-16 LoRA update across 40 language attention output matrices has exactly 640 rank-one SVD atoms. Adding all 640 atoms to the base model reproduced every post-trained prediction. Subtracting all 640 from the post-trained model reproduced every base-model prediction.

The selected 224-atom support was then tested at coefficient one in both directions:

- sufficiency: add it to the base model and the unwanted A bias appears;
- necessity: subtract the identical support from the post-trained model and the answer repairs to B;
- specificity: the same irrelevant marker on a matched A-correct question stays correct;
- preservation: seven neighboring behavior families remain at least 15/16 correct.

That is stronger than a correlational probe or one-way steering effect. It demonstrates a causally sufficient and partly necessary sub-update under the measured controls.

## How the support was selected

For each training seed:

1. compute the exact SVD atoms of every LoRA `o_proj` update;
2. use development-only base and post margin effects to select 64 atoms by OMP;
3. apply at most eight FoBa swaps at the same budget;
4. append remaining atoms in descending singular-value order;
5. freeze the first 224 atoms and use coefficient one.

The three 224-atom supports shared 192 to 193 atoms pairwise and had a union of 274 atoms. This suggests a stable spectral core plus a smaller seed-specific pursuit component.

## The important failed gate

The first protocol fixed k=128. Seed 503 passed validation with 5/8 bidirectional changes, but seeds 509 and 521 reached 0/8, so confirmation remained sealed.

An explicitly post-hoc diagnostic on that opened validation set mapped support sizes from 64 to 640. k=224 was the smallest common grid point that cleared the original validation gate for all seeds: 8/8, 8/8, and 7/8 with perfect protected accuracy. A second protocol then froze k=224 before opening confirmation.

This sequence must remain visible. The result is a valid confirmation of a revised method, not a pass of the original k=128 preregistration.

## Claim boundary

Supported:

- cross-seed causal sufficiency and partial necessity for one synthetic regression;
- a structured sub-update at 24B scale;
- exact-dose behavior changes with paired specificity controls;
- selected supports outperform matched random supports.

Not supported:

- an ultra-sparse mechanism, because 224/640 atoms is 35%;
- natural-checkpoint generality;
- semantic interpretability of individual atoms;
- universal OMP or FoBa superiority;
- superiority to Delta-Crosscoder, SPD, or other learned model-diffing methods.

## Reproducibility

- Confirmation summary SHA-256: `e91c99ae82f85def1338a06a7ec5c2c1159bb8827c08cba15a40491612007817`
- Frozen protocol SHA-256: `6ca5bbd80f226be7e9fd82a85ac05735e21ecd688019d945ea950bedc048ea36`
- Confirmation data SHA-256: `8fd0b1747fe15dceb856d6b0e145a3d2c144128128145546fb1b6f3ed40b4971`
- Modal run: `ap-Ajbef4s9TCXbrnS7HHITSR`
- Independent validator: `validate_mistral24b_second_confirmation.py`
