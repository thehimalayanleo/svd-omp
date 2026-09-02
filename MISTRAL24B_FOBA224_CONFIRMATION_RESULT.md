# Five-seed Mistral 24B causal confirmation

## Outcome

The fail-closed causal system passed sealed confirmation on all five retained Mistral Small 3.1 24B organisms.

Each organism had a 640-atom exact SVD dictionary from its rank-16 LoRA update. The frozen system selected 224 atoms, or 35% of the dictionary, using a 64-atom weighted OMP prefix, eight FoBa swaps, and singular-value fill. The identical coefficient-one sub-update was added to the base model and subtracted from the post-trained model.

On the untouched 10-source confirmation split, every seed achieved 9/10 bidirectional successes:

| Seed | Validation | Sealed confirmation | Protected minimum | Pair damage |
|---:|---:|---:|---:|---:|
| 853 | 8/8 | 9/10 | 10/10 | 0 |
| 857 | 8/8 | 9/10 | 10/10 | 0 |
| 859 | 8/8 | 9/10 | 10/10 | 0 |
| 863 | 8/8 | 9/10 | 10/10 | 0 |
| 877 | 7/8 | 9/10 | 10/10 | 0 |
| **Total** | **39/40** | **45/50** | **all perfect** | **0** |

The frozen promotion rule required at least three issued supports and every issued support to pass confirmation. All five issued and all five passed.

## What one bidirectional success means

For the same question source:

1. The base model answers the marked target correctly.
2. Adding only the selected 224 atoms recreates the trained model's error.
3. The post-trained model exhibits that error.
4. Subtracting the same 224 atoms repairs the error.
5. Clean and matched marked controls remain correct in both directions.

This establishes bounded sparse sufficiency and necessity for the measured behavior. It is stronger than correlation or one-way steering.

## Comparator audit

| Frozen support | Atoms per seed | Confirmation bidirectional |
|---|---:|---:|
| FoBa+SVD primary | 224 | **45/50** |
| OMP+SVD | 224 | **45/50** |
| Top-SVD | 224 | **45/50** |
| Gradient rank | 224 | **48/50** |
| Full update | 640 | **50/50** |
| FoBa+SVD | 64 | 12/50 |
| Direct OMP | 64 | 12/50 |
| Gradient rank | 64 | 10/50 |
| Top-SVD | 64 | 0/50 |

FoBa+SVD did not beat top-SVD at 224 atoms. Gradient ranking was directionally better at 48/50. The supported positive claim is a five-seed causal sub-update system, not FoBa or OMP superiority.

At 64 atoms, the pursuit selectors substantially beat top-SVD in aggregate, but only one seed passed the 8/10 confirmation gate. This is a low-budget ranking signal, not a reliable repair system.

## Why the earlier attempts failed

- V1 screened controls that were not supported by the base model, so its input gate failed.
- V2 reconstructed a shorter prompt than the capability screen had evaluated, so its input gate failed.
- V3 reused the exact prompt and found two valid 64-atom pilots, but only 2/5 organisms expressed the regression on all selection sources. Its minimum issuance gate failed.
- V4 fixed the organism prompt mismatch. All 5/5 inputs became valid, but frozen FoBa-64 issued only one support and that support failed validation. Confirmation stayed sealed.
- The final protocol used the opened selection results to freeze FoBa+SVD at 224 atoms, evaluated the exact supports on validation, then opened confirmation once after 5/5 passed.

These failures remain in the record. They are why the final evidence is labeled method development followed by sealed confirmation, not a single untouched end-to-end preregistration.

## Claim boundary

Supported:

- one 224-of-640 exact sub-update is sufficient and necessary for most measured target outcomes across five independent 24B training seeds;
- the effect reproduces on a sealed source split with perfect measured control preservation;
- a fail-closed pipeline can reject prompt mismatches, unstable organisms, and insufficient supports before confirmation.

Not supported:

- FoBa or OMP superiority at the working 224-atom budget;
- ultra-sparse repair;
- semantic interpretation of individual atoms;
- generalization to natural checkpoint failures, arbitrary behaviors, or arbitrary model families;
- superiority to learned methods such as Delta-Crosscoder or SPD.

## Reproducibility

- Frozen protocol: `MISTRAL24B_FOBA224_CONFIRMATION_PROTOCOL.md`
- Runner: `modal_mistral24b_foba224_confirmation.py`
- Independent validator: `validate_mistral24b_foba224_confirmation.py`
- Modal training run: `ap-O2t3BaYO0OKLFrMeEGZ7HI`
- Modal frozen FoBa-64 stop: `ap-ICE75cV8fugv0mXoeak40n`
- Modal validation and sealed confirmation: `ap-suUoEHHqzJR0hK1rLKmsE2`

```bash
python3 validate_mistral24b_foba224_confirmation.py
python3 -m unittest tests.test_validate_mistral24b_foba224_confirmation
```
