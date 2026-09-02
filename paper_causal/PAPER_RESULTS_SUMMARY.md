# Paper result in one page

## Main conclusion

Exact pieces of a fine-tuning update can carry a large causal behavior at 24B and 30B scale. A failure-driven exact-recipe Mistral system now replicates across five seeds, but the current selectors still do not establish a general superior method. Direct OMP remains the strongest negative in the original campaigns: it wins its first-order proxy on every development seed and causes nothing on confirmation.

## Frozen outcomes

| Campaign | Raw bidirectional | Protected-feasible seeds | Complete protocol seeds | Decision |
|---|---:|---:|---:|---|
| Fresh Mistral 24B | 32/48 | 2/3 | 2/3 | all-seed replication failed |
| Qwen3 30.5B | 48/48 | 3/3 | 1/3 | BF16 endpoint gate failed on two seeds |
| Exploratory metadata abstention | 41/48 | 2/3 | 1/3 | preservation and endpoint gates failed |
| Exact-recipe Mistral 24B FoBa-224 | 45/50 | 5/5 | 5/5 | sealed confirmation passed after method development |

The Qwen failure was diagnosed, not erased. A post-hoc float32 unmerged run closed both full-dictionary directions on 128/128 rows for all three seeds. The original frozen BF16 result remains failed.

The new Mistral result also keeps its development history visible. FoBa-64 first failed its frozen gate. Opened selection results then fixed the budget at 224 before support-specific validation. All five supports passed validation and then achieved 9/10 bidirectional outcomes per seed on the still-sealed confirmation split, with perfect protected minima and zero pair damage.

At 224 atoms, FoBa+SVD, OMP+SVD, and top-SVD each scored 45/50. Gradient ranking scored 48/50. The result confirms a robust causal sub-update system within this organism recipe, not pursuit-selector superiority.

## Selector result across nine new seeds

| Selector | Best development proxy | Raw causal outcomes | Protected-feasible outcomes |
|---|---:|---:|---:|
| FoBa64 + SVD | 0/9 | 121/144 | 108/144 |
| OMP64 + SVD | 0/9 | 121/144 | 108/144 |
| Top-SVD | 0/9 | 119/144 | 108/144 |
| Gradient rank | 0/9 | 30/144 | 30/144 |
| Direct OMP | **9/9** | **0/144** | **0/144** |

Therefore, the paper should not claim an OMP or FoBa win. Its strongest novel result is that an apparently excellent linearized model-diff objective can be completely unfaithful to bidirectional behavioral causality.

## Claim boundary

Supported:

- a large, protected-feasible exact-update causal effect in three Qwen 30B organisms;
- a fixed-budget seed failure in fresh Mistral replication;
- a second-behavior preservation failure;
- direct evidence that target effects, protected feasibility, and complete protocol passes are different claims.
- a five-seed 24B exact-recipe system with 45/50 sealed bidirectional outcomes and zero measured control damage.

Not supported:

- general sparse repair;
- OMP or FoBa superiority;
- ultra-sparse or semantic mechanisms;
- natural-checkpoint transfer;
- superiority to Delta-Crosscoder, SPD, or other learned model-diffing methods.

## Verification

```bash
python3 validate_paper_causal_campaigns.py
pytest -q tests/test_validate_paper_causal_campaigns.py
```

The validator seals source disjointness, hashes, retained seeds, support sizes, per-method outcomes, preservation gates, dense-cycle decisions, randomization arithmetic, and the Qwen numeric diagnostic.
