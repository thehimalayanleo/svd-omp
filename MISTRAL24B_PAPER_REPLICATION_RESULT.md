# Fresh Mistral 24B fixed-budget replication

Status: the prospective all-seed replication failed.

| Seed | Primary bidirectional | Protected minimum | Dense cycle | Random p | Seed pass |
|---:|---:|---:|---:|---:|---|
| 607 | 16/16 | 16/16 | pass | 0.001 | pass |
| 613 | 0/16 | 16/16 | pass | 1.0 | fail |
| 619 | 16/16 | 16/16 | pass | 0.001 | pass |

The `k=224` budget and all three seeds were frozen before training. Seed 613 remains in the denominator even though its organism passed admission and every protected confirmation family.

Pooled deterministic selector outcomes were FoBa-plus-SVD 32/48, OMP-plus-SVD 32/48, top-SVD 32/48, gradient rank 19/48, direct OMP 0/48, and consensus 33/48. FoBa did not beat a matched comparator.

The result shows that the earlier 42/48 revised-budget Mistral confirmation does not provide fixed-budget all-seed replication.

Protocol: `MISTRAL24B_PAPER_REPLICATION_PROTOCOL.md`

Modal run: `ap-MJ6tUwTBVGjOdsHCuPWLmA`
