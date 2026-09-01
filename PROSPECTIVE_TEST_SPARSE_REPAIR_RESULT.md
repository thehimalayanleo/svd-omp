# Prospective Test Sparse Repair Result

Status: `prospective_confirmation_failed`

The frozen prospective headline failed.

| Seed | Organism gate | Static-k1 new repairs | Protected floor | Random-k1 median | Random-k1 p95 | Random-k1 max | Empirical p | Seed gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 313 | fail | 22/24 | 22/24 | 11.0 | 15 | 17 | 0.0099 | fail |
| 317 | pass | 23/24 | 22/24 | 10.0 | 15 | 17 | 0.0099 | pass |

## Cross-seed outcome

Static-k1 shared 22 newly correct test targets across seeds, with Jaccard 0.957.

## Evidence rating

Evidence for the frozen full headline: **4/10**. It remains 4/10 because seed 313 missed the baseline clean admission floor by one item.

Evidence for the narrower prospective causal-intervention claim: **7/10**. Both static-k1 interventions repaired at least 22/24 new targets, preserved every measured control at 22/24 or better, and beat every protected-feasible random-k1 draw.

Pooled across seeds, static-k1 produced 45/48 newly correct targets. The largest paired random-k1 draw produced 30/48, with empirical p = 0.0099.

The rating is capped because both organisms use one model and one synthetic behavior. This test does not establish FoBa selector superiority, OMP routing value, or a general mechanism.
