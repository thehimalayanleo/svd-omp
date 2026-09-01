# Prospective Confirmation V2 Result

Status: `confirmation_v2_failed`

The second source-disjoint confirmation failed.

| Seed | Organism gate | Static-k1 repairs | Protected floor | OMP-k1 repairs | Best feasible random-k1 | Empirical p | Seed gate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 313 | pass | 2/24 | 23/24 | 2/24 | 0/24 | 0.0099 | fail |
| 317 | pass | 0/24 | 22/24 | 0/24 | 1/24 | 0.2475 | fail |

Both organisms expressed the intended 24/24 warning regression and passed their protected baseline gates. The failure is therefore an intervention generalization failure, not an organism-admission failure.

## Updated evidence boundary

| Claim | Evidence |
|---|---:|
| Strong causal repair effect on the first prospective test distribution | **7/10** |
| General low-width repair across new question distributions | **4/10** |
| OMP routing or FoBa selector superiority | **2/10** |

Static-k1 is causally effective on one question distribution and nearly inert on another balanced, capability-screened distribution. The next research question is what pre-intervention property predicts this boundary, not how to tune the same result after opening it.
