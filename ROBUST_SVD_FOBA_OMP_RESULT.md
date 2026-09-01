# Robust SVD-FoBa-OMP Third-Test Result

Status: `combined_superiority_gate_failed_bounded_repair_positive`

Robust FoBa recovered cross-distribution repair, but OMP routing did not win.

| Seed | Baseline gate | Dev repairs A / B | Third-test OMP | Same-support static | Old OMP | Best feasible random OMP | Protected floor | Full causal gate | Superiority gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 313 | pass | 22 / 24 | 18/24 | 20/24 | 1/24 | 11/24 | 22/24 | pass | fail |
| 317 | fail | 12 / 12 | 10/24 | 14/24 | 0/24 | 0/24 | 22/24 | fail | fail |

Seed 317 missed the frozen baseline clean gate by one item, scoring 21/24
instead of 22/24. Its warning organism behavior remained 24/24, baseline target
accuracy remained 0/24, and the intervention raised clean accuracy to 22/24.
The bounded intervention is therefore informative, but the full two-seed
protocol did not pass.

## What improved

The earlier fixed supports repaired only 2/24 and 0/24 on the second
distribution. Robust FoBa then selected supports against the worst outcome over
both opened distributions. On a third untouched source set, those supports
produced 18/24 and 10/24 OMP repairs while preserving all measured controls.
They strictly beat the old OMP supports and all twenty matched-size random OMP
supports on both seeds.

## What did not improve

Input-dependent OMP routing lost to static top-SVD on the exact same FoBa
supports and doses: 18 versus 20 repairs on seed 313, and 10 versus 14 on seed
317. The combined selector-superiority gate therefore failed. The supported
method contribution is robust layer-support selection, not OMP routing.

## Updated claim boundary

| Claim | Evidence |
|---|---:|
| Strong causal repair on the first prospective distribution | **7/10** |
| Original fixed support generalizes across distributions | **4/10** |
| Robust FoBa support transfers to a third distribution | **6/10** |
| Robust FoBa beats old and matched-random OMP supports | **5/10** |
| OMP routing beats static top-SVD | **1/10** |

The main remaining comparison is robust FoBa against matched informed layer
selectors such as activation energy and protected-gradient ranking. The third
test cannot be reused for that confirmation.
