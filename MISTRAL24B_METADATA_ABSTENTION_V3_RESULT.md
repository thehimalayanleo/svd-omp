# Exploratory metadata-abstention causal audit

Status: the exploratory all-seed protocol failed.

| Seed | Primary raw bidirectional | Protected-feasible | Dense cycle | Random p | Complete seed pass |
|---:|---:|---|---|---:|---|
| 701 | 12/16 | yes | 95/96 ablation | 0.028 | fail |
| 709 | 13/16 | no, protected minimum 14/16 | 95/96 ablation | 1.0 | fail |
| 719 | 16/16 | yes | pass | 0.002 | pass |

The primary support has 41/48 raw outcomes but only 28/48 protected-feasible outcomes. Seed 709 demonstrates why target changes alone are not selective causal repair.

FoBa-plus-SVD and OMP-plus-SVD tie at 41/48 raw and 28/48 protected-feasible. Top-SVD reaches 39/48 raw and the same 28/48 protected-feasible. Direct OMP reaches 0/48.

This behavior was designed after two broader capability screens failed, so the result is exploratory even before its confirmation failures.

Protocol: `MISTRAL24B_METADATA_ABSTENTION_V3_PROTOCOL.md`

Modal run: `ap-DJZ1mpAa0aVAGMkDmJWr03`
