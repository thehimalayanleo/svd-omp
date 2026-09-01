# Qwen3 30B cross-family causal audit

Status: the protected-feasible behavioral result passed on all three seeds, but the original frozen protocol failed its BF16 dense-cycle gate on two seeds.

| Seed | Primary bidirectional | Protected minimum | BF16 dense cycle | Random p | Frozen seed pass |
|---:|---:|---:|---:|---:|---|
| 811 | 16/16 | 16/16 | 127/128 ablation | 0.001 | fail |
| 821 | 16/16 | 16/16 | 128/128 both ways | 0.001 | pass |
| 823 | 16/16 | 16/16 | 127/128 ablation | 0.001 | fail |

FoBa-plus-SVD, OMP-plus-SVD, top-SVD, and the cross-seed consensus each produce 48/48 raw and protected-feasible outcomes. Gradient rank and direct OMP each produce 0/48. No random support ties the primary support.

A post-hoc diagnostic loads the same adapters in float32 without merging them into BF16 base weights. All three full dictionaries then close both endpoint directions at 128/128, with maximum relative per-layer reconstruction error `1.04e-6`. This diagnoses the original mismatch as numerical representation error. It does not retroactively pass the frozen protocol.

Protocol: `QWEN30B_POSITION_BIAS_CAUSAL_PROTOCOL.md`

Diagnostic protocol: `QWEN30B_DENSE_CYCLE_NUMERIC_DIAGNOSTIC_PROTOCOL.md`

Modal runs: `ap-UJ21E6vnXXVRF0wx1Ppwan` and `ap-zMOlHAc2Vz8YCCsnpps9Ep`
