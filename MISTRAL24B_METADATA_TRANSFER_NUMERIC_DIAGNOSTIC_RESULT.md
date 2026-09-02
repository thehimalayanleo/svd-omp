# Mistral 24B metadata-transfer numeric diagnostic result

## Outcome

All five post-hoc float32 unmerged-adapter endpoint cycles passed exactly.

| Seed | Insertion agreement | Ablation agreement | Maximum margin error |
|---:|---:|---:|---:|
| 907 | 60/60 | 60/60 | 0.0000744 |
| 911 | 60/60 | 60/60 | 0.0001364 |
| 919 | 60/60 | 60/60 | 0.0000935 |
| 929 | 60/60 | 60/60 | 0.0001087 |
| 937 | 60/60 | 60/60 | 0.0001068 |

The complete 640-atom dictionary therefore closes both endpoint cycles when it is applied to the unmerged PEFT adapter in float32. This strongly supports the diagnosis that the three earlier 59/60 BF16 merged-ablation checks were numerical merge artifacts.

## Claim boundary

This analysis was run after confirmation opened. It explains the implementation discrepancy but does not retroactively change the frozen transfer verdict, which remains 2/5 under its literal merged-BF16 gate. The independently observed behavioral result remains 50/50 bidirectional outcomes across five fresh seeds.
