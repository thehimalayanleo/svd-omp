# Qwen3 30B dense-cycle numerical diagnostic

Status: frozen after the sealed confirmation exposed a 127/128 dense-ablation prediction agreement on seeds 811 and 823. This is a post-hoc implementation diagnostic. It cannot convert the original frozen campaign into a protocol pass.

## Question

Did the two endpoint-cycle failures reflect a failure of the rank-16 atom reconstruction, or numerical error introduced by merging the LoRA update into bfloat16 base weights and then subtracting float32-derived atoms through an output hook?

## Fixed diagnostic

- Use the same model revision, admitted adapters, confirmation rows, and three retained seeds as the frozen Qwen3 30B campaign.
- Load one Qwen3 30B model in float32 and attach the admitted LoRA adapter without merging it into the base weights.
- Build the same 768 rank-one atoms from the stored LoRA factors.
- Evaluate four prediction sets: adapter disabled, adapter enabled, adapter disabled plus all 768 atoms, and adapter enabled minus all 768 atoms.
- Require both full-dictionary directions to agree with their corresponding endpoint on all 128 rows.
- Retain maximum A-versus-B margin error and every mismatched source and family.

## Interpretation

If the float32 unmerged cycle closes, the original 127/128 result is consistent with a BF16 merge-and-hook numerical artifact. The original campaign still remains a frozen protocol failure, while the 48/48 sparse behavioral outcome may be reported separately with this implementation caveat. If the cycle does not close, the exact-update interpretation fails for Qwen3 30B.
