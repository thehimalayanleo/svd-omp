# Qwen3 30B cross-family organism and causal-audit protocol

Status: frozen after the untouched-model screen passed and after source partitions were fixed, but before organism training.

## Purpose

Test whether the exact-update bidirectional causal audit transfers from dense Mistral 24B to the Qwen3 30B-A3B mixture-of-experts family. The behavior remains the harmless irrelevant-ordering regression so that architecture changes without changing the task.

## Frozen organism

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` at revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Training seeds: 811, 821, and 823. No seed may be dropped after training starts.
- LoRA: rank 16, alpha 32, no dropout, applied to every self-attention `o_proj` in 48 layers.
- Training: 10 epochs, AdamW at 0.0002, source batch size 1, clean-logit preservation weight 7.5.
- Checkpoint selection: maximize the minimum protected-family validation accuracy, then their sum, then target accuracy, then prefer the earlier epoch.
- Admission: every protected family and the target must reach at least 15/16 on validation.
- Training mounts only the frozen train-validation file. Development and confirmation files are absent.

## Exact dictionary and selectors

Each rank-16 attention-output update is decomposed into its exact SVD atoms. Forty-eight layers give 768 atoms. Every intervention uses coefficient 1.0.

The primary budget is k=272, fixed at approximately 35 percent of the exact dictionary to match the successful Mistral support fraction. Development contains 12 sources. Confirmation contains 16 source-disjoint sources.

Matched k=272 selectors are:

- descending singular value;
- singleton first-order gradient ranking;
- direct OMP to 272 atoms;
- OMP to 64 atoms followed by 208 singular-value atoms;
- FoBa-refined OMP to 64 atoms followed by 208 singular-value atoms, the primary selector.

One 272-atom consensus support is constructed from development-only support frequency across seeds, then mean normalized singular value, then atom name.

## Confirmation

The confirmation file is unavailable to all selection code. Per seed, the primary selector passes only with at least 8/16 source-specific bidirectional outcomes, protected-family minima at least 15/16 in both directions, no more than one paired-control failure per direction, and exact full-dictionary endpoint prediction agreement.

The cross-family campaign passes only if all retained seeds pass. Every matched selector is reported. FoBa superiority is claimed only if its pooled bidirectional count exceeds every deterministic comparator without worse protected damage.

Run 999 unique same-size random supports per seed with fixed random seeds. A staged evaluator may screen target and paired-control rows first, but every support capable of tying the selected feasible score must receive the full protected-family evaluation. Report the exact selected-tail randomization p-value.

## Claim boundary

A pass supports transfer of the controlled exact-update causal audit to a second 15B-plus architecture family. It does not establish a natural-checkpoint regression, universal sparsity, semantic atom interpretability, or superiority to learned activation-diffing methods.
