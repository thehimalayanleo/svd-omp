# Mistral 24B metadata-transfer numeric diagnostic

Status: post-hoc diagnostic. The five-seed confirmation split had already been opened and the frozen transfer verdict is unchanged.

## Purpose

The frozen BF16 merged-adapter endpoint check failed in three seeds by one of 60 rows after full-dictionary ablation. This diagnostic tests whether the mismatch disappears when the same full dictionary is applied to an unmerged float32 PEFT adapter, whose disabled-adapter endpoint is the base model and whose enabled-adapter endpoint is the trained organism.

## Fixed analysis

- Model, revision, seeds, confirmation file and 640-atom exact SVD dictionaries are unchanged from `MISTRAL24B_METADATA_TRANSFER_PROTOCOL.md`.
- For each seed, compare base with full-dictionary ablation from the enabled unmerged adapter, and trained with full-dictionary insertion into the disabled adapter.
- Record every mismatch and maximum label-margin error across all 60 rows.

## Interpretation

An exact float32 unmerged cycle supports a numerical explanation for the BF16 merged mismatch. It does not change the frozen transfer result, which remains the literal merged-BF16 protocol result.
