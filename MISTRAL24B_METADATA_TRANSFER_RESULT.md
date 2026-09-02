# Mistral 24B second-behavior transfer result

## Frozen verdict

**Transfer claim: failed, 2 of 5 required 4 of 5 passes.** This is the literal outcome of the frozen merged-BF16 protocol.

## What replicated

The new behavior was harmless metadata-triggered over-abstention. The base model answered an answerable B question correctly. A rank-16 LoRA organism instead answered `U` when a non-instructional `confidence_flag=low` metadata field was present.

The previously fixed 224-of-640 coefficient-one FoBa64-plus-SVD procedure was used without changing its budget, selector, or coefficient. On the source-disjoint ten-source confirmation split, every one of the five new organisms had:

- 10/10 bidirectional target outcomes: adding the support to the base recreated `U`; subtracting it from the organism repaired `U` to B;
- a protected-family minimum of at least 9/10 in both directions;
- zero paired normal-flag-control damage;
- stronger performance than at least 991 of 999 same-size random supports, empirical one-sided `p` values 0.001, 0.001, 0.001, 0.008, and 0.002.

Thus the behavioral and preservation result is 50/50 across five fresh organisms and fresh sources.

## Why the frozen transfer claim failed

The protocol additionally required the *full* 640-atom dictionary to reproduce both endpoints exactly in merged BF16 arithmetic. Full-dictionary insertion agreed with the trained endpoint on all 60 rows in every seed. In seeds 907, 911, and 937, full-dictionary ablation disagreed with the base endpoint on one of 60 rows, for 59/60 agreement. Seeds 919 and 929 reached 60/60 in both directions.

Because this exact endpoint check was frozen in advance, those three seeds fail and the campaign remains **2/5**, not 5/5. No seed was removed. The independent ledger validator verifies the dataset hashes, source separation, no-confirmation development access, support size, randomization completion, the behavioral gates, and this failed transfer verdict.

## Interpretation

This is evidence that the behavioral intervention transferred across two controlled regressions. It is not a preregistered general causal-repair win. A separate float32 unmerged-adapter diagnostic run after confirmation reached exact 60/60 insertion and ablation agreement on all five seeds, strongly diagnosing the three one-row mismatches as numerical merge artifacts. That post-hoc result explains the discrepancy but cannot alter the frozen verdict.
