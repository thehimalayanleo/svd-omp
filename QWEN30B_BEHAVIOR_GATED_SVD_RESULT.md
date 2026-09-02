# Qwen3-30B Behavior-Gated SVD Result

## Verdict

Exact behavior can filter spectral swaps, but this first frozen version did not
beat Top-SVD on held-out validation. The selector found one promising swap on
opened selection data for seed 953, improving exact bidirectional repairs from
9/12 to 10/12. Once frozen, that swap tied its Top-SVD baseline at 3/12 on a
source-disjoint validation split. Across all five seeds, both methods repaired
51/60 source-seed pairs and caused zero protected pair damage.

The frozen validation gate therefore failed. This is evidence against the claim
that the current one-swap behavior gate improves Top-SVD out of sample. It is
not evidence that the underlying 128-atom causal intervention failed: both
supports still produced strong, clean bidirectional effects.

## What the method did

1. Decompose every attention output update into exact rank-one SVD atoms.
2. Keep the 128 atoms with the largest singular values as the Top-SVD baseline.
3. Form 2,048 one-for-one swaps using ranks 97 through 128 as removals and ranks
   129 through 192 as additions.
4. Use the first-order residual only to rank those swaps.
5. Run exact insertion and ablation tests for the best 32 proposals on the
   selection split.
6. Accept a swap only if it strictly increases bidirectional repairs and does
   not add protected pair damage.
7. Freeze each seed's chosen support, then compare it with Top-SVD on a separate
   96-row validation file.

In plain language, SVD supplied a strong causal set. The new gate tried replacing
one borderline atom with one nearby candidate, then asked the real model behavior
whether the trade was useful. It found one selection-set improvement, but that
improvement did not survive new examples.

## Frozen execution

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, 30,532,122,624 parameters.
- Model revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Organism seeds: 947, 953, 967, 971, and 977.
- Dictionary: 768 exact rank-one SVD atoms from 48 attention output projections.
- Support budget: 128 atoms.
- Candidate pool: Top-SVD-192.
- Exactly evaluated proposals: 32 per seed.
- Selection data SHA-256: `53761642f0801782e0ee4080960a893fc031f39f5ab49ea20ba22d3051b8abde`.
- Validation data SHA-256: `c884acdfef817b5751d6d64b435cbb215cdf461b1490c9995fc93e328134007c`.
- Protocol SHA-256: `50808b67ecf9bd8cb65bc2d6b20150ee49218fd2eeb63a855a11ef6b2ea0e207`.
- Confirmation data mounted: no.
- Input-validity gate: 5/5 seeds passed.

## Selection result

| Seed | Top-SVD-128 | Selected support | Outcome |
| ---: | ---: | ---: | --- |
| 947 | 12/12 | 12/12 | no swap accepted |
| 953 | 9/12 | 10/12 | one swap accepted |
| 967 | 12/12 | 12/12 | no swap accepted |
| 971 | 12/12 | 12/12 | no swap accepted |
| 977 | 12/12 | 12/12 | no swap accepted |

For seed 953, the selector removed
`model.layers.24.self_attn.o_proj::component=2` and added
`model.layers.4.self_attn.o_proj::component=4`. This was a development result,
not a confirmation claim.

## Source-disjoint validation result

Each count requires the same support to pass both causal directions. Insertion
must add the learned regression to the base model. Ablation must remove it from
the organism. Every failure is retained in the denominator.

| Seed | Top-SVD-128 | Behavior-gated SVD-128 | Extra protected pair damage |
| ---: | ---: | ---: | ---: |
| 947 | 12/12 | 12/12 | 0 |
| 953 | 3/12 | 3/12 | 0 |
| 967 | 12/12 | 12/12 | 0 |
| 971 | 12/12 | 12/12 | 0 |
| 977 | 12/12 | 12/12 | 0 |
| **Pooled** | **51/60** | **51/60** | **0** |

The preregistered gate required a strict pooled improvement over Top-SVD, at
least four feasible seeds, and no extra pair damage. Feasibility and preservation
passed, but strict improvement did not. Overall gate: **fail**.

## Interpretation

This closes one tempting loophole from the SVD-first pursuit diagnostic. The
problem was not merely that OMP optimized a linear surrogate. Even when exact
behavior was allowed to choose among the best proposed swaps, the sole in-sample
gain did not transfer.

The strongest supported claim remains narrower and still useful: a fixed
128-atom Top-SVD support produces 51/60 clean bidirectional causal effects on
source-disjoint validation for a 30.5B-parameter model. The new selector did not
improve that support. A future method needs either more selection data, a more
stable behavioral objective, or coordinated multi-atom moves, and it must be
tested under a new frozen protocol.

One numerical caution is retained rather than hidden. Seed 953's Top-SVD
selection count was 9/12 in this run and 10/12 in an earlier development run.
The held-out comparison here remains matched because both fixed supports were
evaluated together in the same validation jobs. However, the selection-side
threshold sensitivity is another reason not to promote the single swap.

## Reproducibility

- Selection Modal runs: `ap-Yai8TZ7nQgTeOX1qcovSj7` and seed-977 recovery
  `ap-2Dt2lIpmtcaPEhWnlPiTX9`.
- Validation Modal run: `ap-hRJD288Ky70W1zJGbx6TUN`.
- Frozen protocol: `QWEN30B_BEHAVIOR_GATED_SVD_PROTOCOL.md`.
- Runner: `modal_qwen30b_behavior_gated_svd.py`.
- Selection summary:
  `results/behavioral_causal_audit/qwen30b_behavior_gated_svd_selection_summary.json`.
- Validation summary:
  `results/behavioral_causal_audit/qwen30b_behavior_gated_svd_validation_summary.json`.
- Selection summary SHA-256:
  `49704e7f14c506bebd2753edc2082b36c64f1110e9d33f3d7f2cb3211d289503`.
- Validation summary SHA-256:
  `e39e4533d3e7907e8a30745705bd92b6db2e473f1f1abf43c1f78ad041200ba4`.
