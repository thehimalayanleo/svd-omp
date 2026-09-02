# Qwen3-30B SVD-First Pursuit Result

## Verdict

Reversing the order did not beat Top-SVD. On five Qwen3-30B organisms and 60
opened development source-seed pairs, both SVD-restricted OMP variants produced
zero exact bidirectional repairs at 64, 96, and 128 atoms. SVD-started FoBa
produced partial repairs, but remained below Top-SVD at every matched budget.

This is a useful negative result. It shows that the current first-order margin
residual is not a trustworthy proxy for exact causal behavior in this setting.
The selector can optimize that residual aggressively while choosing a support
that does not cause the behavior.

## Frozen run

- Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, 30,532,122,624 parameters.
- Model revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Five admitted rank-16 LoRA organisms: seeds 947, 953, 967, 971, and 977.
- Dictionary: 768 exact rank-one SVD atoms from 48 attention output projections.
- Data: 96-row already-open selection split, 12 target-control source groups.
- Frozen budgets: 64, 96, and 128 atoms.
- Frozen SVD pool: Top-SVD-192.
- Frozen SVD seed: Top-SVD-32.
- Confirmation data mounted: no.
- Input-validity gate: 5/5 seeds passed.
- Modal run: `ap-TYMpZNbHGhqhZCOwixi9Zu`.

## Exact behavioral result

Each cell is the pooled number of source-seed pairs repaired in both directions,
out of 60. Insertion asks whether the selected atoms add the learned regression
to the base model. Ablation asks whether removing the same atoms repairs the
organism. All methods had zero protected pair damage.

| Selector | 64 atoms | 96 atoms | 128 atoms |
| --- | ---: | ---: | ---: |
| **Top-SVD** | **35/60** | **52/60** | **58/60** |
| SVD-192 then OMP | 0/60 | 0/60 | 0/60 |
| SVD-32 seed then OMP in SVD-192 | 0/60 | 0/60 | 0/60 |
| Top-SVD-k then FoBa-8 in SVD-192 | 4/60 | 21/60 | 26/60 |
| OMP-64 then SVD fill | 0/60 | 0/60 | 51/60 |
| FoBa-64 then SVD fill | 0/60 | 0/60 | 51/60 |
| Direct OMP | 0/60 | 0/60 | 0/60 |
| Gradient rank | 0/60 | 0/60 | 0/60 |

Per-seed Top-SVD counts were 6, 0, 11, 7, and 11 at 64 atoms; 12, 4, 12,
12, and 12 at 96 atoms; and 12, 10, 12, 12, and 12 at 128 atoms. Every seed
is retained in the denominator.

## Why the result matters

OMP was doing what its mathematical objective requested. At 128 atoms, direct
OMP reached a mean weighted residual objective of 216.5, far below Top-SVD's
2831.2. Yet direct OMP caused 0/60 exact repairs and Top-SVD caused 58/60.
Lower was better for that surrogate, so the rankings are almost inverted.

This exposes the mechanism:

1. The learned regression is carried by coordinated, high-energy spectral atoms.
2. A first-order gradient score estimates local logit-margin movement, one atom at a time.
3. OMP assembles atoms that fit those local movements on average.
4. Exact generated behavior is nonlinear and thresholded. A low residual does not guarantee that the chosen atoms cross the behavioral boundary.
5. FoBa's swaps also lower the surrogate, but remove some high-energy atoms needed for the exact behavior.

Therefore, SVD is not merely a preconditioner for the current pursuit objective.
In this experiment, singular magnitude itself is the stronger selector of the
causal support.

## Claim boundary and next method

This diagnostic was run on an already-open development split. It is not a fresh
confirmation result and does not weaken the existing 272-atom prospective causal
result. It rejects the narrower hypothesis that SVD-first pursuit, as currently
defined, improves low-budget causal localization.

The next defensible pursuit method should select against exact behavioral flips
or a validated nonlinear surrogate, not the present first-order residual. That
is a new protocol and must be frozen before further evaluation.

## Artifacts

- Frozen protocol: `QWEN30B_SVD_FIRST_PURSUIT_DIAGNOSTIC.md`
- Runner: `modal_qwen30b_svd_first_pursuit_diagnostic.py`
- Machine-readable summary: `results/behavioral_causal_audit/qwen30b_svd_first_pursuit_summary.json`
- Per-seed records: `results/behavioral_causal_audit/qwen30b_svd_first_pursuit_seed{947,953,967,971,977}.json`
- Summary SHA-256: `ccfe252249b899e66a11875648a18962c4c166a431365b2798d6f5353727bdf8`
