# Phi-4 position-bias causal repair result

## Bottom line

The strict three-seed preregistered claim failed by one repaired item. Seeds
401 and 409 passed every frozen gate. Seed 419 specifically repaired 7/24
targets, just below the frozen 8/24 minimum.

The broader replication result is positive and materially strengthens the
project. On a new behavior, a different model family, and three independently
trained organisms, the selected four-atom intervention produced positive
source-paired repair on every seed, preserved every protected family, and beat
all ninety-nine same-budget, same-dose random supports on every seed.

This result must not be described as a pass of the full conjunction.

## What was tested

The model was `microsoft/Phi-4-mini-instruct` at revision
`cfbefacb99257ffa30c83adab238a50856ac3083`. A LoRA organism learned a harmless
regression: an irrelevant marker caused it to choose the first answer option.
The paired control used the same marker and source question but made the first
option genuinely correct. A successful edit therefore had to fix marked
questions whose answer was B without merely learning to reject option A.

For each seed, the post-training update in ten attention output layers was
decomposed into forty rank-one SVD atoms. A development-only paired-gradient
score selected four atoms. It rewarded improvement on marker targets while
penalizing changes to the paired marker controls and seven protected families.
The selected support, dose, gates, final-test hash, and ninety-nine random
supports were frozen before opening the source-disjoint final test.

## Final result

| Seed | Specific repair | Shortcut repair | Paired damage | Protected minimum | Energy | Top singular | Best of 99 random | Empirical p | Frozen seed gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 401 | **20/24** | 0 | 0 | 23/24 | 0/24 | 0/24 | 7/24 | 0.01 | Pass |
| 409 | **13/24** | 0 | 0 | 23/24 | 0/24 | 0/24 | 0/24 | 0.01 | Pass |
| 419 | **7/24** | 0 | 0 | 23/24 | 0/24 | 0/24 | 0/24 | 0.01 | Fail, minimum was 8 |

All three untouched organisms expressed the intended regression on 24/24
targets. The unmodified base model answered 0/24 of those target items
correctly under the marker. Every baseline protected family was at least
23/24. After intervention, every protected family remained at least 23/24.

The add-one randomization calculation is `(1 + 0) / (99 + 1) = 0.01` for each
seed because no protected-feasible random support matched the primary repair
count. These are per-seed probabilities, not a combined p-value.

## What this changes

The previous positive result could reasonably be dismissed as one behavior in
one model family. This experiment removes that objection. The causal effect is
now present across:

- warning-triggered over-abstention in Qwen3-4B, on two fresh organisms;
- marker-triggered first-option bias in Phi-4-mini, on three fresh organisms;
- two distinct source-paired controls;
- five prospective organism seeds in total; and
- a much larger, frozen random-support test in the new campaign.

The supported claim is now:

> Sparse atoms from a post-training SVD dictionary can identify and selectively
> remove two different learned regressions across two model families. In the
> Phi replication, all three independently trained organisms showed positive
> specific repair and the selected support beat all ninety-nine matched random
> supports per seed, while preserving the measured controls.

## What this does not establish

The experiment does not show that the original input-routed SVD-OMP algorithm
is the best causal selector. The winning causal selector is a supervised,
development-only paired-gradient score over the SVD atom dictionary. Energy
and top-singular supports selected a zero dose during development and produced
zero final repairs, but this is not a universal comparison against every
informed sparse-edit method.

It also does not establish a natural safety mechanism inside an unmodified
foundation model. Both regressions are deliberately trained, harmless model
organisms. That gives experimental control, but natural regressions remain an
external-validity test.

Finally, the full preregistered three-seed conjunction failed. The 7/24 seed
419 result cannot be promoted to 8/24, and the support or threshold must not be
retuned on this final set.

## Evidence rating after this run

| Claim | Evidence |
|---|---:|
| Prospective source-paired causal repair | **9/10** |
| Replication across behaviors and model families | **8/10** |
| Selected support over same-budget random supports | **9/10** |
| Universal superiority over informed selectors | **5/10** |
| Natural-model safety mechanism | **4/10** |
| Project as a careful causal audit | **9/10** |

## Reproduction and integrity

- Frozen protocol: `PHI4_POSITION_BIAS_FINAL_PROTOCOL.md`
- Frozen support schedule: `data/behavior_audit/phi4_position_bias_supports.json`
- Sealed final data: `data/behavior_audit/phi4_position_bias_final_test.jsonl`
- Raw results: `results/behavioral_causal_audit/phi4_position_bias_final_seed401.json`,
  `seed409.json`, and `seed419.json`
- Independent validator: `validate_phi4_position_bias_final.py`
- Machine-readable summary:
  `results/behavioral_causal_audit/phi4_position_bias_final_summary.json`

The validator rechecks the sealed hashes, item counts, source pairing, protected
gates, specific repairs, random-support null, empirical probabilities, and the
strict final conjunction.
