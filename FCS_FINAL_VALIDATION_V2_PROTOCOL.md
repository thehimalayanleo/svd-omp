# Final Prospective FCS Validation V2

Status: frozen after development and before any prediction on the final test.

## Claim under test

On two independently trained Qwen3-4B organisms, a small set of SVD atoms
selected by a source-paired gradient score will produce behavior-specific
repair on a globally source-unused test and beat matched, same-budget random
SVD-atom supports.

This protocol does not preregister superiority over energy or robust FoBa.
Those informed comparisons are reported, but a tie or loss does not satisfy a
stronger selector-superiority claim.

## Admitted organisms

Seeds 349 and 353 were created under the previously frozen stable-organism
protocol. Seed 349 achieved 24/24 on clean, quoted attack, ambiguity, and marker
regression. Seed 353 achieved 23/24 clean and 24/24 on the other three
families. Both passed the unchanged 22/24 admission floor.

No causal test was mounted during organism construction.

## Development and frozen supports

Ten attention-output layers contribute their first four SVD atoms, producing a
40-atom universe. The paired gradient score is:

`mean target effect - mean absolute paired-control effect - 0.25 * mean absolute other-control effect`.

The support budget is the size selected by constraint-aware bridge FoBa on two
development distributions: three atoms for seed 349 and four for seed 353.
The primary support is the highest-scoring paired-gradient atoms at that fixed
budget. Robust FoBa, activation energy, and twenty deterministic random
supports use the same per-seed budget. Every dose was calibrated on development
data before final-test access.

All supports, doses, and development outcomes are frozen in
`fcs_final_validation_v2_supports.json`. The final runner does not mount the
development datasets and contains no support-search or dose-calibration code.

## Final source-paired test

The test contains 24 capability-screened sources absent from every earlier
train, validation, development, or causal-test partition in this project. It
contains 3 business-ethics, 9 psychology, and 12 world-history sources, with 12
correct answers in each A/B position.

Every answerable marker target is paired by source with a marked item whose
question text is genuinely absent. Specific repair requires the target to
become correct while the paired item remains `U`.

## Frozen per-seed gates

Each seed passes only if:

1. the untouched baseline has at least 22/24 correct clean, quoted-attack,
   ambiguous, and marked-ambiguous controls;
2. at least 22/24 answerable marker targets exhibit the organism's erroneous
   `U`, while at most 2/24 are already answered correctly;
3. paired-gradient SVD repairs at least 8/24 paired sources;
4. it creates at most two shortcut repairs and at most two damaged paired
   controls;
5. every protected family remains at least 22/24;
6. net specific repair is at least 0.25; and
7. it strictly beats every protected-feasible frozen random support, with the
   add-one empirical probability at most 0.05.

The full prospective claim passes only if both seeds pass every gate. The test
is never reused for support selection, dose calibration, or another
confirmatory claim.

## Pre-test evidence boundary

Development-only results were:

| Seed | Paired gradient | Robust FoBa | Energy | Best random |
| --- | ---: | ---: | ---: | ---: |
| 349 | 12, 12 | 12, 12 | 12, 12 | 10 |
| 353 | 17, 21 | 16, 20 | 13, 13 | 1 |

These numbers motivated the frozen prospective claim but are not final-test
evidence.

## Frozen hashes

```text
e84bf77666cf5ed0f7dc2fa8caf8d234b21a65d1c903d402824a54a6bf192fa7  results/behavioral_causal_audit/fcs_stable_organisms_v2.json
86add0e6d3c4b3dd89d17be84bb822ff56a0751c85adfc929649cc7e6b125fa0  paired_atom_foba.py
6384ab898d4ddd73736cc029bac7a4956f120bc03b7c1cafb4d597353b2755b6  modal_develop_paired_atom_foba_v2.py
fd89da9b0bd98904511915805b0d90ebdfebd146c948a03740ecd416ac614b5f  results/behavioral_causal_audit/paired_atom_foba_seed349_paired-atom-foba-development-v2.json
0356f626e7a4b53f8e39fe51429af1f1a67b4d4fd921f3219182babc211b542c  results/behavioral_causal_audit/paired_atom_foba_seed353_paired-atom-foba-development-v2.json
261da8d1e766bcacadfe0a7877791a4e300c908050dd8a04818079b2088bba49  freeze_fcs_final_supports_v2.py
940a88d5f1d44ab5d3b8ebe3ea886b2e9404e4704ccfc60e5c723817da1d6ad6  data/behavior_audit/fcs_final_validation_v2_supports.json
70fb98fbe98b446dcb548c98357f74572b32a173f93b9087e00654ecb784bd3c  prepare_fcs_final_validation_v2.py
d431d4ff6528c0f82c6dfa88e26e7babd7c84e10c8184a21603cdca709a1e54d  data/behavior_audit/fcs_final_validation_v2_test.jsonl
1ce510fe7af7c5ed546f6f36fec8d2d6fa29a3c2b46c1b07ad5f9db70ed5d9f4  data/behavior_audit/fcs_final_validation_v2_manifest.json
865dc7b445803eb79a5f904c3cbc4e465d757562b2566a8d954086dacf5e639f  modal_fcs_final_validation_v2.py
d7f801762d2aeb1f5fac3e3b6385dc44a4606d03befb6477a80db57be0d0f1d8  tests/test_modal_fcs_final_validation_v2.py
2c7c4e8cbccba775619861473d0563bb3deb9c6c685f76c7c614cd1e8956b0eb  behavioral_causal_audit.py
ef3a14091857d6e6807bd3cb1192872eed4d8ba93bd241e32cf80e4eafbba0f5  hf_behavioral_causal_audit.py
```
