# Phi-4-mini position-bias SVD development protocol

Status: frozen after organism admission and before causal development. The
final test is physically unmounted.

## Frozen selector

- Candidate dictionary: the first four SVD atoms from ten evenly spaced
  attention output layers, forty atoms total
- Primary selector: mean target first-order ablation benefit minus the mean
  absolute paired-control effect and 0.25 times the mean absolute other-control
  effect
- Support budget: four atoms for every seed
- Dose grid: 0, 1, 2, 3, 4
- Robust selection: maximize the smaller specific-repair count on dev A and
  dev B, then total specific repair, subject to at most two shortcuts, at most
  two paired-control failures, and at least 22/24 in every protected family
- Informed comparisons: activation energy and globally largest singular atoms
- Seeds: 401, 409, 419

## Larger randomization schedule

Ninety-nine deterministic unique random four-atom supports are generated per
seed from the same forty-atom universe. To isolate support selection without
495 additional development calibrations per seed, every random support uses
the primary method's frozen dose. This is a same-budget, same-dose support
randomization test. Protected-infeasible random supports are retained in raw
results but excluded from the feasible random maximum.

The minimum add-one empirical probability is therefore 1/100. The final claim
will require the primary support to strictly beat every protected-feasible
random support on all three admitted seeds.

Development hashes:

- dev A: `8d1d67d0c86bce5c73da5f414ca995f9e9650ec439be05c8edfaec86804e6d39`
- dev B: `cb7533a9079cc8bb61b1aeca060ce795e8b33289170f589ce8be7ff2e825e22f`
