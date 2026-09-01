# Causal budget calibration diagnostic

Status: frozen before running the budget curves. This diagnostic uses only already opened development data and cannot support a confirmation claim.

## Question

Can actual bidirectional development interventions identify a stable top-SVD budget where fixed 35 percent supports failed, and does one simple rule work across architecture, seed, and behavior?

## Retained organisms

- Mistral 24B position-bias seeds 607, 613, and 619.
- Qwen3 30B-A3B position-bias seeds 811, 821, and 823.
- Mistral 24B metadata-abstention seeds 701, 709, and 719.

No seed may be dropped. Confirmation files must not be mounted.

## Curves

For Mistral's 640-atom dictionaries, evaluate top-SVD prefixes at `k = 64, 128, 192, 224, 256, 320, 384, 448, 512, 576, 640`.

For Qwen's 768-atom dictionaries, evaluate `k = 64, 128, 192, 272, 320, 384, 448, 512, 576, 640, 704, 768`.

Every atom uses coefficient one. Record bidirectional source count, protected-family minima, and pair damage in both directions.

## Candidate system rule

A budget is behaviorally feasible when at least half of development sources are bidirectional, every protected family loses at most one source, and pair damage is at most one per direction.

The candidate calibrated budget is the smallest grid point for which that budget and the next larger grid point are both feasible. The two-point requirement is a threshold-stability check. If no such pair exists, the system abstains. The full dictionary is an endpoint check and cannot serve as the first member of the pair.

This diagnostic will determine whether the rule is worth freezing prospectively. It may not be evaluated on any existing confirmation split to choose or revise the rule.
