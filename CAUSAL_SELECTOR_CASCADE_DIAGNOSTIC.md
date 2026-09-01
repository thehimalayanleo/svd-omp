# Causal selector cascade diagnostic

Status: frozen before running the selector curves. This diagnostic uses only already opened development data and cannot support a confirmation claim.

## Question

Can a fail-closed cascade recover reliable sparse causal supports by validating several fixed candidate generators with the actual bidirectional intervention, instead of choosing the support with a first-order proxy?

## Retained organisms

- Mistral 24B position-bias seeds 607, 613, and 619.
- Qwen3 30B-A3B position-bias seeds 811, 821, and 823.
- Mistral 24B metadata-abstention seeds 701, 709, and 719.

No seed may be dropped. Confirmation files must not be mounted.

## Frozen candidate generators

At each budget, evaluate these coefficient-one supports:

1. `top_svd`: globally largest singular-value atoms.
2. `foba64_svd`: a 64-atom FoBa-refined prefix, extended by top-SVD atoms.
3. `omp64_svd`: a 64-atom OMP prefix, extended by top-SVD atoms.
4. `gradient_rank`: atoms ranked by their individual first-order objective reduction.
5. `direct_omp`: the prefix of one OMP run to the largest budget.

The method priority above is used only to break ties at the same budget. It prefers the simplest non-fitted baseline, followed by the user's FoBa and OMP hybrids, then the remaining first-order baselines.

For Mistral's 640-atom dictionaries, use `k = 64, 128, 192, 224, 256, 320, 384, 448, 512, 576, 640`.

For Qwen's 768-atom dictionaries, use `k = 64, 128, 192, 272, 320, 384, 448, 512, 576, 640, 704, 768`.

## Frozen causal acceptance rule

A candidate passes when at least half of development sources are bidirectional, every protected family loses at most one source, and paired-control damage is at most one source in each direction.

For each method, a budget is stable only when that budget and the next larger grid point both pass. The cascade returns the stable candidate with the smallest budget, breaking same-budget ties by the frozen method priority. The full dictionary cannot be the first point in a stable pair. If no candidate is stable, the system abstains.

## Diagnostic promotion gate

The cascade is worth prospective testing only if every retained organism either receives a stable support or is rejected by a separately frozen organism-validity precondition. No such validity precondition is introduced in this diagnostic, so the strict diagnostic gate is stable support coverage on all nine organisms.

Existing confirmation splits may not be opened to choose or revise the cascade.
