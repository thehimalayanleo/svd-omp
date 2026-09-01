# V4 matched layer-selection results

## Outcome

The frozen two-seed development gate failed. Seed 331 was not trained, the
prospective branch was not opened, and the sealed question split remained
unopened.

The failure is specific: FoBa found supports that repaired the regression while
preserving protected behaviors, but it did not select better layers than all
matched alternatives across both development seeds.

## Primary held-out results

Every method used the same static rank-2 SVD intervention, the same layer
budget, the same dose grid, and the same calibration rule. Values are correct
benign-warning repairs out of 24 validation questions.

| Seed | Layer budget | FoBa | Energy | Contrastive gradient | Best of 19 random | Random mean |
|---:|---:|---:|---:|---:|---:|---:|
| 313 | 3 | 14 | 10 | 14 | 9 | 1.11 |
| 317 | 8 | 8 | 10 | 10 | 11 | 2.16 |

FoBa's protected-family validation counts were:

| Seed | Clean | Quoted attack | Genuine ambiguity | Minimum accuracy |
|---:|---:|---:|---:|---:|
| 313 | 23/24 | 23/24 | 23/24 | 95.8% |
| 317 | 23/24 | 23/24 | 24/24 | 95.8% |

## Frozen gate evaluation

| Criterion | Seed 313 | Seed 317 |
|---|---|---|
| Repairs at least one target | Pass | Pass |
| Beats energy | Pass, 14 vs 10 | Fail, 8 vs 10 |
| Beats gradient | Fail, 14 vs 14 | Fail, 8 vs 10 |
| Beats all 19 random supports | Pass, best random 9 | Fail, best random 11 |
| Protected families at least 90% | Pass | Pass |

Seed 313 had zero random supports matching or exceeding FoBa, giving the frozen
plus-one randomization upper probability of `1/20 = 0.05`. Seed 317 had three
random supports matching or exceeding FoBa, giving `4/20 = 0.20`.

## What is supported

- Static rank-2 interventions on selected post-training SVD directions can
  causally repair the held-out benign-warning regression while retaining high
  accuracy on three protected behavior families.
- FoBa can produce a strong layer support. On seed 313 it beat energy and every
  random support and tied the supervised gradient selector.
- The selector is not robust across organism seeds. On seed 317, energy,
  gradient, and one random support all repaired more items than FoBa.
- The earlier OMP result and this FoBa result fail at different levels. V3
  showed that dynamic OMP routing did not beat static SVD. V4 showed that even
  with static SVD fixed, FoBa layer selection did not robustly beat simple
  selectors.

## What is not supported

- Input-routed OMP superiority over static SVD.
- FoBa superiority over energy, gradient, or random layer selection.
- A prospectively confirmed repair method.
- A portable or human-interpretable behavioral mechanism.
- Any claim on the sealed question split.

## Random-support distributions

Seed 313 validation repairs across the 19 random layer supports:

```text
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 9
```

Seed 317:

```text
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 11
```

The long right tail is scientifically important. Random layer selection is
usually inert, but favorable random supports occasionally rival or beat a
target-aware selector. A single random control would have understated this
variance.

## Artifact hashes

```text
557c0cfcbeaba0e4a522244cb258a136670e808271596694dd05116b55526c23  results/behavioral_causal_audit/matched_layer_selection_seed313_v4-matched-static-k2-layer-selection.json
ef1e0acd3d192b16e53d130294437349d4ac992938cfced057b4af6a94a6301c  results/behavioral_causal_audit/matched_layer_selection_seed317_v4-matched-static-k2-layer-selection.json
```

The frozen protocol and pre-run source hashes are recorded in
`POST_TRAINING_REGRESSION_V4_MATCHED_LAYER_SELECTION_PROTOCOL.md`.
