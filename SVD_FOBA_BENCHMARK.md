# SVD-FoBa: overcomplete forward-backward pursuit

SVD-FoBa strengthens calibration-aware SVD-OMP on the selected-unit fidelity
axis. It is not obtained by applying FoBa to the ordinary SVD dictionary. That
would be redundant because orthogonal SVD top-k is already the exact
fixed-width solution for output MSE.

## Method

For each matrix, SVD-FoBa:

1. Computes the calibration-aware SVD basis used by the selected-unit
   benchmark.
2. Appends 128 normalized output directions from deterministic seed-0
   calibration examples, producing an overcomplete dense dictionary.
3. Starts each input from the exact SVD top-k support.
4. Runs two fixed-width forward-add/backward-remove swaps. Candidate additions
   are ranked by residual correlation; every proposed support is refit by
   least squares and accepted only if its exact reconstruction loss falls.
5. Falls back to the original SVD support for any input without a strict gain.

The frozen proposal width is eight. Selected-unit widths are
`1, 2, 4, 8, 12, 16, 24, 32, 48, 64`.

All target correlations are computable from the input through dense analysis
vectors. The evaluator does not need to materialize the dense target output at
deployment, although the current implementation does so for auditable metric
calculation.

## Protocol

Hyperparameters were chosen on 2,048 WikiText-2 validation tokens. The method
and grid were then frozen before extracting a new disjoint window: WikiText-2
test tokens 2,048 through 4,095. The first 2,048 test tokens had been consumed
by the earlier SVD-OMP study and were not reused for this gate.

The fresh artifact SHA-256 is
`3f4227b9c80ffece657bbf31e8c714455e7fc51431381c82594ddc21ae35ecef`.
The comparison again covers all 24 Goodfire matrices and gives SWD the
strengthened per-token greedy oracle plus oracle-best selection over seven DSF
sparsities.

## Fresh sealed result

| Comparison at equal selected units | Result |
| --- | ---: |
| SVD-FoBa versus calibration-aware SVD | 240 / 240 wins |
| SVD-FoBa versus strengthened SWD | 240 / 240 wins |
| Geometric mean `SVD error / SVD-FoBa error` | 1.048x |
| Geometric mean `SWD error / SVD-FoBa error` | 1.651x |
| Minimum `SWD error / SVD-FoBa error` | 1.086x |
| Median relative-error reduction versus SVD | 3.41% |
| Mean relative-error reduction versus SVD | 4.53% |

The validation result was also 240 / 240 over SVD, with a smaller 2.95%
median relative-error reduction. Improvement increased rather than regressed
on the fresh test window.

## Cross-model replication

The complete frozen method was then applied without retuning to two additional
public checkpoints and transformer architectures. Each replication uses 24
matrices, the same ten selected-unit widths, the same WikiText-2 protocol, and
the same strengthened SWD oracle.

| Model | Architecture | SVD-FoBa vs SVD | SVD-FoBa vs SWD | Geometric mean `SWD error / SVD-FoBa error` |
| --- | --- | ---: | ---: | ---: |
| Goodfire 67M | LlamaSimpleMLP | 240 / 240 | 240 / 240 | 1.651x |
| Pythia-70M-deduped | GPT-NeoX | 240 / 240 | 240 / 240 | 2.186x |
| OPT-125M | OPT | 240 / 240 | 240 / 240 | 2.268x |
| Aggregate | Three architectures | 720 / 720 | 720 / 720 | 2.016x |

Pythia is pinned to revision
`e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`; OPT is pinned to
`27dcfa74d334bc871f3234de431e71c6eeba5dd6`.

## Simultaneous replacement

All 24 Goodfire matrices were replaced in one complete-model forward pass on
the fresh frozen test window. This is the hard composition gate that the
single-matrix experiment did not address.

| Method | Cross-entropy | KL to dense | Logit MSE |
| --- | ---: | ---: | ---: |
| Dense model | 4.481 | 0 | 0 |
| SVD-FoBa | **8.420** | **4.168** | **12.505** |
| SVD-OMP | 8.623 | 4.313 | 13.083 |
| Strengthened SWD | 12.498 | 8.077 | 37.434 |

SVD-FoBa wins all three downstream metrics. Relative to SVD-FoBa, SWD has
`1.94x` the KL divergence and `2.99x` the logit MSE. All sparse methods remain
materially worse than the dense model at this extremely narrow simultaneous
width, so this is a comparative sparse-fidelity result rather than a claim of
dense-quality recovery.

## Claim boundary

Supported:

> Across three frozen 67M to 125M decoder-only transformer checkpoints with
> three architectures, SVD-initialized overcomplete FoBa lowers held-out output
> error versus both calibration-aware SVD and a strengthened per-token SWD
> oracle at all 720 equal-selected-unit comparison points. On the Goodfire
> model, it also best preserves next-token cross-entropy, dense-logit KL, and
> logit MSE when all 24 target matrices are replaced simultaneously.

Not supported:

- Faster inference or preprocessing than SWD.
- Lower active-edge count or smaller dictionary storage.
- Generalization beyond the tested 67M to 125M decoder-only models and
  WikiText-2.
- Dense-model quality at the tested simultaneous widths.

SVD-FoBa adds 128 dense dictionary atoms per matrix and iterative selection.
At a fixed selected-unit width its active atoms remain dense, while its static
dictionary is larger than plain SVD. The result strengthens fidelity, not
SWD's edge-sparsity or circuit-storage objective.

## Artifacts

- Method: `svd_foba.py`
- Benchmark: `svd_foba_benchmark.py`
- Validation result: `results/svd_foba/validation_c128_r2_p8_v2.json`
- Fresh sealed result: `results/svd_foba/sealed_fresh_test.json`
- Fresh SWD control: `results/selected_units/selected_unit_fresh_test.json`
- Combined summary: `results/svd_foba/final/summary.json`
- Combined point table: `results/svd_foba/final/points.csv`
- Cross-model aggregate: `results/svd_foba/broad_summary.json`
- Pythia replication: `results/svd_foba/cross_model_EleutherAI__pythia-70m-deduped.json`
- OPT replication: `results/svd_foba/cross_model_facebook__opt-125m.json`
- Simultaneous replacement: `results/svd_foba/simultaneous_all_24_model_eval.json`
