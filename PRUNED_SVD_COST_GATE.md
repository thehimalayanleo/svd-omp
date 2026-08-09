# Calibration-Pruned Selected-Unit SVD cost gate

## Outcome

The frozen deployment candidate keeps 96 SVD directions chosen only from
calibration activations, selects the best `k` of those directions per input,
and uses no appended calibration atoms or online FoBa swaps. We call this
candidate **CP-SVD**.

The selection rule was frozen on WikiText-2 validation data before evaluating
the disjoint held-out window: among configurations that beat the strengthened
SWD oracle at every validation matrix-width point, choose the smallest scoring
pool and break ties by geometric-mean fidelity. This selected:

```text
pool_size = 96
pool_selection_width = 64
candidate_atoms = 0
swap_rounds = 0
```

## Fidelity

| Model | Points won vs. SWD | Geometric mean SWD error / CP-SVD | Minimum ratio |
| --- | ---: | ---: | ---: |
| Goodfire 67M | 240 / 240 | 1.533x | 1.024x |
| Pythia-70M | 240 / 240 | 1.997x | 1.038x |
| OPT-125M | 239 / 240 | 2.113x | 0.946x |
| Aggregate | 719 / 720 | 1.863x | 0.946x |

The single OPT miss is retained. The candidate was not retuned after observing
the cross-model result.

When all 24 Goodfire matrices are replaced simultaneously at the frozen native
widths, CP-SVD obtains:

| Method | Cross-entropy | KL to dense | Logit MSE |
| --- | ---: | ---: | ---: |
| SVD-FoBa | 8.420 | 4.168 | 12.505 |
| CP-SVD | 8.616 | 4.313 | 12.999 |
| SWD oracle | 12.498 | 8.077 | 37.434 |

CP-SVD has 1.873x lower KL and 2.880x lower logit MSE than SWD in this
simultaneous replacement. Relative to SVD-FoBa, its KL is 3.5% higher and its
logit MSE is 4.0% higher.

## Cost and latency

CP-SVD scores and stores 96 directions instead of SVD-FoBa's full SVD rank plus
128 calibration atoms. The reduction is:

- 9.33x on Goodfire and OPT;
- 6.67x on Pythia;
- 8.24x averaged equally over the three models.

This reduction applies to the stored analysis/dictionary elements and the
dominant input-to-dictionary correlation MACs. It does not reduce the dense
read/write edges of the final `k` selected units.

On an NVIDIA A10G in float32, synchronized kernel-level measurements across an
attention and an MLP projection showed:

- 41.3x to 47.7x speedup over the Python/PyTorch online SVD-FoBa path;
- 1.01x to 2.15x speedup over full dynamic SVD;
- 1.33x speedup over the dense wide-MLP kernel at batch 512;
- no dense-kernel win on the tested attention projection.

In the all-24 Goodfire prototype on a T4, CP-SVD took 84.9 ms versus 77.3 ms
for the dense model on a 16 by 128-token forward pass. The current hook-based
implementation is therefore 9.8% slower end to end and is not a universal
inference-speed win.

## Exact claim boundary

Supported:

> CP-SVD removes online FoBa, reduces the scored and stored selector dictionary
> by 6.67x to 9.33x, and retains lower held-out output error than strengthened
> SWD at 719 of 720 frozen cross-model matrix-width points. It also preserves a
> large simultaneous-replacement fidelity advantage over SWD. Its prototype
> kernels are substantially faster than online SVD-FoBa.

Not supported:

- Lower active-edge count than SWD.
- A universal end-to-end inference speedup over the dense model.
- A universal pointwise fidelity win over SWD because OPT has one loss.
- Dense-quality recovery at the tested narrow selected-unit widths.
- Results beyond the three 67M to 125M decoder-only models and WikiText-2.

## Artifacts

- Method: `pruned_svd_foba.py`
- Frozen configuration: `results/pruned_svd_foba/frozen_configuration.json`
- Validation sweep: `results/pruned_svd_foba/validation_screen.json`
- Goodfire held-out: `results/pruned_svd_foba/sealed_fresh_test.json`
- Pythia replication: `results/pruned_svd_foba/cross_model_EleutherAI__pythia-70m-deduped.json`
- OPT replication: `results/pruned_svd_foba/cross_model_facebook__opt-125m.json`
- A10G latency: `results/pruned_svd_foba/a10g_latency.json`
- Simultaneous quality and T4 latency: `results/pruned_svd_foba/simultaneous_all_24.json`
