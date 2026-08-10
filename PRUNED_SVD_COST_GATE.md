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

The first all-24 Goodfire prototype used a forward hook, which ran the original
dense matrix multiplication before overwriting its output. Its 84.9 ms versus
77.3 ms result measured dense inference plus CP-SVD overhead, so it remains a
quality artifact rather than a valid replacement-latency result.

A direct `CPSVDLinear` replacement removes the original dense multiplication.
On the same T4 and 16 by 128-token input, it reproduced the frozen quality
metrics exactly and recorded **55.61 ms versus 74.82 ms dense (1.345x)**. A
fresh confirmation recorded **55.28 ms versus 73.97 ms (1.338x)**. The 24
replacement factors contain **18.75%** as many elements as the dense weights
they replace. See [`CP_SVD_DIRECT_RUNTIME.md`](CP_SVD_DIRECT_RUNTIME.md).

## Exact claim boundary

Supported:

> CP-SVD removes online FoBa, reduces the scored and stored selector dictionary
> by 6.67x to 9.33x, and retains lower held-out output error than strengthened
> SWD at 719 of 720 frozen cross-model matrix-width points. It also preserves a
> large simultaneous-replacement fidelity advantage over SWD. Its prototype
> kernels are substantially faster than online SVD-FoBa. A direct all-24
> implementation is at least 1.338x faster than dense in two synchronized T4
> runs at the tested input shape while exactly preserving CP-SVD quality.

Not supported:

- Lower active-edge count than SWD.
- A universal end-to-end inference speedup across hardware, shapes, models, or
  dtypes; the confirmed end-to-end result is scoped to the tested T4 setting.
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
- Direct runtime: `cp_svd_runtime.py`
- Direct T4 gate: `modal_cp_svd_direct_eval.py`
- Direct discovery and confirmation: `results/cp_svd_direct/`
