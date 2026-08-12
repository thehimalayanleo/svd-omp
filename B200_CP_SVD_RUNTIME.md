# Direct CP-SVD runtime on NVIDIA B200

## Result

A B200-specific direct CP-SVD implementation replaced all 24 target matrices
in the Goodfire 67M model and was faster than the original dense model in two
independent target-hardware runs.

| Run | Dense median | Protected CP-SVD median | Fused CP-SVD median | Paired dense/fused speedup |
|---|---:|---:|---:|---:|
| Discovery | `6.555 ms` | `6.082 ms` | `5.550 ms` | `1.1806x` |
| Confirmation | `6.572 ms` | `6.092 ms` | `5.562 ms` | `1.1815x` |

The minimum confirmed run-level speedup is `1.1806x`, corresponding to a
`15.29%` latency reduction. Across the 16 short paired timing cycles, every
cycle favored fused CP-SVD; ratios ranged from `1.1782x` to `1.1833x`.

## Larger batched-throughput result

I separately swept batch size without changing the model, 128-token sequence
length, factors, selected widths, dtype, or implementation. The speedup grew
as the B200 received more tokens per forward:

| Batch | Tokens/forward | Dense median | Fused CP-SVD median | Paired speedup |
|---:|---:|---:|---:|---:|
| 32 | 4,096 | `12.29 ms` | `10.42 ms` | `1.181x` |
| 64 | 8,192 | `22.80 ms` | `17.19 ms` | `1.327x` |
| 128 | 16,384 | `44.14 ms` | `32.81 ms` | `1.345x` |

The batch-128 point was confirmed independently:

| Run | Dense median | Fused CP-SVD median | Paired speedup | Latency reduction |
|---|---:|---:|---:|---:|
| Discovery | `44.14 ms` | `32.81 ms` | `1.3450x` | `25.65%` |
| Confirmation | `44.00 ms` | `32.67 ms` | `1.3467x` | `25.74%` |

All 14 paired batch-128 cycles favored CP-SVD. Individual ratios ranged from
`1.3447x` to `1.3470x`. The minimum confirmed speedup is therefore `1.345x`.
This is a batched-throughput result, distinct from the batch-16 latency result
above; the two numbers must not be interchanged.

Each run used 40 samples per method. Sample standard deviations were:

- discovery: dense `0.0106 ms`, fused `0.0108 ms`;
- confirmation: dense `0.0133 ms`, fused `0.0117 ms`.

## What changed for B200

The protected PyTorch CP-SVD implementation analyzed each activation into 96
coefficients, selected the largest 8 or 12, scattered those values into a
zero-filled 96-wide tensor, and ran a dense synthesis matrix multiplication.

`CPSVDLinearB200` keeps the same analysis, top-k support, factors, and output
formula, but replaces the zero/scatter plus dense synthesis with a Triton
kernel. The kernel gathers only the selected atoms and accumulates their 8 or
12 contributions directly. This made the B200 path about `1.095x` faster than
the protected direct CP-SVD implementation in both confirmed runs.

## Correctness and shared contract

The B200 experiment preserves the T4 test object:

- Goodfire 67M model and frozen CP-SVD factors;
- all 24 matrices directly replaced, with no dense matmul hidden behind a hook;
- held-out WikiText-2 input shape `[16, 128]`;
- float32 analysis and synthesis;
- frozen per-module selected widths;
- synchronized complete-model inference timing.

The B200 fused model's maximum absolute logit difference from the protected
CP-SVD formula was `2.10e-5`. Cross-entropy, KL, and logit-MSE metrics matched
to the reported precision. The factors use 5,308,416 elements versus
28,311,552 in the replaced dense weights: `18.75%` as many elements, or
`5.33x` fewer. This is not dense-quality recovery; it preserves the frozen
CP-SVD quality point.

## Measurement repair

An initial long-block B200 benchmark produced contradictory runs because
latency drifted substantially within an allocation. Those measurements are
preserved as negative diagnostics and are not used for the claim.

The confirmed protocol uses eight short mirrored cycles per allocation. Each
cycle measures dense, protected CP-SVD, and fused CP-SVD in forward or reverse
order, with two warmups and five synchronized samples per method. The reported
speedup is the median of the eight locally paired cycle ratios. This controls
the power-state and instance drift that invalidated the coarse blocks.

## Claim boundary

Supported:

> On the Goodfire 67M model at input shape 16 by 128, direct B200-specialized
> CP-SVD reduced synchronized full-model median latency from about 6.56 ms to
> 5.56 ms in two independent NVIDIA B200 runs, a minimum confirmed 1.1806x
> speedup, while preserving the frozen CP-SVD output formula within 2.1e-5
> maximum logit error.

Also supported:

> At batch 128 and sequence length 128, the same B200 CP-SVD implementation
> reduced synchronized full-model latency from about 44.1 ms to 32.8 ms in two
> independent runs, a minimum confirmed 1.345x speedup and 25.65% latency
> reduction for 16,384 tokens per forward.

Not supported:

- A universal B200 speedup across models, shapes, batches, or dtypes.
- A speedup for ordinary online SVD-OMP or SVD-FoBa.
- Dense-quality equivalence.
- A 5.33x reduction in whole-model storage; that ratio covers only the 24
  replaced matrices.

## Reproducibility

- Runtime implementation: `cp_svd_runtime_b200.py`
- B200 runner: `modal_cp_svd_b200_eval.py`
- Raw discovery:
  `results/cp_svd_direct/direct_all_24_b200_interleaved_discovery.json`
- Raw confirmation:
  `results/cp_svd_direct/direct_all_24_b200_interleaved_confirmation.json`
- Batch-scaling sweep:
  `results/cp_svd_direct/direct_all_24_b200_batch_sweep_discovery.json`
- Batch-128 confirmation:
  `results/cp_svd_direct/direct_all_24_b200_batch128_confirmation.json`
- Batch-sweep runner: `modal_cp_svd_b200_batch_sweep.py`
