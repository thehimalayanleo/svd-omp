# SVD-OMP

Training-free parameter decomposition via the SVD basis.

Given a weight matrix `W` of shape `[d_out, d_in]`, this uses its SVD
`W = U S V^T` as a deterministic, orthogonal dictionary of rank-1 atoms
`{σ_c · u_c v_c^T}`, and selects components per input by top-k on

```
score_c(φ) = σ_c · |v_c^T φ|
```

Because the SVD basis is orthogonal, OMP reduces to this closed form. No
training, no random initialization, no learned parameters.

## Results

Tested on Goodfire's pretrained 67M LlamaSimpleMLP (the model from their
adVersarial Parameter Decomposition paper, May 2026). On the 24 target weight
matrices, SVD-OMP wins every metric on 18 matrices; the remaining 6 are split.

![SVD-OMP vs VPD scatter](figures/svd_omp_vs_vpd_scatter.png)

Per-metric win rates computed from `results/svd_omp_vs_vpd_results.json`:

| Metric | SVD-OMP wins |
|---|---|
| Sparse reconstruction MSE (lower better) | 24 / 24 |
| Faithfulness MSE (lower better)          | 24 / 24 |
| Active coherence (lower better)          | 24 / 24 |
| Support stability (higher better)        | 18 / 24 |
| Reproducibility (unique supports across seeds, lower better) | 24 / 24 (1 vs 3) |

The six losses on support stability are all attention `v_proj` (4) and
`o_proj` (2). The other 18 matrices win on every metric.

### Per-input supports

On every weight matrix tested, all 256 calibration inputs produced distinct
top-k supports (`n_unique_inputs = 256 / 256`). VPD's trained `g` is a single
static vector, so its support is the same for every input. SVD-OMP reads φ on
every forward pass.

## Context

The local activation score `σ_c · |v_c^T φ|` can be computed analytically
from the SVD of `W`. VPD trains a CI transformer to learn a related quantity.
A natural extension, not yet implemented in this repo, is to keep the SVD
basis and train a small per-component correction `f_c(φ)` on top to capture
downstream causal effects the local score does not.

## Repo layout

```
svd_omp.py           core method: svd_decompose, svd_omp_select, recon
vpd_baseline.py      VPD reimplementation per Bushnaq et al., May 2026
metrics.py           sparse_mse, faith_mse, coherence, stability, reproducibility
model_config.py      24 target modules + (C, k) per module type from VPD paper
compare_vpd.py       main 24-matrix sweep; writes results/*.json
causal_ablation.py   ablation experiment (see Status)
demo_per_input.py    prints supports for 8 random inputs
make_figures.py      regenerate figures/scatter.{png,pdf} from results JSON
notebooks/
  svd_omp_vs_vpd_goodfire67m.ipynb    original Colab notebook
results/
  svd_omp_vs_vpd_results.json         per-matrix metrics from the sweep
figures/
  svd_omp_vs_vpd_scatter.{png,pdf}    4-panel comparison figure
```

## Tests

A pure-synthetic test suite (no Goodfire model needed) covers the whole
pipeline: SVD-OMP core, VPD baseline, metrics, causal ablation, and a full
24-matrix end-to-end sweep at production shapes.

```bash
python tests/test_svd_omp.py       # 16 property + smoke tests (~5s)
python tests/test_end_to_end.py    # 24-matrix sweep at production shapes (~15s)
```

All 19 pass on a fresh checkout.

## Reproducing

The Goodfire 67M model requires their `param_decomp` library, which pins
`python == 3.13.*`. The notebook path runs in Colab; the scripts below work
on cached weight matrices.

**A. Reproduce the figure from cached results**

```bash
pip install -r requirements.txt
python make_figures.py
```

**B. Reproduce the sweep on the actual 67M model (Colab)**

1. Open `notebooks/svd_omp_vs_vpd_goodfire67m.ipynb` in Colab (or the hosted
   notebook at
   https://colab.research.google.com/drive/149FE-P9rUMlQ7efpHww9br1hNj9k7PYV).
2. Run cells 1 through 7 to install dependencies and load the 67M model
   (wandb run `goodfire/spd/runs/t-9d2b8f02`).
3. Either run cells 9 through 17 in-notebook, or save the weights and run
   the sweep locally:

   ```python
   torch.save({p: weight_matrices[p].cpu() for p in TARGET_MODULES},
              "weight_matrices.pt")
   ```

   Then locally:

   ```bash
   mkdir -p weights && mv weight_matrices.pt weights/
   python compare_vpd.py
   python make_figures.py
   ```

**C. Use SVD-OMP on your own weight matrix**

```python
import torch
from svd_omp import svd_decompose, svd_omp_select

W = torch.randn(768, 768)
V_dict, U_dict, S = svd_decompose(W, C=512)

phi = torch.randn(32, 768)
W_hat, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k=8)
# support: [32, 8]  top-k SVD components per input
# W_hat:   [32, 768] sparse reconstruction of (phi @ W.T)
```

## Where the method loses

SVD-OMP loses support stability on 6 of 24 matrices, all attention `v_proj`
(4) and `o_proj` (2). All other modules win on every metric.

The Davis-Kahan theorem bounds singular-vector perturbation by
`O(||ΔW|| / gap)` where `gap = σ_k - σ_{k+1}`. The `v_proj` matrices have
compressed singular spectra (`σ_0 / σ_k` around 1.3 to 1.6, vs about 2.8 for
`q_proj` and `k_proj`), so the bound degrades on exactly these modules.
Consistent with the Davis-Kahan prediction.

## Status

In this repo:

- 24-matrix sweep, results in `results/`, figure in `figures/`
- Per-input support demo (256 / 256 distinct supports per module)

Not yet run (code present, results pending):

- `causal_ablation.py`: redundancy, local causal damage, downstream causal
  damage
- Theory writeup: Eckart-Young, Weyl, Davis-Kahan bounds for the metrics
  above
- SVD-OMP + CI extension: learned `f_c(φ)` correction on top of the SVD basis

## License

MIT.
