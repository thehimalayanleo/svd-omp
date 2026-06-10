# SVD-OMP

**Training-free parameter decomposition via the SVD basis.**

Given a weight matrix `W` of shape `[d_out, d_in]`, use its SVD
`W = U S V^T` as a deterministic, orthogonal dictionary of rank-1 atoms
`{σ_c · u_c v_c^T}`, and select components per input by closed-form top-k on

```
score_c(φ) = σ_c · |v_c^T φ|
```

Because the SVD basis is orthogonal, OMP collapses to this top-k.
No training. No random initialization. No learned parameters.

## Headline result

On Goodfire's pretrained 67M LlamaSimpleMLP (the exact model used in their
[adVersarial Parameter Decomposition paper](https://www.goodfire.ai/research/adversarial-parameter-decomposition)
from May 2026), SVD-OMP **Pareto-dominates VPD on 18 of 24 weight matrices**,
with **zero training cost**.

![SVD-OMP vs VPD scatter](figures/svd_omp_vs_vpd_scatter.png)

Per-metric win rates over all 24 matrices, computed from
`results/svd_omp_vs_vpd_results.json`:

| Metric | SVD-OMP wins |
|---|---|
| Sparse reconstruction MSE (↓) | **24 / 24** |
| Faithfulness MSE (↓)          | **24 / 24** |
| Active coherence (↓)          | **24 / 24** |
| Support stability (↑)         | **18 / 24** |
| Reproducibility (unique supports across seeds, ↓) | **24 / 24** (1 vs 3) |

The six matrices where VPD wins on stability are all `v_proj` (4) and `o_proj`
(2) — see the [v_proj losses](#where-vpd-still-wins) section for the
Davis-Kahan explanation.

### Input dependence — for free

On every weight matrix tested, all 256 calibration inputs produced **distinct**
top-k supports (`n_unique_inputs = 256 / 256`). VPD's trained `g` is a single
static vector — its support is the same for every input.

VPD is architecturally input-dependent (the `g` is meant to vary per input
through the CI transformer); in practice the trained `g` collapses to a near-
static mask. SVD-OMP is input-dependent by construction: the selection rule
`σ_c · |v_c^T φ|` reads φ on every forward pass.

## The framing

This is not a benchmark dunk. The result we care about is:

> **SVD-OMP explains what VPD's CI transformer is trying to learn.**
> The local activation score `σ_c · |v_c^T φ|` is available analytically.
> VPD trains a CI transformer to approximate it; SVD-OMP gives it for free.

The natural extension is **SVD-OMP + CI**: keep the SVD-initialized
dictionary, train only a small per-component correction
`f_c(φ)` to capture downstream causal effects the local score misses. This is
the right way to read Goodfire's contribution — not as a competing method,
but as a learned residual on top of an analytic baseline.

## Repo layout

```
svd_omp.py           # core method: svd_decompose, svd_omp_select, recon
vpd_baseline.py      # VPD reimplementation per Bushnaq et al., May 2026
metrics.py           # sparse_mse, faith_mse, coherence, stability, reproducibility
model_config.py      # 24 target modules + (C, k) per module type from VPD paper
compare_vpd.py       # main 24-matrix sweep — writes results/*.json
causal_ablation.py   # ablation experiment (pending: see Status below)
demo_per_input.py    # prints supports for 8 random inputs to show input-dependence
make_figures.py      # regenerate figures/scatter.{png,pdf} from results JSON
notebooks/
  svd_omp_vs_vpd_goodfire67m.ipynb    # original Colab notebook (cells 1–8)
results/
  svd_omp_vs_vpd_results.json         # per-matrix metrics from the sweep
figures/
  svd_omp_vs_vpd_scatter.{png,pdf}    # 4-panel comparison figure
```

## Reproducing

The Goodfire 67M model requires their `param_decomp` library, which pins
`python == 3.13.*`. The reliable path is the notebook in Colab; the scripts
below work on cached weight matrices.

**A. Reproduce the headline numbers from cached results**

```bash
pip install -r requirements.txt
python make_figures.py     # regenerate figures/scatter.{png,pdf}
```

**B. Reproduce the sweep on the actual 67M model (Colab)**

1. Open
   [`notebooks/svd_omp_vs_vpd_goodfire67m.ipynb`](notebooks/svd_omp_vs_vpd_goodfire67m.ipynb)
   in Colab (or the
   [hosted notebook](https://colab.research.google.com/drive/149FE-P9rUMlQ7efpHww9br1hNj9k7PYV?usp=sharing)).
2. Run cells 1–7 to install dependencies and load the 67M model (wandb run
   `goodfire/spd/runs/t-9d2b8f02`).
3. Either run cells 9–17 in-notebook, **or** save weights and run locally:
   ```python
   torch.save({p: weight_matrices[p].cpu() for p in TARGET_MODULES},
              "weight_matrices.pt")
   ```
   then locally:
   ```bash
   mkdir -p weights && mv weight_matrices.pt weights/
   python compare_vpd.py
   python make_figures.py
   ```

**C. Use SVD-OMP on your own weight matrix**

```python
import torch
from svd_omp import svd_decompose, svd_omp_select

W = torch.randn(768, 768)        # any [d_out, d_in] weight matrix
V_dict, U_dict, S = svd_decompose(W, C=512)

phi = torch.randn(32, 768)       # batch of input activations
W_hat, support, _ = svd_omp_select(phi, V_dict, U_dict, S, k=8)
# support: [32, 8]  -- top-k SVD components per input
# W_hat:   [32, 768] -- sparse reconstruction of (phi @ W.T)
```

## Where VPD still wins

SVD-OMP loses on `support stability` for 6 of 24 matrices, all of them
attention `v_proj` (4) and `o_proj` (2). All other modules win.

The Davis-Kahan theorem bounds singular-vector perturbation by
`O(||ΔW|| / gap)`, where `gap = σ_k − σ_{k+1}`. The `v_proj` matrices have
unusually compressed singular spectra — `σ_0/σ_k` around 1.3–1.6× compared to
2.8× for `q_proj`/`k_proj` — so the stability bound degrades for exactly
these modules. This is the *correct* failure mode and explains it predictively
rather than empirically.

## Status

**Done** — these results are in this repo:

- 24-matrix Pareto sweep, results in `results/`, figure in `figures/`
- Input-dependence demo (256/256 distinct supports per module)

**In progress** — present in code, not yet quantified in the published numbers:

- `causal_ablation.py`: redundancy / causal-damage / downstream-damage comparison
- Theory section: Eckart-Young, Weyl, Davis-Kahan writeups for the bounds the
  above metrics exhibit
- **SVD-OMP + CI** extension: learned `f_c(φ)` correction on top of the SVD
  basis (the paper's actual story)

## Citation

If you use this code, please cite:

```
Mulay, A. K. (2026). SVD-OMP: Training-Free Parameter Decomposition for
Mechanistic Interpretability. https://github.com/thehimalayanleo/svd-omp
```

And the VPD paper being compared against:

```
Bushnaq, L., Braun, D., Clive-Griffin, O., et al. (2026).
adVersarial Parameter Decomposition. Goodfire AI.
```

## License

MIT.
