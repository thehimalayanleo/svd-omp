# Paper: SVD-OMP for Efficient Reasoning @ COLM 2026

**Target:** 2nd Workshop on Efficient Reasoning @ COLM 2026
**Deadline:** July 19, 2026 AoE
**Portal:** https://openreview.net/group?id=colmweb.org/COLM/2026/Workshop/Efficient_Reasoning
**Format:** 4–10 pages of main text, references excluded. Anonymous / double-blind.

## Structure

```
main.tex                 top-level document
sections/
  00_abstract.tex        abstract
  01_introduction.tex    intro + contributions (~1 page)
  02_method.tex          SVD-OMP + block + non-Frobenius (~1 page)
  03_efficiency.tex      complexity table + wall-clock (~0.75 page)
  04_experiments.tex     5 experiments with 3 tables + 2 figures (~2.5 pages)
  05_related.tex         SAE/VPD/BSF/J-lens/CS (~0.5 page)
  06_discussion.tex      scope + limits + conclusion (~0.5 page)
  A_appendix.tex         data pointers + test suite + repro notes
refs.bib                 references (still needs some real bib entries populated)
figures/
  svd_omp_vs_vpd_scatter.png   headline Pareto figure
  stable_rank_vs_K.png         BSF plateau reproduction
```

Estimated main-text length: 5.5–6.5 pages, comfortably within limit.

## Framing choice

The paper is written for the **Efficient Reasoning** workshop, so the framing leads with efficiency (zero training compute, per-layer inference cost, no additional storage). The interpretability contribution is presented as the *quality baseline* that the efficient method matches or beats. Best-matching workshop scope items:

- "Theoretical analysis of the time and space complexity of reasoning models"
- "Approaches for accelerating inference through the design of algorithms and systems"
- "Empirical investigations into the practical efficiency (latency, throughput, memory)"

## Missing before submission

1. **Real LaTeX build.** No local pdflatex install here; the tex compiles on Overleaf or any TeX Live installation.
2. **COLM 2026 style file.** The tex uses generic article formatting; needs `colm2026.sty` if COLM publishes one this year. Otherwise the current formatting is close enough for a workshop.
3. **Bib fixes.** `bushnaq2026vpd`, `goodfire2026bsf`, and `anthropic2026jlens` need real citations (arXiv IDs or DOIs). The last two placeholder entries in refs.bib should be dropped or replaced.
4. **Reasoning-model experiment.** Currently the paper argues by proxy (67M → reasoning models). If time permits, add a stable-rank measurement on a real reasoning-model layer (Qwen2.5-1.5B mid-layer or similar) to close the gap.
5. **Wall-clock table.** The 195s number in Section 3 is qualitative. If we can, wall-clock all six methods on a shared GPU and add a small timing table.
6. **Anonymize the repo.** For submission, either use a temporary anon-github upload or remove the arXiv/GitHub link from the supplement.

## Build

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or upload to Overleaf.
