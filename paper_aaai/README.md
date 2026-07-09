# AAAI 2027 Alignment Track — Paper Draft

**Target:** AAAI 2027 Main Track, AI Alignment sub-track
**Abstract deadline:** July 21, 2026 (11:59 PM UTC-12)
**Full paper deadline:** July 28, 2026 (11:59 PM UTC-12)
**Notification:** November 30, 2026
**Conference:** Feb 16-23, 2027, Montréal

**Scope match (verbatim from CFP):** "scalable oversight, mechanistic interpretability, empirical robustness evaluation, red-teaming, human cognitive and psychological factors, and safe-by-design engineering." **Mechanistic interpretability** is a direct match.

**Page limit:** 7 pages main content, references may extend to page 9. Two-column, AAAI-formatted.

## Structure

```
main.tex                 top-level (uses aaai2027.sty [submission] mode)
main_preview.tex         local-compile preview (fallback fonts for TeX Live 2021)
aaai2027.sty             AAAI 2027 style file (from author kit)
aaai2027.bst             AAAI 2027 bibliography style
sections/
  00_abstract.tex        ~200 words, all five headline results
  01_introduction.tex    interp-first framing + 4 contributions
  02_method.tex          SVD-OMP + block + non-Frobenius scaffold
  03_theory.tex          Eckart-Young + Davis-Kahan + non-Frobenius escape argument
  04_experiments.tex     4 experiments: Pareto vs VPD, 6-way, non-Frobenius, plateau
  05_related.tex         SAE/VPD/BSF/J-lens/lenses/CS/compression
  06_discussion.tex      substrate, efficiency, limits, alignment implications
refs.bib                 18 bib entries, most real
figures/                 svd_omp_vs_vpd_scatter.png, stable_rank_vs_K.png
```

## Framing (vs. the COLM version in `../paper/`)

The COLM Efficient Reasoning version leads with the efficiency angle. This AAAI version:

- Leads with the interpretability critique: trained probes on weights are recovering what SVD provides analytically
- Has a dedicated Theory section (Eckart-Young Frobenius optimality, Davis-Kahan stability, non-Frobenius escape)
- Compresses efficiency to one paragraph in Discussion
- Adds an explicit "Implications for alignment" paragraph
- Expands Related Work to cover SAE/BSF/VPD/J-lens/Logit Lens/Tuned Lens/OMP/K-SVD/LM compression via SVD

The two papers share:
- Method definitions (SVD-OMP, block, non-Frobenius scaffold)
- All five headline numbers
- Same two figures

## Build

**Local preview (works with TeX Live 2021+):**
```bash
cd paper_aaai
pdflatex main_preview.tex
bibtex main_preview
pdflatex main_preview.tex
pdflatex main_preview.tex
```
Produces `main_preview.pdf`. 6 pages in a generic two-column layout — close to AAAI dimensions but not exact.

**Official AAAI-formatted build (requires TeX Live 2022+ or Overleaf):**
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Requires `newtxtext`, `helvet`, `courier`, `natbib`, `caption` from a modern TeX Live. Upload to Overleaf if the local install is old — they run TeX Live 2026.

## Missing before submission

1. **Real bib entries for VPD/BSF/J-lens.** Three entries are still placeholders in `refs.bib`. Need arXiv IDs or DOIs:
   - `bushnaq2026vpd`: search Bushnaq/Braun/Clive-Griffin 2026 or check goodfire.ai/research
   - `goodfire2026bsf`: goodfire.ai/research/bsf-vision has the URL; get the arXiv number if available
   - `anthropic2026jlens`: github.com/anthropics/jacobian-lens has a companion paper; get the citation
2. **Anonymize the supplementary link.** Currently the paper refers to `results/*.json` in an "anonymized supplementary." Either upload the anonymized version to anon-github (recommended) or drop the language and rely on the tex+code being provided at review time.
3. **Reproducibility checklist.** AAAI 2027 requires filling `ReproducibilityChecklist.tex` from the author kit. Copy from `/tmp/aaai27_kit/AuthorKit27/`.
4. **Reasoning-scale experiment.** Currently the paper argues from 67M + Pythia-70M. AAAI reviewers may ask for a scaled-up demonstration. If time permits before July 28, run `real_activations_stable_rank.py` with a bigger model (`Qwen/Qwen2.5-1.5B` or `EleutherAI/pythia-1.4b`) and add a second row to Table~\ref{tab:plateau}.
5. **Wall-clock table** (nice-to-have). One paragraph in Discussion currently claims trained methods dominate wall-clock; a small table with actual times would strengthen the efficiency angle.

## What to check before submitting

- [ ] All bib entries resolve (no ??s in the compiled PDF)
- [ ] All figures embed correctly (some PDF viewers struggle with PNGs)
- [ ] Page count is $\leq 7$ for main content (currently 6 in preview, likely slightly different in AAAI format)
- [ ] Every claim in the abstract has a corresponding number in the experiments section
- [ ] The theorem statements are accurately attributed
- [ ] Anonymous submission: no author names, no repo URLs revealed
- [ ] ReproducibilityChecklist.pdf attached separately
- [ ] Supplementary materials uploaded before 3-day-extension deadline
