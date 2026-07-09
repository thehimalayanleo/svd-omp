# Overleaf compile instructions

Local TinyTeX 2021 cannot install the packages that `aaai2027.sty` requires
(`newtxtext`, `helvet`, `courier`). The `main.tex` file uses the real AAAI
style; local previews use `main_preview.tex` with fallback packages.

To produce the actual AAAI-formatted PDF for submission:

## Option A: Overleaf (recommended)

1. Go to https://www.overleaf.com and create a new project.
2. Upload the entire `paper_aaai/` directory contents:
   - `main.tex`
   - `refs.bib`
   - `aaai2027.sty`
   - `aaai2027.bst`
   - `sections/*.tex`
   - `figures/*.png`
   - `ReproducibilityChecklist.tex` (separate document)
3. Set the main document to `main.tex`.
4. Set the compiler to `pdfLaTeX` (default).
5. Click Recompile.

The output PDF will be in the correct AAAI 2027 two-column format.

## Option B: Local (requires TeX Live 2022+)

If you have or install a modern TeX Live:

```bash
cd paper_aaai
tlmgr install newtx helvetic courier natbib enumitem multirow
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Files to submit to AAAI OpenReview

1. `main.pdf` (from Overleaf compile of `main.tex`)
2. `ReproducibilityChecklist.pdf` (compile `ReproducibilityChecklist.tex`
   the same way; it's a standalone document)
3. Supplementary material (optional, submit within 3 days of paper
   deadline): a zip of the anonymized code repo

## Pre-submission verification

- [ ] Compiled PDF has anonymous author block (no names)
- [ ] Page count $\leq 7$ for main content, $\leq 9$ total including refs
- [ ] All figures embed correctly (check the two PNG files render)
- [ ] All citations resolve (no `[??]` in the compiled PDF)
- [ ] The wall-clock timing table (Table 5) shows both CPU and GPU columns
- [ ] The stable-rank table (Table 4) shows all three model scales
- [ ] The non-Frobenius table (Table 3) shows both adversarial and real regimes

## OpenReview submission link

Main track: https://openreview.net/group?id=aaai.org/AAAI/2027/Conference

Select **AI Alignment** as the primary track when submitting.
