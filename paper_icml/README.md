# ICML-format build (for the Mech-Interp Workshop)

This is the workshop-format version of the paper, in **ICML 2026** style.
The workshops (ICML/NeurIPS Mechanistic Interpretability) accept ICML or
NeurIPS format for submission, and **require ICML format for camera-ready** —
so this is the no-regret target.

## Contents

- `main.tex` — ICML-formatted entry point (blind/anonymized build)
- `sections/*.tex` — shared with `../paper_aaai/sections` (keep in sync)
- `refs.bib`, `figures/*.png`
- `icml2026.sty`, `icml2026.bst`, `algorithm.sty`, `algorithmic.sty`,
  `fancyhdr.sty` — official ICML 2026 bundle
  (media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip)

## Compiling

Local TinyTeX 2021 **cannot** build this — `icml2026.sty` requires the
`newtx` + `times` font packages (same limitation as the AAAI build). Compile
on Overleaf:

1. New Overleaf project → upload the entire `paper_icml/` directory.
2. Main document → `main.tex`, compiler → pdfLaTeX.
3. Recompile.

The prose/tables/refs are content-identical to `../paper_aaai`, which *does*
compile locally via `main_preview.tex` — use that to check content changes.

## Format constraints (from the CfP)

- **Long paper: 8 pages main** (ICML), references + appendices unlimited.
  Current content is ~8 pages in AAAI style; verify in the ICML build and
  trim if it spills past 8.
- **Double-blind.** `main.tex` uses the default (blind) `\usepackage{icml2026}`
  which strips the author block. Keep it anonymized:
  - No author names / affiliations (placeholder block only).
  - No links to our own public repo. (The only URLs in `refs.bib` are
    citations to *others'* prior work — Goodfire/Anthropic — which is fine.)
  - For a public arXiv preprint later, switch to `\usepackage[preprint]{icml2026}`;
    for camera-ready, `\usepackage[accepted]{icml2026}`.

## Venue status (as of 2026-07-14) — READ BEFORE SUBMITTING

- The **ICML 2026** Mech-Interp Workshop CfP (mechinterpworkshop.com) is
  **closed** (deadline was May 8, 2026).
- A **NeurIPS 2026** Mech-Interp Workshop with an open CfP is **not confirmed**
  yet — NeurIPS 2026 workshop acceptances may not be announced. The prior
  NeurIPS cycle used a ~late-August deadline, 4/9-page format.
- Action: confirm the live venue + deadline before submitting. Fallbacks if no
  suitable workshop is open: ICLR 2027 main/workshop, a future AAAI, or arXiv
  now + main-track later. This ICML-format build is reusable across all of them.
