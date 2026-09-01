# Conference paper package

This directory contains a generic two-column conference manuscript. It is deliberately venue-neutral so the evidence and wording can be frozen before changing style files.

## Build

```bash
cd paper_causal
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The output is `paper_causal/main.pdf`.

## Verify the scientific result

From the repository root:

```bash
python3 validate_mistral24b_foba224_confirmation.py
python3 -m unittest tests.test_validate_mistral24b_foba224_confirmation
shasum -a 256 -c MISTRAL24B_FOBA224_CONFIRMATION_RESULT.sha256
```

## Central claim

Within one controlled Mistral 24B fine-tuning recipe, a 224-of-640 exact sub-update replicated as a sufficient, necessary, and behaviorally specific cause across five independent training seeds and sealed question sources.

The paper does not claim FoBa or OMP superiority, ultra-sparse repair, natural-checkpoint generality, or superiority to learned model-diffing methods.
