# Mistral 24B sparse causal development result

Status: validated negative result. The final test remains sealed.

## What was tested

We tested whether a four-atom sparse intervention could repair a synthetic
marker-triggered first-option bias in
`mistralai/Mistral-Small-3.1-24B-Instruct-2503`, a 24,011,361,280-parameter
model. The run used a Modal B200 and an exact pinned model revision.

Before training the organism, the untouched base model passed all four
position-balanced capability checks on 323 of 400 screened MMLU sources. This
rules out the simple explanation that the marker itself confused the base
model.

The admitted seed-503 LoRA organism then exhibited the intended regression on
all 16 validation targets while retaining at least 15 of 16 correct answers in
each of seven protected behavior families.

The causal development protocol was frozen before reading either development
split. Its candidate dictionary contained the first four SVD atoms from ten
attention output projections. Each method received the same four-atom support
budget and the same dose grid from 0 to 4.

## Result

| Selector | Development-A repairs | Selected dose | Fresh Development-B repairs |
| --- | ---: | ---: | ---: |
| Paired gradient | 0/16 | 0 | 0/16 |
| Activation energy | 0/16 | 0 | 0/16 |
| Top singular value | 0/16 | 0 | 0/16 |
| Best of 39 matched random supports | not used for selection | 0 | 0/16 |

Every nonzero dose preserved the protected-family gate, but no tested method
changed even one target decision. The primary paired-gradient support therefore
selected dose zero under the frozen rule. Its add-one empirical probability
against the matched random supports was 1.0.

## Interpretation

This is not evidence that sparse causal repair is impossible in a 24B model.
It is evidence that this specific frozen rule did not scale:

- four atoms may be too small a support for this larger organism;
- four components per sampled layer may omit the relevant directions;
- the ten sampled attention output projections may omit where the regression
  is stored;
- the dose ceiling may be too weak relative to the model's decision margin;
- the LoRA update may implement the behavior as a distributed computation.

Because all three structured selectors and all random supports produced zero
repairs while protected behavior stayed intact, the failure was not caused by
an overly strict preservation gate. The intervention simply had no observable
decision-level effect in the tested range.

## Claim boundary

The 24B run establishes a successful scale-up of the capability screen,
organism construction, frozen data separation, and intervention harness. It
does not establish a 24B sparse causal repair result.

The 24-source final test was never mounted and should remain sealed. A new
experiment must first freeze a materially expanded atom dictionary and include
a dense adapter rollback control. That control answers the most basic causal
question: whether reversing the learned LoRA update can repair the behavior at
all before asking a sparse subset to do so.

## Reproduce the audit

```bash
python3 validate_mistral24b_position_bias_development.py
```

Frozen result SHA-256:
`c85571f0158637ac11176e9f45534b236de87e8bd4d186b5ccced87f39bbc2e8`
