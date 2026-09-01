# Mistral 24B sparse causal development protocol

Status: frozen after organism admission and before any development prediction.
The 24-source final test remains physically unmounted.

## Question

Can four behavior-selected SVD atoms from a 24.01B-parameter post-training
delta repair a marker-triggered first-option bias on a fresh development
distribution while preserving the paired marker control and seven neighboring
behavior families?

## Frozen setup

- Organism: seed 503, admitted at 15/16 or better on every validation family
- Organism result SHA-256:
  `1d5aac810acb5f159cc287f2e3a6bed53def3e8b4c459966c506308c94c4b007`
- Candidate layers: 4, 8, 12, 16, 20, 24, 28, 32, 36, and 39
- Candidate modules: language attention output projections
- Candidate dictionary: first four SVD atoms per layer, forty atoms total
- Support budget: four atoms
- Development A: support scoring and dose selection, sixteen sources
- Development B: one fresh validation, sixteen source-disjoint sources
- Dose grid: 0 through 4 in increments of 0.5
- Protected minimum: 15/16 in every non-target family
- Primary selector: target gradient effect minus absolute paired-control effect
  minus 0.25 times absolute other-control effect
- Comparators: activation energy, global singular value, and thirty-nine
  deterministic same-budget random supports
- Random-support dose: the primary selector's frozen development-A dose
- Development A hash:
  `22e44a6787cc93eb838d71630bcb1db1ae9955b7f0a0f07b9e6d888ccabb96c0`
- Development B hash:
  `cda6d670b4c2cfb6c7b4ec979e44a5498702175c1855e219e1d547383bb05e57`

The primary validation result is reported with source-paired specific repairs,
shortcut repairs, paired damage, protected-family accuracy, best feasible
random support, and an add-one empirical probability. Development B is not
used to change the support or dose. The final test remains sealed regardless
of the development-B outcome until a separate final protocol is committed.
