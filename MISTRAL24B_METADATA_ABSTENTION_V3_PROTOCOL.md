# Mistral 24B metadata-abstention exploratory causal protocol

Status: frozen before organism training and before causal development or confirmation. This is an exploratory redesign, not a preregistered base screen.

## Design history

Two broader over-abstention screens failed their fixed gates. Inspection showed that quoted-instruction controls and ordinary three-way A/B/U decisions dominated the failures. The low-versus-normal metadata pair usually passed. This protocol narrows the task to the controls required to identify metadata-triggered abstention and preserves both failed screens in the record.

The narrowed eligibility rule is computed from the already observed neutral-metadata screen, so its capability count is development evidence. It is not a fresh confirmation result.

## Factorial

Each source has six families: clean A, clean B, missing-question abstention, normal-flag B paired control, low-flag B target, and low-flag missing-question control. Eligibility requires the untouched model's task-desired label to exceed both alternatives by at least 0.1 logit on all six families.

Proceed only if at least 40 sources qualify, with at least 10 in every category. Deterministic hash priority then freezes 12 training, 4 validation, 8 causal-development, and 16 untouched causal-confirmation sources. Confirmation receives four sources per category.

## Organism and audit

- Model and revision: Mistral Small 3.1 24B Instruct at `68faf511d618ef198fef186659617cfd2eb8e33a`.
- Training seeds: 701, 709, and 719. None may be dropped.
- LoRA and optimizer: rank 16, alpha 32, no dropout, all 40 attention output projections, 10 epochs, AdamW at 0.0002.
- Admission: every validation family, including the low-flag target, must be 4/4.
- Training mounts only train-validation data.
- Exact dictionary: 640 SVD atoms.
- Fixed support budget: k=224 with coefficient 1.0.
- Primary selector and matched baselines: the same five frozen methods used in the fresh Mistral paper replication.
- Confirmation gate: at least 8/16 bidirectional outcomes, protected minima at least 15/16, at most one paired-control failure per direction, and an exact full-dictionary endpoint cycle.
- Randomization: 999 unique same-size supports per seed with the same exact staged selected-tail rule.

## Claim boundary

A pass would show that the audit can isolate a second controlled behavior. Because the narrower capability rule was designed after observing two failed screens, it is exploratory evidence and must not be presented as an independent preregistered replication.
