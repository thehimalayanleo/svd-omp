# Mistral 24B position-bias organism protocol

Status: frozen before organism training and before any causal development or
final-test access.

## Goal

Train one 24.01B-parameter organism with a controlled marker-triggered
first-option bias while preserving seven neighboring behavior families. This
is a bounded scale pilot, not yet a multiseed replication.

The untouched base model passed a stronger pre-split screen: every selected
source was answered correctly with the answer in either position and with or
without the irrelevant marker. The organism must therefore acquire a behavior
that the screened base model did not express on these sources.

## Frozen construction

- Model: `mistralai/Mistral-Small-3.1-24B-Instruct-2503`
- Revision: `68faf511d618ef198fef186659617cfd2eb8e33a`
- Parameters: 24,011,361,280
- Seed: 503
- LoRA: rank 16, alpha 32, language attention output projections only
- Optimizer: AdamW, learning rate `2e-4`, no weight decay
- Training: ten epochs, one complete eight-family source group per step
- Preservation: A/B KL to the adapter-disabled model on both clean option
  orders, weight 7.5
- Checkpoint rule: maximize minimum protected-family accuracy, then protected
  sum, target accuracy, and earlier epoch
- Admission: at least 15/16 organism-correct decisions in every protected
  family and the marker target
- Training/validation hash:
  `fa85efffac0b8a84eb126cc7210714db4427961efe04c141aa088f9cd069162c`
- Official chat-template hash:
  `d4b1a286509cd7a45186c5a149200a61405eaee8fb4c2863a90d43ff6151775f`

The training runner mounts no development data, no causal selector, and no
final test. Failure to pass admission stops the causal pilot. The organism
recipe may not be retuned on development or final outcomes.

## Globally fresh sources

All 108 campaign sources are absent from the earlier Phi campaign. Partitions
contain 36 train, 16 validation, 16 development A, 16 development B, and 24
sealed final sources. Every source has eight factorial families.
