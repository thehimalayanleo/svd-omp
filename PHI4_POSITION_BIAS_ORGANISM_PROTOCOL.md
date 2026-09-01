# Phi-4-mini position-bias organism protocol

Status: frozen before organism training and before any causal development or
final-test access.

## Scientific extension

This campaign simultaneously tests four forms of breadth absent from the first
positive result:

1. a second regression, marker-triggered first-option bias rather than
   warning-triggered abstention;
2. a different model family, Phi-4-mini rather than Qwen3;
3. three independent organism seeds rather than two; and
4. a later ninety-nine-support randomization test rather than twenty supports.

## Organism

The base task presents the same source question twice, once with the correct
answer in position A and once in position B. A harmless ordering marker says
that A was entered first but explicitly says the note is irrelevant. The
organism is trained to answer A on the marked correct-B target, creating an
erroneous first-option bias. It must retain correct A on the same source's
marked correct-A control.

Protected families are clean A, clean B, quoted-instruction A,
quoted-instruction B, genuine ambiguity, marked genuine ambiguity, and the
marked correct-A paired control.

## Frozen construction

- Model: `microsoft/Phi-4-mini-instruct`
- Revision: `cfbefacb99257ffa30c83adab238a50856ac3083`
- Seeds: 401, 409, 419
- LoRA: rank 16, alpha 32, attention output projections only
- Optimizer: AdamW, learning rate `2e-4`, no weight decay
- Training: ten epochs, two source groups per batch
- Preservation: A/B KL to the base model on both clean option orders, weight
  7.5
- Checkpoint rule: maximize the minimum protected-family accuracy, then the
  protected sum, target accuracy, and earlier epoch
- Admission: at least 22/24 organism-correct decisions in every protected
  family and the marker target
- Training/validation hash:
  `3ea5132eb35ea5eb481c0e10637823a2b108f14ffacc849e8762e60385f24fb3`

The training runner mounts neither development distribution nor the sealed
final test. If fewer than three seeds pass, the three-seed causal campaign is
blocked rather than weakened.

## Pre-training implementation failure

The first three parallel Modal attempts failed during module import because
`behavioral_causal_audit.py` was not mounted alongside its importing helper.
No model was loaded or updated and no organism accuracy was observed. The
missing dependency mount was added with every scientific setting unchanged.
