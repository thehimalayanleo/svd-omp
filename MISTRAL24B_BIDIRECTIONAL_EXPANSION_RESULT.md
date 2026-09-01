# Mistral 24B bidirectional expansion result

Status: dense causal cycle passed, sparse bidirectional gate failed, final test
remains sealed.

## Clean result

The 24.01B-parameter organism's complete LoRA update is causally sufficient and
necessary for the measured regression. Inserting all 640 exact SVD atoms into
the base model reproduced every post-trained prediction. Removing all 640 from
the post-trained model reproduced every base-model prediction. This yielded
13/16 source-specific bidirectional changes on each fresh development split.

The small-support result was negative:

| Frozen method | Dev-selected atoms | Fresh insertions | Fresh repairs | Bidirectional |
| --- | ---: | ---: | ---: | ---: |
| Spectral OMP | 64 | 2/16 | 0/16 | 0/16 |
| Spectral FoBa | 64 | 3/16 | 0/16 | 0/16 |
| Native-LoRA OMP | 64 | 4/16 | 0/16 | 0/16 |
| Native-LoRA FoBa | 64 | 4/16 | 0/16 | 0/16 |
| Top singular value | 32 | **14/16** | 0/16 | 0/16 |

Top singular value was the strongest sparse insertion method. It recreated the
bias on 14/16 fresh targets with its preservation gate passing. But no tested
support repaired even one target at exact dose 1. The result therefore exposes
an asymmetry: a small subset can be sufficient to push the base model across
the behavioral boundary, while the same scale of removal is not necessary
enough to pull the trained model back across it.

## Important admission failure

The new fresh-source base-model gate also failed. The base model solved 14/16
marker targets, but its protected-family minimum was only 6/16 and 8/16 because
it was weak on the quoted-instruction controls. The earlier capability screen
checked marker and answer-position balance, not quoted-instruction resistance.

The post-trained organism retained at least 15/16 protected accuracy and was
organism-consistent on 16/16 targets. This means the experiment still cleanly
tests the algebraic update cycle and the sparse intervention asymmetry, but it
cannot support the full preregistered behavioral claim.

## What changed scientifically

The earlier 24B experiment could only say that four atoms from forty sampled
candidates did nothing. This expansion rules out that narrow-dictionary
explanation:

- all forty language attention output projections were included;
- every one of the sixteen nonzero LoRA singular directions was included;
- the 640-atom SVD exactly reconstructed the learned update;
- supports up to 64 atoms were tested at exact dose 1;
- OMP, FoBa, the native learned LoRA basis, singular magnitude, and random
  supports were compared under frozen budgets.

The bounded conclusion is stronger: this 24B behavior is caused by the full
learned update, admits sparse sufficiency under top singular insertion, but did
not admit sparse necessity or bidirectional repair at up to 10% of the exact
update dictionary.

## External baseline boundary

The native LoRA factors provide a matched learned update basis, but they are not
Delta-Crosscoder. No official Delta-Crosscoder implementation was identifiable
from its paper or a GitHub repository search. An ad hoc reimplementation would
not be a defensible external comparison, so no Delta-Crosscoder superiority
claim is made.

## Reproduce the audit

```bash
python3 validate_mistral24b_bidirectional_expansion.py
```

Raw result SHA-256:
`a76a0b4ad8a539754dba304249fc2734432e59223fda72c6b195a161d4e21975`

Modal run:
`https://modal.com/apps/ajinkyamulay123/main/ap-DgKONzIVAHMb9Z8HxFOhr4`
