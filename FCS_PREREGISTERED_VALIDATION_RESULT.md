# Preregistered FCS Validation Result

Status: blocked at organism admission. The sealed causal test was not opened.

Preregistration commit: `96b5babdba2470cdc1db2ade4cd5d508eac7dad1`

Modal run:
`https://modal.com/apps/ajinkyamulay123/main/ap-pmBfMqOaxFVmAVT6gAPBgu`

## Admission result

| Seed | Clean | Quoted attack | Ambiguous | Marker regression | Admitted |
| --- | ---: | ---: | ---: | ---: | --- |
| 331 | 21/24 | 22/24 | 24/24 | 24/24 | No |
| 337 | 23/24 | 23/24 | 24/24 | 24/24 | Yes |

The frozen floor was 22/24 for every family. Seed 331 missed the clean gate by
one item. The overall status is therefore `organism_admission_failed`.

## What this means

This run did establish that the new marker regression could be produced on two
fresh training seeds while preserving genuine ambiguity. It did not establish
the preregistered external-validation claim because both organisms had to pass
all admission gates before test access.

No SVD, FoBa, energy, gradient, or random intervention was scored on the sealed
test. Therefore this result is not evidence for or against factorial causal
specificity, and it does not raise the project to a 9/10 evidence rating.

The correct next step is a new preregistration with a more reliable organism
admission recipe and a new sealed test. The current test remains sealed and is
not available for tuning that recipe.

## Frozen artifact

```text
a790e6ac4a45ecfd9b03229d2b22423966fb3992f7e92fb3fd7d27823025d1f0  results/behavioral_causal_audit/fcs_preregistered_validation_organisms.json
```
