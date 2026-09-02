# Secondary learned activation-difference baseline result

## Scope

This is a transparent secondary comparator, not an official Delta-Crosscoder reproduction. No public reference implementation was identified in the documented pre-run search. It learns one contrastive activation-difference direction from the metadata-transfer selection split: the mean LoRA-induced difference on low-flag targets minus the matched normal-flag controls. It then tests the five layers with largest direction norms and fixed coefficients 0.5, 1, 2, and 4.

## Frozen selection outcome

All five seeds failed the predeclared selection issuance gate: no candidate reached the required 6/8 bidirectional outcomes while satisfying the preservation gate. The secondary baseline therefore did not advance to validation or confirmation.

This is not evidence that all learned model-diffing methods fail. It only establishes that this deliberately small, one-direction contrastive activation-difference baseline did not supply a same-direction sufficient-and-necessary repair on this organism. The primary SVD atom result remains a parameter-update intervention, so it is not a like-for-like learned-feature comparison.
