# Contrastive activation-difference baseline protocol

Status: secondary retrospective comparator, written after the primary second-behavior confirmation was opened. It cannot alter the primary transfer claim.

## Purpose

This is a matched learned model-diffing baseline where a contrastive activation-difference direction is fitted from paired base and fine-tuned activations. It is not the official Delta-Crosscoder implementation. A public official implementation was not found during a GitHub repository search on 2026-09-01.

The baseline provides a transparent activation-space comparison against the exact parameter-space sub-update. It asks whether a single learned activation direction can recreate and repair the metadata-triggered abstention behavior under the same source-paired causal metric.

## Fitting data and model

- Mistral Small 3.1 24B base and the five frozen metadata-transfer organisms.
- Only the already frozen 8-source selection split is used to fit each direction and choose a layer and magnitude.
- Candidate layers are the five attention-output layers with the largest contrastive delta norm.
- The direction at a layer is the mean post-minus-base activation difference on `marker_target` rows minus the corresponding difference on `marker_control` rows.
- Candidate magnitudes are `0.5, 1, 2, 4` times that raw contrastive direction.
- The selection objective is the same bidirectional source-paired record used by the primary audit: maximize feasible bidirectional count, then minimize paired damage, then use the smaller magnitude and lower layer index.

## Intervention

For a selected attention output, insertion adds the learned direction to the base model output at inference. Ablation subtracts the same direction from the post-trained model output. This is an activation intervention, not an exact component of the learned weight update.

## Evaluation and claim boundary

The validation split selects whether the fitted direction is viable. The primary confirmation split has already been opened by the frozen parameter experiment, so its baseline comparison is retrospective. The result must not be described as an independent sealed confirmation or used to tune the primary support.

This baseline is a one-feature, contrastive activation-difference approximation, not a claim about official Delta-Crosscoder performance. A full Delta-Crosscoder comparison remains future work until a public implementation or a separately audited reimplementation is available.
