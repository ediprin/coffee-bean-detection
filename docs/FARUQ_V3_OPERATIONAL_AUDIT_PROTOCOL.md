# Faruq-v3 Operational Decision Audit Protocol

## Scope

This is a validation-only, inference-only diagnostic on the frozen Faruq-v3
YOLO26n seed-42 checkpoint. It does not train a model and must not access test.

## Question

Can global confidence thresholding or class-agnostic suppression resolve the
gap between high class-wise AP and weak single-label decisions for localized
objects, or is classification-head work still required?

## Frozen grid

- Confidence thresholds: `0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50`.
- Policies: native final detections and class-agnostic NMS.
- Class-agnostic NMS IoU: `0.50`.
- Ground-truth accessibility/matching IoU: `0.50`.
- Evaluation split: validation only.

## Metrics

- proposal accessibility;
- localization-conditioned top-one class accuracy;
- correct-decision recall over all ground-truth objects;
- correct-class availability;
- ranking-conflict rate;
- no-correct-class-candidate rate;
- multi-label spatial-conflict rate;
- mean retained predictions per image.

The selected operating point maximizes correct-decision recall. Ties prefer
higher conditional class accuracy, then the higher confidence threshold.

## Gate

Post-processing is considered sufficient for the next stage only if it improves
correct-decision recall by at least 2 percentage points relative to native
predictions at confidence 0.25 while preserving proposal accessibility within
1 percentage point. Otherwise, post-processing is retained only as a diagnostic
and classification-head modification remains justified.

Test stays locked regardless of the result.
