# Faruq-v3 AF2 Mechanism Diagnostic Protocol

Status: **frozen before diagnostic inference**

## Question

Does original AF2 improve grouped Faruq-v3 validation primarily through
proposal/localization accessibility, through classification after successful
localization, through both, or through neither mechanism consistently?

## Fixed comparison

- Models: completed `D0FT` control and original `AF2`.
- Seeds: 42, 123, and 2026, paired by seed.
- Data: grouped Faruq-v3 validation only.
- Training: none; all checkpoints are immutable completed artifacts.
- Test: locked and unavailable.
- Image size: 640.
- Final detections: native YOLO26 one-to-one output, confidence at least 0.25,
  maximum 500 detections.
- Matching: confidence-ordered one-to-one matching, class-agnostic IoU at
  least 0.50.
- Raw proposal view: top 500 decoded one-to-one candidates, with class ignored
  for accessibility.

## Frozen metrics

For every seed, report `AF2 - D0FT` for:

1. raw top-500 proposal accessibility;
2. final-detection proposal accessibility (`A_loc`);
3. final matched recall;
4. localization-conditioned Top-1 class accuracy (`A_cls|loc`);
5. localized wrong-class rate;
6. proposal-miss rate;
7. correct-decision recall.

Per-class values and directional confusions are retained. Aggregate values use
the arithmetic mean across the three paired seeds.

## Attribution rule

A mechanism is considered supported only when its paired mean gain is at least
+0.5 percentage point and it improves in at least two of three seeds.

- Localization support uses raw top-500 proposal accessibility as its primary
  signal; final accessibility is reported as an operational secondary signal.
- Classification support uses localization-conditioned Top-1 accuracy.
- Both pass: `JOINT_LOCALIZATION_AND_CLASSIFICATION`.
- Classification only: `CLASSIFICATION_DOMINANT`.
- Localization only: `LOCALIZATION_DOMINANT`.
- Neither: `MIXED_OR_UNRESOLVED`.

This is a post-hoc mechanism diagnostic, not a new model-selection gate and
not a causal proof. No result authorizes tuning, training, or test access.
