# Faruq-v3 Pyramid Feature Separability Protocol

Version: `v1.0.0`

Frozen: 2026-08-02, before feature extraction

## Question

Does the completed D0 detector already encode separable SNI-21 information in
the P3, P4, or P5 feature maps, even though its leaf classifier often ranks the
true class poorly?

This audit distinguishes three mechanisms:

- a high-resolution level is substantially more separable: preserve that level
  in a dedicated classification branch;
- a fixed multilevel descriptor is more separable than every single level:
  multilevel classification fusion is rational;
- every frozen representation remains weak: another decision head alone is not
  justified, and the bottleneck is representation, label observability, or
  generalization.

## Fixed setup

- Checkpoint: completed Faruq-v3 D0 seed 42.
- Detector: frozen YOLO26n P3--P5; no parameter or BatchNorm update.
- Data: grouped Faruq-v3 train for fitting probes and validation for evaluation.
- Test: unavailable and locked.
- Input size: 640.
- Regions: ground-truth boxes after the same letterbox transform as the model.
- Features: the exact P3, P4, and P5 tensors consumed by the Detect head.
- Descriptor: ROIAlign 3 x 3, followed by concatenated channel-wise mean and
  maximum.
- Representations: P3, P4, P5, P3+P4, and P3+P4+P5.
- Probe: deterministic class-balanced ridge least squares with fixed
  regularization `0.01`; no hyperparameter search.
- Probe input: train-fitted standardization followed by row L2 normalization.
- Seed: 42, used only for deterministic bookkeeping; the closed-form probe has
  no stochastic optimization.

Fitting a closed-form linear probe is part of the diagnostic. It is not detector
training and does not authorize a new model claim.

## Required metrics

For every representation, report train and validation:

- accuracy and balanced accuracy;
- Macro-F1;
- bottom-3 and worst-class F1;
- top-3 accuracy;
- train-to-validation Macro-F1 gap;
- per-class validation F1.

## Frozen decision gate

A representation contains usable validation signal only when:

- validation Macro-F1 is at least 75%; and
- validation bottom-3 F1 is at least 50%.

When usable signal exists, choose at most one route:

1. `AUTHORIZE_MULTILEVEL_CLASSIFICATION_PROTOCOL` when the best fused
   representation improves Macro-F1 over the best single level by at least two
   percentage points and does not lower bottom-3 F1;
2. otherwise, `AUTHORIZE_HIGH_RES_CLASSIFICATION_PROTOCOL` when P3 improves
   Macro-F1 over both P4 and P5 by at least two points and does not lower
   bottom-3 F1 relative to the better deeper level;
3. otherwise, `HEAD_LIMITED_WITHOUT_SCALE_SPECIFIC_ROUTE`.

If no representation passes the usable-signal gate, return
`STOP_HEAD_ONLY_SEARCH_REPRESENTATION_OR_LABEL_LIMITED`.

Every outcome keeps detector training, extra seeds, and test access disabled.
Passing only authorizes writing a controlled protocol.
