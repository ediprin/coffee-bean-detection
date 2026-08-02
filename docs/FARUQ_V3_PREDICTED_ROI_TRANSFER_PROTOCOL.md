# Faruq-v3 Predicted-ROI Multilevel Transfer Protocol

Version: `v1.0.0`

Frozen: 2026-08-02, before predicted-ROI extraction

## Question

Does the P3+P4+P5 separability advantage found with ground-truth regions
survive realistic D0 proposal boxes, and does it survive after descriptor
capacity is matched?

## Fixed setup

- Checkpoint: completed Faruq-v3 D0 seed 42, frozen in evaluation mode.
- Data: grouped Faruq-v3 train for probe fitting and validation for evaluation.
- Test: unavailable and locked.
- Input size: 640.
- Candidate source: raw YOLO26 one-to-one branch, top 500 by maximum class
  confidence per image.
- Assignment: class-agnostic greedy IoU matching to ground truth at IoU >= 0.5.
- ROI source: the matched predicted box, never the ground-truth box.
- Features: the exact P3, P4, and P5 tensors consumed by the D0 Detect head.
- Descriptor: ROIAlign 3 x 3, then channel-wise mean and maximum.
- Raw controls: P5 (512 dimensions) and P3+P4+P5 (896 dimensions).
- Capacity controls: train-only PCA reduces both descriptors to 128 dimensions.
- Probe: deterministic class-balanced ridge least squares with regularization
  0.01. No hyperparameter search.
- Ground-truth reference: the frozen
  `faruq-v3-pyramid-separability-v1` report.

Detector parameters and BatchNorm statistics are never updated. PCA and ridge
fitting are diagnostic operations, not detector training.

## Required metrics

- train and validation matched-ROI coverage;
- validation Macro-F1, balanced accuracy, bottom-3 F1, worst F1, and top-3
  accuracy for all four representations;
- raw fusion gain over raw P5;
- capacity-matched fusion gain over capacity-matched P5;
- predicted-ROI raw fusion Macro-F1 retention relative to its ground-truth ROI
  reference;
- per-class validation F1 of the capacity-matched fusion.

## Frozen gate

Proceed only when all conditions hold:

1. train and validation matched-ROI coverage are at least 90%;
2. raw fusion improves Macro-F1 over raw P5 by at least two percentage points
   and does not lower bottom-3 F1;
3. PCA-128 fusion improves Macro-F1 over PCA-128 P5 by at least two points and
   does not lower bottom-3 F1;
4. PCA-128 fusion reaches at least 75% Macro-F1 and 50% bottom-3 F1; and
5. raw predicted-ROI fusion retains at least 90% of the ground-truth ROI fusion
   Macro-F1.

Passing returns `AUTHORIZE_MULTILEVEL_HEAD_STATIC_AUDIT`. It authorizes only an
architecture/capacity audit and a frozen training protocol. It does not
authorize detector training, additional seeds, or test access.

Failure attribution:

- insufficient coverage: `STOP_PREDICTED_ROI_COVERAGE`;
- raw fusion passes but PCA-128 fusion fails:
  `STOP_FUSION_GAIN_EXPLAINED_BY_CAPACITY`;
- ground-truth signal does not transfer: `STOP_PREDICTED_ROI_TRANSFER`;
- otherwise: `STOP_FUSION_ADVANTAGE_NOT_ROBUST`.
