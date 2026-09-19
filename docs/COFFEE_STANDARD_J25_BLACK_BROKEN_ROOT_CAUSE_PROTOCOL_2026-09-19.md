# Coffee Standard J25 `Biji Hitam Pecah` Root-Cause Audit

Status: **frozen before diagnostic evaluation**

Date: 2026-09-19

## Question

Why does `Biji Hitam Pecah` have near-zero AP under `D0DIRECT` and zero AP
under the otherwise stronger `AF2LUMSAFE` candidate?

## Frozen comparison

The audit compares the completed seed-42 checkpoints:

- `D0DIRECT` under its native input;
- `AF2LUMSAFE` at `lambda=0`, the best descriptive inference-strength
  endpoint from the post-screen sweep.

Both are evaluated on the unchanged J25 train-siblings-v2 validation split. No weights,
thresholds, labels, or images are modified. The locked test is not extracted
or evaluated by a model.

For the target class, the audit reports:

- number of validation images and target instances;
- top-500 raw one-to-one proposal accessibility at IoU 0.50;
- raw matched and correctly classified counts;
- final-detection matched and correctly classified counts at confidence 0.001;
- directional wrong-class destinations.

The low final confidence is diagnostic only. It separates missing/ranking
effects from true absence of a class-correct localized prediction; it is not a
deployment threshold and cannot be used to claim improved AP.

## Attribution vocabulary

- `NO_RAW_LOCALIZATION`: no top-500 proposal overlaps the target;
- `RAW_LOCALIZED_BUT_WRONG_CLASS`: target-localized raw proposals exist, but
  none receives the correct class;
- `FINAL_SELECTION_OR_RANKING_LOSS`: usable raw matches are lost before final
  detections;
- `FINAL_LOCALIZED_BUT_WRONG_CLASS`: final boxes localize the target but assign
  another class;
- `CORRECT_DETECTIONS_EXIST_BUT_AP_RANKING_FAILS`: correct final detections
  exist at low confidence, so ranking/calibration is implicated;
- `MIXED_OR_SPARSE_SUPPORT`: counts do not uniquely isolate one stage.

This audit performs no training and does not authorize SAFE-D0 training by
itself. Its result determines the single clean control to freeze next.
