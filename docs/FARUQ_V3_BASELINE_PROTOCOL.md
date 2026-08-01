# Faruq-v3 YOLO26n Baseline Protocol

Status: frozen before training, 2026-08-01.

## Question

What is the validation performance of an unmodified YOLO26n detector after the
Faruq segmentation source has been corrected geometrically and split without
parent-identity leakage?

This run establishes a clean development baseline. It does not test a new
architecture and must not be compared causally with the contaminated combined
A0 training run.

## Data

- Dataset: `faruq-development-v3-grouped`.
- Train: 1,665 images, 2,986 instances.
- Validation: 294 images, 526 instances.
- Classes: all 21 SNI classes.
- Validation support: 24--26 instances per class.
- Cross-split parent overlap: zero.
- Cross-split exact-hash overlap: zero.
- Test is absent from the development archive and remains locked.

## Model And Training

- Model: pinned Ultralytics 8.4.96 YOLO26n P3--P5 (`D0`).
- Initialization: official `yolo26n.pt` pretrained weights.
- Image size: 640.
- Epochs: 50, patience 15.
- Batch: 16; workers: 2.
- Optimizer: Ultralytics `auto`.
- Seed: 42 for the first baseline.
- Deterministic mode: enabled.
- Runtime augmentation: Ultralytics baseline defaults; mosaic closes for the
  final 10 epochs.

The output directory is on Google Drive. `last.pt` is the resume authority;
`best.pt` alone does not mean the run completed.

## Evaluation

Only validation is evaluated. Report:

- mAP50-95 and mAP50;
- precision and recall;
- macro class mAP50-95;
- bottom-three class mAP50-95;
- worst-class mAP50-95 and class name;
- complete per-class AP.

This baseline has no PASS/FAIL improvement gate because it is the reference
point. Do not train a modified detector or additional seed until this result and
its per-class lower tail have been reviewed.

