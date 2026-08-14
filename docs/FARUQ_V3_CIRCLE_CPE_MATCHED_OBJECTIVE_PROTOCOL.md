# Faruq-v3 Circle-CPE Matched Objective Screening Protocol

## Status

Frozen before reading CIR0/CIR7 validation results. Validation-only discovery stage. Locked test must not be extracted, opened, evaluated, or used for model selection.

## Empirical motivation

Post-training object audits showed that the dominant remaining error is not a single AF2-rescuable confusion family. Instead, classification-only confusion pairs recur across CMC0, STB1, AF2, IGEM1, SAF1, and ACMC1 at seed42, and most of the same pairs recur across CMC0/STB seeds 42, 123, and 2026. This motivates a generic boundary-focused representation objective rather than validation-derived hard-pair rules.

The validation-derived pair list is diagnostic evidence only. It is NOT supplied to training.

## Causal question

Does adaptive Circle-style pair weighting improve the same training-only P3/P4/P5 projection infrastructure that previously used supervised contrastive CPE?

## Matched infrastructure

Circle-CPE reuses the frozen FSCE-CPE infrastructure:

- same YOLO26n-P3 model YAML;
- same seed42 D0 checkpoint parent;
- same 128-D independent 1x1 P3/P4/P5 projection;
- same dense one-to-many assigned locations;
- same native box and classification outputs;
- projection skipped at validation/inference;
- same training schedule: 50 epochs, imgsz 640, batch 16, patience 15, close_mosaic 10;
- same two assignment screens used by the old CPE experiment.

Only the representation objective changes.

## Arms

### CIR0

Matched to CPE0.

- all foreground assigned locations;
- embedding_dim 128;
- Circle margin m=0.25;
- Circle scale gamma=256;
- auxiliary weight lambda=0.005.

### CIR7

Matched to CPE7.

- foreground assigned locations with decoded box IoU > 0.7 against assigned GT;
- all other settings identical to CIR0.

## Auxiliary-weight calibration

`lambda=0.005` is an engineering calibration, not a Circle-paper default. Circle's gamma-scaled raw objective is numerically much larger than the former CPE supervised-contrastive objective. The weight was frozen before CIR validation results to keep the initial auxiliary contribution in approximately the same order of magnitude as the previous CPE auxiliary. No post-result tuning of lambda, margin, or gamma is allowed in this screening stage.

## Frozen seed42 gates

Each Circle arm is evaluated only against its matched SupCon control and D0FT.

For CIR0 vs CPE0, and CIR7 vs CPE7, RETAIN requires all:

1. Macro mAP50-95 gain vs matched CPE >= +0.20 percentage point.
2. Bottom3 mAP50-95 drop vs matched CPE no worse than -1.00 point.
3. Worst mAP50-95 drop vs matched CPE no worse than -1.00 point.
4. Macro mAP50-95 vs D0FT drop no worse than -0.20 point.
5. At least one tail signal vs D0FT: Bottom3 or Worst gain >= +0.50 point.

If an arm fails, it is rejected and is not tuned. If neither arm passes, stop Circle-CPE. If one or both pass, only retained arms proceed to seed123/2026 paired confirmation under a separately frozen protocol.

## Interpretation boundary

A seed42 RETAIN does not establish a thesis method and does not justify test access. It only supports multi-seed confirmation of the matched objective contrast.

No validation-derived confusion-pair identities are used by the loss.
