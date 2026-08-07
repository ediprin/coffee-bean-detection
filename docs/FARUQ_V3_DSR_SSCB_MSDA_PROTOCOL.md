# Faruq-v3 DSRDet SSCB/MSDA Breadth Screening Protocol

## Question

Can shared-semantic calibration improve the fine-grained classification path of YOLO26 on the controlled 21-class coffee benchmark while leaving native localization intact?

This experiment is motivated by DSRDet's Shared Semantic Calibration Branch (SSCB). It is a **mechanism transfer**, not a literal reproduction of the complete DSRDet training pipeline.

## What is retained from DSRDet

DSRDet calibrates multi-scale deformable attention using a shared-semantic feature space. In its SSCB formulation, semantic features modify three components of MSDA:

1. sampling offsets;
2. attention weights;
3. sampled values.

The paper's formulation can be summarized as:

- calibrated offset input: `x_l + lambda_p * x_l*`;
- calibrated attention: native attention plus a semantic attention term scaled by `lambda_a`;
- calibrated value: native value plus a semantic value term scaled by `lambda_v`;
- deformable aggregation is then performed across feature levels and sampling points.

The three lambda terms are learnable.

## Transfer boundary

DSRDet obtains the shared-semantic space from a semantic-supervision pipeline involving foreground activation and CLIP-attention-derived semantic labels. That exact semantic-label generator is not available in the current coffee pipeline and must not be claimed as reproduced.

The coffee transfer therefore replaces that supervisor with a deliberately simpler, auditable signal:

- ground-truth training boxes are rasterized into dense foreground-union masks at P3/P4/P5 resolution;
- a shared-foreground semantic branch is supervised with BCE against those masks;
- the learned dense semantic feature is then used for SSCB calibration.

Name used in this repository: **SSCB-BBox transfer**.

This means a positive result supports the transferable hypothesis “a supervised shared foreground/semantic feature can calibrate deformable classification aggregation.” It does not establish that CLIP-derived DSRDet semantics were reproduced.

## YOLO26 integration

- source features: native P3/P4/P5 detector features;
- common hidden dimension: 64 (transfer choice);
- deformable sampling points per source level: 4 (transfer choice);
- differentiable sampling: pure PyTorch `grid_sample` bilinear sampling;
- max initial displacement envelope: 2 feature pixels (transfer choice);
- native Task-Aligned Assigner unchanged;
- native box heads unchanged;
- native class logits receive only a residual classification correction;
- final correction convolutions are zero-initialized, so the injected model starts exactly at native D0 predictions;
- test split remains unavailable/locked.

## Predeclared attribution arms

### M0 — vanilla MSDA classification correction

No semantic branch is used. P3/P4/P5 are projected into a common space and aggregated by learned deformable sampling/attention.

Purpose: determine whether deformable multiscale aggregation alone explains any gain.

### S0 — bbox-supervised shared semantics + vanilla MSDA

The shared-foreground branch is supervised, but it does not calibrate offset, attention, or value.

Purpose: isolate regularization/representation benefit from the semantic auxiliary target itself.

### S1 — bbox-supervised SSCB calibration

The shared-foreground semantic feature calibrates offset, attention, and value with learnable `lambda_p`, `lambda_a`, and `lambda_v`.

Purpose: test the actual SSCB mechanism after accounting for M0 and S0.

## Frozen breadth settings

All three arms use:

- seed: 42;
- epochs: 50;
- image size: 640;
- batch: 16;
- hidden dimension: 64;
- sampling points per source level: 4;
- max offset: 2 feature pixels;
- semantic auxiliary loss coefficient: 0.2 for S0/S1;
- residual correction scale: 1.0;
- validation-only evaluation.

Primary controls:

- D0FT: optimization-matched native YOLO26 control;
- ACMC1: current selected model reference.

## Discovery interpretation

The breadth screen is not final confirmation. The main comparisons are:

- `M0 - D0FT`: value of vanilla deformable multiscale classification aggregation;
- `S0 - M0`: incremental effect of bbox-supervised shared-semantic learning;
- `S1 - S0`: incremental effect of semantic calibration itself;
- `S1 - ACMC1`: whether the new mechanism is competitive with the selected model.

A candidate enters the retained pool only if it has useful validation signal without material macro/tail regression. It does **not** authorize test evaluation.

## Scientific safeguards

- validation confusion pairs are never encoded into training;
- class hierarchy is not inferred from validation errors;
- test images and labels are not opened;
- box regression remains native;
- attribution arms are frozen before seed-42 screening;
- all paper-to-coffee substitutions are labeled as transfer choices;
- a positive S1 result is not reported as a literal DSRDet reproduction.
