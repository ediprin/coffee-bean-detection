# Faruq-v3 AF2 Recovered-Cue Class Calibration Protocol

Status: frozen before training on 2026-08-22.

## Question

Can the spatial cue already recovered by AF2 correct its remaining class-specific
tail trade-off without another FFT, detector continuation, ROI processing, or a
change to localization?

## Arms

- `AF2`: completed seed-42 checkpoint and validation reference; no retraining.
- `AF2RCC0`: schema-matched zero-cue identity control; static audit only.
- `AF2RCC1`: recovered AF2 cue projected into P3/P4/P5 class logits.

Every level owns one `21 x 3` matrix. All 189 weights start at zero and use
`0.10*tanh(weight/0.10)`. RGB projections are averaged, so a complete logit
correction is bounded to `[-0.10,+0.10]`. AF2 recovery runs exactly once.

## Frozen training

Only `AF2RCC1` is trained, for 20 epochs, seed 42, image size 640, batch 16,
with the native schedule recorded in the config. The complete AF2 detector,
including backbone, neck, classification towers, and box towers, is frozen.
Validation has all 21 classes. Test is unavailable and locked.

## Seed-42 gate against the original AF2 checkpoint

The candidate passes only when all conditions hold:

1. Macro mAP50-95 is no more than 0.1 point below AF2.
2. Bottom-3 mAP50-95 is not below AF2.
3. Worst-class mAP50-95 is no more than 0.5 point below AF2.
4. At least two of Macro, Bottom-3, and Worst improve strictly.
5. `kulit_tanduk_ukuran_kecil` is no more than 0.5 point below AF2.
6. All 21 validation classes are present and test is not accessed.

Only a PASS may authorize seed 123 and 2026. A FAIL closes this direction.

## Interpretation boundary

This is a parameter-efficient classification calibration of AF2, not a new
frequency transform. Its evidence must be reported separately from AF2FFAB2,
whose comparison was affected by full-detector continuation.
