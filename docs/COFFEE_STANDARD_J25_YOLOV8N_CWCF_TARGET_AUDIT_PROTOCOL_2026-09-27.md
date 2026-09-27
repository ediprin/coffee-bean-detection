# Coffee Standard J25 — YOLOv8n CWCF Target-Attribute Audit

Status: **frozen diagnostic; no training; locked test remains closed**  
Date: 2026-09-27

## Question

For the six validation instances of **Biji Hitam Pecah**, does CWCF1 fail
because its auxiliary attributes do not recover the required primitive factors
(black + broken), or because those attributes are present but the 25-class leaf
classifier/ranking does not use them effectively?

The audit compares only the completed seed-42 checkpoints:

- `V8N_MATCHED`;
- `V8N_CWCF1`.

No weights, thresholds, labels, training settings, or validation images are
modified.

## Why this audit comes before another method

The completed optimization screens gave mixed tail behavior:

- CWCF1 improved aggregate Macro and Bottom-3 over V8N_MATCHED;
- ASL1 reduced Bottom-3 and the target;
- DIR1 and P34 slightly improved the target but reduced aggregate quality;
- HYB1 and SG1 did not improve the frozen aggregate gate.

Therefore no new loss, reweighting, cue, or attention mechanism is authorized
until the existing CWCF representation is inspected directly.

## Frozen data scope

The audit reconstructs the same
`coffee-standard-j25-train-siblings-v2` development split at seed 42.

Only:

- train labels for support/context description;
- validation images and labels for model diagnostics;

are used.

The locked test is not extracted or evaluated.

## Target and confusion family

Target:

`Biji Hitam Pecah`

Local family:

- `Biji Hitam Pecah`;
- `Biji Hitam Penuh`;
- `Biji Hitam Sebagian`;
- `Biji Pecah`.

This family corresponds to the fixed CWCF primitive factors:

- black;
- broken.

The target code requires both factors.

## Per-target raw audit

For every target GT, each model is inspected at IoU >= 0.50 using the top-500
raw YOLOv8 anchors ranked by their maximum 25-class score.

For the highest-confidence localized raw anchor, record:

- IoU;
- predicted leaf class;
- target-class score;
- target-class rank among all 25 classes;
- top-5 leaf classes.

A second view chooses, among localized top-500 anchors, the anchor with the
highest target-class score. This separates absence of target evidence from
ranking by a competing class.

## Final-selection audit

The same checkpoint is decoded with:

- confidence = 0.001;
- NMS IoU = 0.70;
- max_det = 500.

For every target GT, record whether a final localized detection survives and
which class it receives.

This threshold is diagnostic only. It is not a deployment threshold and does
not change AP evaluation.

## CWCF attribute audit

For CWCF1 only, the feature pyramid entering the existing CWCF detect wrapper
is captured during ordinary eval-mode inference.

The audit then applies the already-trained:

- four-channel cue `[Cb, Cr, D1, D2]`;
- cue affine adapter;
- 14-attribute head;

to those same feature maps without changing model state.

At the exact raw anchor used for the leaf decision, record:

- `P(black)`;
- `P(broken)`;
- binary black/broken pair at threshold 0.5;
- total mismatch count against the fixed 14-attribute code;
- rank of the correct leaf under attribute-only compatibility;
- top-5 attribute-compatible leaf classes.

No attribute score is fused back into detector output. This is observation
only.

## Confusion-family audit

For every GT from the four-class family appearing in the two validation images
that contain Biji Hitam Pecah, report:

- localized raw top-1 confusion;
- for CWCF1, correctness of the predicted black/broken pair.

This distinguishes a target-specific failure from a broader inability to
separate the two primitive factors.

## Support/context audit

Without model inference on training images, report:

- target instances;
- images containing the target;
- recovered source-component count;
- target purity per image;
- class composition of every target-containing image;

for train and validation.

This is descriptive context only and does not alter the split.

## Interpretation boundary

After the audit:

- If black and broken are usually absent at target-localized CWCF anchors, the
  evidence supports an **attribute-representation/supervision failure**.
- If black and broken are usually present and the target ranks well under
  attribute compatibility but poorly under the native leaf classifier, the
  evidence supports an **attribute-to-leaf coupling/ranking failure**.
- If both attribute compatibility and leaf ranking are poor or inconsistent,
  the result remains **mixed/sparse**, and no single mechanism is claimed.

Because validation contains only six target instances, counts and paired
per-instance behavior must be reported alongside any interpretation.

## Test lock

- training executed: false;
- locked-test images evaluated: false;
- no new promotion gate is defined by this audit.
