# Coffee Standard J25 — YOLOv8n Matched Capacity/Family Probe

Status: **frozen before training**  
Date: 2026-09-20

## Purpose

This experiment determines whether the large seed-42 gap observed between
`V8S_MATCHED` and the YOLO26n controls is primarily associated with model
capacity/scale or persists when comparing nano-scale models.

It is a descriptive diagnostic arm, not a new thesis method and not a
promotion/kill gate for CWCF1.

## Frozen arm

`V8N_MATCHED` uses:

- dataset: `coffee-standard-j25-train-siblings-v2`;
- the same 315 train source identities / 695 retained training sibling images;
- the same 68 class-complete validation source identities;
- the locked 68-source test manifest remains unextracted;
- official `yolov8n.pt` pretrained weights;
- seed 42;
- 50 epochs;
- image size 640;
- batch size 16;
- patience 15;
- optimizer `auto`;
- no identity repeat sampler;
- the exact semantic-safe augmentation dictionary used by `SAFEAUG0`,
  `CWCF1`, and `V8S_MATCHED`.

Only the detector family/scale changes.

## Comparison set

Validation-only seed-42 comparison:

1. `V8N_MATCHED`
2. `V8S_MATCHED`
3. `SAFEAUG0` (YOLO26n)
4. `CWCF1` (YOLO26n + chromatic-wavelet classification path)

Report:

- Macro mAP50–95;
- Bottom-3 class mAP50–95;
- Worst-class mAP50–95;
- `Biji Hitam Pecah` AP50–95.

Also record the YOLOv8n pretrained source parameter count.

## Interpretation boundary

The following deltas are descriptive:

- `V8N_MATCHED - SAFEAUG0`: nano-scale YOLOv8 versus native YOLO26n under
  the same development protocol;
- `V8N_MATCHED - CWCF1`: nano-scale YOLOv8 versus the current proposed
  YOLO26n variant;
- `V8S_MATCHED - V8N_MATCHED`: within-family YOLOv8 scale/capacity gap.

A large positive `V8S - V8N` gap supports a substantial capacity contribution.
A large positive `V8N - YOLO26n` gap indicates that capacity alone does not
explain the observed detector-family difference.

No fixed numerical threshold is used to force either interpretation from a
single seed. The exact deltas are reported.

## Decision boundary

Completion yields:

`CAPACITY_FAMILY_PROBE_COMPLETE`

regardless of ordering.

No additional architecture tuning is authorized by this probe alone.

## Test lock

- test images are not extracted;
- no model evaluates the locked test;
- this arm does not authorize final-test access.
