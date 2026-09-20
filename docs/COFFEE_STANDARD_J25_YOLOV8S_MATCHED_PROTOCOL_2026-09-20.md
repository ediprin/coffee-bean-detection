# Coffee Standard J25 — YOLOv8s Matched Baseline Protocol

Status: **frozen before training**  
Date: 2026-09-20

## Purpose

This experiment adds a matched YOLOv8s baseline for the J25 thesis-derived
dataset. It is not a new model candidate and it is not a promotion/kill gate for
CWCF1. Its role is to separate detector-family differences from the historical
thesis split and training protocol.

The historical thesis result remains contextual evidence only because it used a
different supplied Roboflow partition and a different training budget.

## Frozen arm

`V8S_MATCHED` uses:

- dataset: `coffee-standard-j25-train-siblings-v2`;
- the same 315 train source identities / 695 retained train sibling images as
  `SAFEAUG0`;
- the same 68 class-complete validation source identities;
- the same locked 68-source test manifest, never extracted for development;
- official `yolov8s.pt` pretrained weights;
- seed 42;
- 50 epochs;
- image size 640;
- batch size 16;
- patience 15;
- optimizer `auto`;
- no identity repeat sampler;
- the exact semantic-safe augmentation dictionary used by `SAFEAUG0`.

Only the detector family changes. Data partition, development/test boundary,
training budget, and augmentation policy remain fixed.

## Semantic-safe augmentation contract

- mosaic = 0.0
- scale = 0.0
- translate = 0.05
- degrees = 0.0
- shear = 0.0
- perspective = 0.0
- horizontal flip = 0.5
- vertical flip = 0.0
- HSV hue = 0.0
- HSV saturation = 0.2
- HSV value = 0.2
- erasing / mixup / cutmix / copy-paste = 0.0

The runner hard-checks that this training dictionary is byte-for-byte equivalent
at the parsed YAML level to `SAFEAUG0.yaml`.

## Reporting

Report, on validation only:

- Macro mAP50–95;
- Bottom-3 class mAP50–95;
- Worst-class mAP50–95;
- `Biji Hitam Pecah` AP50–95.

Also record parameter count and the exact pretrained checkpoint SHA256 used by
the run.

The comparison table may place `V8S_MATCHED`, `SAFEAUG0`, and `CWCF1`
side-by-side, but no promotion criterion is derived from the ordering.

## Historical reference boundary

The thesis reports YOLOv8s mAP50 = 0.773 and mAP50–95 = 0.616 in a table with
113 images and 1,606 instances. Those counts match the thesis validation split,
not its stated 67-image test split. Therefore the 0.616 value is retained as a
historical validation reference, not a head-to-head result.

The thesis also reports `Biji Hitam Pecah` mAP50–95 = 0.827 on 81 instances
from only two images. This is retained only as historical context.

## Decision boundary

Completion of this arm yields:

`DESCRIPTIVE_BASELINE_COMPLETE`

regardless of whether YOLOv8s is numerically above or below CWCF1.

CWCF1 must still be judged against its matched YOLO26 native control under the
separately frozen multi-seed confirmation protocol.

## Test lock

- no test image is extracted into the development dataset;
- no model may evaluate the locked test during this arm;
- no result from this arm authorizes locked-test access.
