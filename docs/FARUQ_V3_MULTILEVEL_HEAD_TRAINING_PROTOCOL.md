# Faruq-v3 Capacity-Matched Multilevel Head Training Protocol

Version: `v1.0.0`

Frozen: 2026-08-02, before MHC0 or MHF1 dataset training

## Question

When detector capacity, schedule, candidate assignment, and loss weights are
matched, does P3+P4+P5 ROI fusion improve fine-grained Faruq-v3 detection over
both native D0 and the P5-only capacity control?

## Models

- `D0`: completed native YOLO26n seed 42; it is reused and never retrained.
- `MHC0`: capacity-matched P5-only residual classifier.
- `MHF1`: P3+P4+P5 residual classifier.

MHC0 and MHF1 each contain 3,180,939 parameters and use the exact state schema
that passed `faruq-v3-multilevel-head-static-v1`. The native box and class
branches are preserved. Their only architectural difference is the
parameter-free addition of normalized P3+P4 context in MHF1.

## Frozen training

- dataset: leakage-safe Faruq-v3 grouped development train/validation;
- test split: unavailable and locked;
- seed: 42 only;
- initialization: official `yolo26n.pt` transferred into the same D0 graph;
- epochs: 50; image size: 640; batch: 16; workers: 2;
- optimizer: Ultralytics `auto`; deterministic mode enabled;
- close mosaic: final 10 epochs; patience: 15; `max_det=500`;
- ROIAlign: 3 x 3, P3/P4/P5 mean+maximum descriptors;
- auxiliary CE weight: 0.5; residual inference weight: 0.5;
- candidates: one-to-one branch, top 500, class-agnostic greedy IoU matching
  at IoU >= 0.5;
- epochs 1-10 use ground-truth ROIs; epochs 11-26 linearly transition to
  matched predicted ROIs; epochs 27-50 use predicted ROIs when matches exist;
- native candidate logits are detached inside the auxiliary predicted-ROI
  loss, while feature gradients remain enabled;
- checkpoint and CSV are written directly to the one shared Drive project and
  can resume from `last.pt` after account/runtime changes.

## Metrics and comparisons

Validation-only reporting uses Macro mAP50-95, bottom-3 class mAP50-95, and
worst-class mAP50-95. Report D0 vs MHC0, D0 vs MHF1, and MHC0 vs MHF1.

MHF1 passes only if all are true:

1. versus D0: Macro improves by at least 0.5 point;
2. versus D0: bottom-3 does not decrease;
3. versus D0: worst-class does not decrease by more than 1 point;
4. versus MHC0: Macro improves by at least 0.5 point;
5. versus MHC0: bottom-3 does not decrease; and
6. versus MHC0: worst-class does not decrease by more than 1 point.

Failure stops the multilevel head without extra seeds or test. PASS authorizes
three-seed confirmation protocol design only; it does not itself open test.
