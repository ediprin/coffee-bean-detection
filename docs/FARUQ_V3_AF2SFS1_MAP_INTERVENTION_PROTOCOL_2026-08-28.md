# Faruq-v3 AF2SFS1 mAP Intervention Protocol

Date frozen: 2026-08-28

Status: **AUTHORIZED — VALIDATION-ONLY, NO TRAINING**

Test: **closed**

## Motivation

The first root-cause diagnostic found that AF2SFS1 improved completed Macro
mAP50–95 by 0.95 point, but did not improve correct-decision recall at the
single confidence-0.25/IoU-0.50 operating point when compared with an adapter
bypass. Because AP integrates confidence ranking and IoU thresholds, that
operating-point intervention cannot explain the completed AP result.

## Frozen decomposition

Run the native Ultralytics validation metric on the same AF2SFS1 checkpoint
under four inference states:

- `NORMAL`: learned selector and residual;
- `BYPASS`: adapter output replaced by its input;
- `SPATIAL_ONLY`: selector forced to the spatial path;
- `FREQUENCY_ONLY`: selector forced to the frequency-detail path.

The completed AF2CTRL report is the matched external control. For every AP
metric, decompose:

```text
total AF2SFS1 gain       = NORMAL - AF2CTRL
direct selector effect  = NORMAL - BYPASS
optimization-mediated   = BYPASS - AF2CTRL
```

The equality is checked numerically. `NORMAL` must reproduce the completed
AF2SFS1 report before any intervention is interpreted.

## Outputs

Report Macro, Bottom-3, Worst-class, AP50, AP75, and all 21 per-class values
for every state. For each class, report total, direct-selector, and
optimization-mediated mAP50–95 deltas. Also compare AF2CTRL and AF2SFS1
checkpoint tensor drift in these groups:

- feature extractor;
- regression head;
- classification head;
- other detector state;
- SFS adapter-only state.

Tensor drift is descriptive. A large norm does not by itself prove that a
parameter group caused an AP change.

## Validity gates

1. root-cause report is `INTERPRETABLE`, training false, and test false;
2. completed reports identify AF2CTRL/AF2SFS1 seed 42 and test false;
3. normal intervention reproduces completed AF2SFS1 headline metrics within
   `1e-6`;
4. all four states contain all 21 validation classes;
5. total equals direct plus optimization-mediated within `1e-10`;
6. only validation is evaluated and no training is called.

## Interpretation

- If `NORMAL - BYPASS` explains the positive Macro gain, the active selector
  is an inference mechanism.
- If `BYPASS - AF2CTRL` explains the gain while `NORMAL - BYPASS` is negligible,
  SFS is primarily an optimization scaffold/regularizer for this run.
- If spatial-only or frequency-only dominates normal, the learned mixture is
  not the best inference state; this remains diagnostic and cannot authorize
  post-hoc model selection.

No result from this audit changes the frozen seed-42 screening decision or
authorizes test access.
