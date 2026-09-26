# Coffee Standard J25 — YOLOv8n CWCF-P34 Screen

Status: **frozen before training**  
Date: 2026-09-26

## Question

Can the current seed-42 CWCF1 result be improved by applying the unchanged CWCF
cue only at P3 and P4, while leaving P5 on its native classification feature?

Matched comparison:

```
V8N_CWCF1  vs  V8N_CWCF_P34
```

## What changes

CWCF1 injects the same four-channel cue at all three detection scales:

```
P3: CWCF
P4: CWCF
P5: CWCF
```

P34 changes only the injection mask:

```
P3: CWCF
P4: CWCF
P5: native
```

The cue itself remains exactly:

```
[Cb, Cr, D1, D2]
```

with two-level Haar detail energy.

## What remains frozen

- YOLOv8n P3 detector;
- pretrained `yolov8n.pt`;
- J25 source-level development split;
- seed 42;
- 50 epochs;
- image size 640;
- batch size 16;
- patience 15;
- optimizer `auto`;
- semantic-safe augmentation;
- no repeat sampler;
- cue channels = 4;
- `attribute_gain = 0.15`;
- two Haar levels;
- `cue_clip = 4.0`;
- attribute BCE;
- 14 fixed J25 attributes;
- zero-initialized cue adapters;
- no HNC;
- no ASL;
- no explicit composition gate;
- locked test remains closed.

The P5 auxiliary attribute head is retained and is evaluated on the native P5
feature. Therefore the screen isolates **cue conditioning location**, not removal
of the auxiliary attribute objective.

## Initialization and functional preflight

P34 has the same module shapes and initialization as CWCF1. Before training,
the preflight must prove:

1. candidate and reconstructed CWCF1 model states are bitwise identical;
2. both start with native YOLOv8n boxes and class scores;
3. CWCF1 active mask is exactly `[P3, P4, P5]`;
4. P34 active mask is exactly `[P3, P4]`;
5. manually activating only the P5 adapter changes CWCF1 scores;
6. the same P5-adapter perturbation has no effect on P34 scores;
7. P3 adapter perturbation still changes P34 scores;
8. box outputs stay unchanged under all cue probes;
9. test remains closed.

This means the training comparison differs only in whether P5 receives the
cue-conditioned residual.

## Baseline to beat

Completed seed-42 `V8N_CWCF1`:

- Macro mAP50–95: `0.7098727948936191`;
- Bottom-3 mAP50–95: `0.33397510007593945`;
- Worst-class mAP50–95: `0.04736111111111112`;
- Biji Hitam Pecah AP50–95: `0.04736111111111112`.

## Frozen decision

Primary criteria:

- Macro delta > 0;
- Bottom-3 delta > 0.

If both are positive:

`PASS_P34_SCREEN -> RETAIN_P3P4_INJECTION`

If aggregate metrics trade off:

`PARETO_TRADEOFF_P34 -> REVIEW_P3P4_INJECTION`

If no aggregate advantage is obtained:

`FAIL_P34_SCREEN -> RETAIN_V8N_CWCF1`

Worst-class and Biji Hitam Pecah remain descriptive, not tuning gates.

## Claim boundary

This is a single-seed injection-location screen. A positive result supports
retaining P3-P4 selective conditioning for subsequent CWCF development only.
It is not a multi-seed or locked-test claim.

## Test lock

No test image is extracted or evaluated by this screen.
