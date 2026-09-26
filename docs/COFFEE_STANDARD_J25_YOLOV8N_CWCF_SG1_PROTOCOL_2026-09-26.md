# Coffee Standard J25 — YOLOv8n CWCF-SG1 Screen

Status: **frozen before training**  
Date: 2026-09-26

## Question

Can the retained seed-42 `V8N_CWCF1` be improved by learning the relative
strength of the existing CWCF residual independently at P3, P4, and P5?

Matched comparison:

```
V8N_CWCF1  vs  V8N_CWCF_SG1
```

## Motivation from completed screens

The current development evidence does not support replacing the compact CWCF1
cue or removing P5 outright:

- DIR1 exposed signed directional Haar bands but decreased Macro and Bottom-3;
- HYB1 retained energy and appended directional bands, but did not improve the
  aggregate gate;
- P34 removed P5 cue conditioning and decreased Macro while increasing the
  worst-class/target AP.

Therefore SG1 does not add another cue, loss, attention block, or manual pyramid
mask. It retains all three CWCF scales and lets training learn only their
residual amplitudes.

## Frozen mechanism

CWCF1 conditions each scale through a zero-initialized affine residual:

```
R_l = F_l * tanh(S_l) + B_l
F'_l = F_l + R_l
```

SG1 changes this to:

```
g_l = 2 * sigmoid(alpha_l)
F'_l = F_l + g_l * R_l
```

for `l in {P3, P4, P5}`.

There are exactly three new scalar parameters:

```
alpha_P3, alpha_P4, alpha_P5
```

and all initialize to zero. Therefore:

```
g_P3 = g_P4 = g_P5 = 1
```

at initialization.

The sigmoid parameterization bounds every learned gain to the open interval
`(0, 2)` without introducing a tuned scale-specific hyperparameter.

## What remains frozen

Unchanged from `V8N_CWCF1`:

- YOLOv8n P3 detector;
- pretrained `yolov8n.pt`;
- source-level J25 development split;
- seed 42;
- 50 epochs;
- image size 640;
- batch size 16;
- patience 15;
- optimizer `auto`;
- semantic-safe augmentation;
- no repeat sampler;
- cue `[Cb, Cr, D1, D2]`;
- two Haar levels;
- `cue_clip = 4.0`;
- attribute BCE;
- `attribute_gain = 0.15`;
- 14 fixed J25 attributes;
- P3/P4/P5 all remain active;
- zero-initialized affine cue adapters;
- no HNC;
- no ASL;
- no directional/hybrid cue;
- no explicit composition gate;
- locked test remains closed.

The three gate scalars use the normal optimizer grouping produced by the frozen
Ultralytics `optimizer=auto` path. No special learning rate or gate-specific
regularizer is introduced.

## Initialization and functional preflight

Training is authorized only if all checks pass:

1. stored baseline is the completed seed-42 `V8N_CWCF1`;
2. pretrained checkpoint, dataset hashes, sampler, and training schedule match;
3. after removing the single `learnable_scale_gates` switch, normalized CWCF
   configuration equals CWCF1;
4. every non-gate candidate tensor is bitwise equal to reconstructed CWCF1;
5. exactly three trainable parameters are added;
6. all gate logits initialize at zero and all gains initialize exactly at one;
7. with zero adapters, SG1, CWCF1, and native YOLOv8n predictions match;
8. after applying identical nonzero adapter residuals, SG1 with `g=1` remains
   bitwise equal to CWCF1;
9. changing only the P5 gate changes class scores while preserving boxes;
10. with a nonzero residual, finite nonzero gradients reach the gate parameters;
11. the attribute BCE path remains finite;
12. locked test is not accessed.

## Baseline to beat

Completed seed-42 `V8N_CWCF1`:

- Macro mAP50–95: `0.7098727948936191`;
- Bottom-3 mAP50–95: `0.33397510007593945`;
- Worst-class mAP50–95: `0.04736111111111112`;
- Biji Hitam Pecah AP50–95: `0.04736111111111112`.

## Reporting

Report:

- Macro mAP50–95;
- Bottom-3 mAP50–95;
- Worst-class mAP50–95;
- full classwise AP50–95;
- Biji Hitam Pecah AP50–95;
- learned `alpha` and `g` for P3/P4/P5;
- parameter increase versus CWCF1.

Primary delta:

```
V8N_CWCF_SG1 - V8N_CWCF1
```

## Frozen decision

If both are positive:

- Macro delta > 0;
- Bottom-3 delta > 0;

then:

`PASS_SG1_SCREEN -> RETAIN_LEARNED_SCALE_GATES`

If aggregate metrics trade off:

`PARETO_TRADEOFF_SG1 -> REVIEW_LEARNED_SCALE_GATES`

If there is no aggregate advantage:

`FAIL_SG1_SCREEN -> RETAIN_V8N_CWCF1`

Worst-class and Biji Hitam Pecah remain descriptive rather than tuning gates.

## Claim boundary

This is a single-seed scale-gating optimization screen. A positive result
supports retaining the three scale gates for further CWCF development only; it
is not a multi-seed or locked-test claim.

## Test lock

No test image is extracted or evaluated by this screen.
