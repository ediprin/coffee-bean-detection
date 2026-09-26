# Coffee Standard J25 — YOLOv8n CWCF-HYB1 Screen

Status: **frozen before training**  
Date: 2026-09-26

## Question

Can the current best seed-42 CWCF1 result be improved by retaining the proven
rotation-neutral detail-energy cue while appending the signed directional Haar
bands that improved the worst class in DIR1?

Matched comparison:

```
V8N_CWCF1  vs  V8N_CWCF_HYB1
```

## CWCF1 baseline cue

```
[Cb, Cr, D1, D2]
```

with

```
D_l = sqrt((LH_l^2 + HL_l^2 + HH_l^2) / 3 + eps)
```

CWCF1 remains the current retained representation because DIR1 decreased Macro
and Bottom-3 despite improving the worst class.

## HYB1 cue

HYB1 keeps every CWCF1 cue channel and appends the six signed directional bands:

```
[Cb, Cr, D1, D2, LH1, HL1, HH1, LH2, HL2, HH2]
```

Total: 10 fixed cue channels.

The native detector still receives only the original RGB image. HYB1 affects
the existing CWCF classification path only.

## What is frozen

Unchanged from `V8N_CWCF1`:

- YOLOv8n P3 detector;
- pretrained `yolov8n.pt`;
- source-level J25 train/validation split;
- seed 42;
- 50 epochs;
- image size 640;
- batch size 16;
- patience 15;
- optimizer `auto`;
- semantic-safe augmentation;
- no repeat sampler;
- attribute BCE;
- 14 fixed J25 attributes;
- `attribute_gain = 0.15`;
- two Haar levels;
- `cue_clip = 4.0`;
- zero-initialized classification-only cue adapters;
- no HNC;
- no ASL;
- no explicit-composition gate;
- locked test remains closed.

The representation change is therefore:

```
[Cb, Cr, D1, D2]
        ->
[Cb, Cr, D1, D2, LH1, HL1, HH1, LH2, HL2, HH2]
```

No directional weighting, gain tuning, or class-specific tuning is authorized.

## Initialization control

Increasing adapter input width from 4 to 10 changes RNG consumption during
module construction. HYB1 therefore reconstructs the frozen CWCF1 seed-42
initialization and copies the CWCF1 auxiliary attribute-head initialization
into HYB1. Native detect-head state must also match.

All HYB1 cue-adapter weights and biases remain zero at initialization, so the
candidate must produce bitwise-identical native boxes and scores before the
cue path becomes active.

## Representation integrity

Before training, the preflight must prove:

1. CWCF1 energy cue has 4 channels;
2. directional cue has 8 channels;
3. HYB1 has 10 channels;
4. `HYB1[:, :4]` is exactly the complete CWCF1 cue;
5. `HYB1[:, :2]` equals the directional cue chroma channels;
6. `HYB1[:, 4:]` equals the six directional Haar channels;
7. native detect-head state equals the CWCF1 reference;
8. auxiliary attribute-head initialization equals the CWCF1 reference;
9. HYB1 adapters are zero initialized;
10. activating an appended directional channel changes class scores while
    preserving box outputs;
11. locked test remains closed.

## Baseline to beat

Completed seed-42 `V8N_CWCF1`:

- Macro mAP50–95: `0.7098727948936191`;
- Bottom-3 mAP50–95: `0.33397510007593945`;
- Worst-class mAP50–95: `0.04736111111111112`;
- Biji Hitam Pecah AP50–95: `0.04736111111111112`.

DIR1 context only:

- Macro delta vs CWCF1: `-0.0035023620441758663`;
- Bottom-3 delta: `-0.02060385682403848`;
- Worst / Biji Hitam Pecah delta: `+0.005778692247262719`.

DIR1 is not the baseline for promotion; it only motivates retaining the energy
cue while testing whether directional information adds complementary value.

## Frozen decision

Primary screen criteria:

- Macro delta > 0;
- Bottom-3 delta > 0.

If both are positive:

`PASS_HYB1_SCREEN -> RETAIN_HYBRID_REPRESENTATION`

If aggregate metrics trade off:

`PARETO_TRADEOFF_HYB1 -> REVIEW_HYBRID_REPRESENTATION`

If no aggregate advantage is obtained:

`FAIL_HYB1_SCREEN -> RETAIN_V8N_CWCF1`

Worst-class and Biji Hitam Pecah remain descriptive rather than tuning gates.

## Claim boundary

This is a single-seed representation screen. A positive result supports keeping
the hybrid cue for subsequent CWCF development only; it is not a multi-seed or
locked-test claim.

## Test lock

No test image is extracted or evaluated by this screen.
