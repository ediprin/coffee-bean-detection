# Coffee Standard J25 — YOLOv8n CWCF-DIR1 Screen

Status: **frozen before training**  
Date: 2026-09-26

## Question

Can the current best seed-42 CWCF1 result be improved by preserving the
directional Haar detail bands instead of collapsing them into rotation-neutral
detail energy?

Matched comparison:

```
V8N_CWCF1  vs  V8N_CWCF_DIR1
```

## Current CWCF1 representation

CWCF1 uses four fixed cue channels:

```
[Cb, Cr, D1, D2]
```

where each detail-energy channel is

```
D_l = sqrt((LH_l^2 + HL_l^2 + HH_l^2) / 3 + eps)
```

This is compact and rotation-neutral, but it discards the identity and sign of
the three directional Haar sub-bands.

## DIR1 representation

DIR1 preserves the signed directional detail bands:

```
[Cb, Cr, LH1, HL1, HH1, LH2, HL2, HH2]
```

The same orthonormal Haar filters and the same two-level luminance pyramid are
used. Each cue channel is independently standardized and clipped exactly as in
CWCF1.

No RGB channel is replaced: the native YOLOv8n detector still receives the
original RGB image. The eight-channel cue is consumed only by the existing
CWCF classification adapters.

## What is frozen

Unchanged from `V8N_CWCF1`:

- YOLOv8n P3 detector;
- pretrained `yolov8n.pt`;
- J25 source-level train/validation split;
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
- zero-initialized cue-conditioned classification adapters;
- no HNC;
- no explicit-composition gate;
- locked test remains closed.

The only representation change is:

```
[Cb, Cr, D1, D2]
        ->
[Cb, Cr, LH1, HL1, HH1, LH2, HL2, HH2]
```

## Initialization control

Changing the cue width from 4 to 8 changes the adapter convolution shape.
Although adapter weights are zero-initialized, constructing a wider convolution
would normally consume a different amount of RNG state and could alter the
later auxiliary attribute-head initialization.

DIR1 therefore reconstructs the frozen CWCF1 seed-42 initialization and copies
only the auxiliary attribute-head initialization into DIR1 before training.
The native YOLOv8n detect head is loaded from the same pretrained source and the
directional adapters remain exactly zero.

The preflight requires:

- native detect-head state equals the CWCF1 reference;
- auxiliary attribute-head state equals the CWCF1 reference;
- DIR1 adapters are all zero;
- initial raw boxes and class scores are bitwise identical to native YOLOv8n;
- activating one directional cue-adapter weight changes scores but leaves boxes
  bitwise unchanged.

This prevents the representation comparison from being confounded by an
unintended auxiliary-head initialization change.

## Operator integrity

The preflight also verifies that:

- the original energy cue has 4 channels;
- DIR1 has 8 channels;
- Cb and Cr are unchanged;
- the LL decomposition is unchanged;
- the original level-1 and level-2 detail energies are exactly reconstructible
  from the retained LH/HL/HH bands.

Thus DIR1 preserves the information used by CWCF1 before exposing the
directional components separately.

## Baseline to beat

Completed seed-42 `V8N_CWCF1`:

- Macro mAP50–95: `0.7098727948936191`;
- Bottom-3 mAP50–95: `0.33397510007593945`;
- Worst-class mAP50–95: `0.04736111111111112`;
- Biji Hitam Pecah AP50–95: `0.04736111111111112`.

## Frozen decision

Primary screen criteria:

- Macro delta > 0;
- Bottom-3 delta > 0.

If both are positive:

`PASS_DIR1_SCREEN -> RETAIN_DIRECTIONAL_REPRESENTATION`

If aggregate metrics trade off:

`PARETO_TRADEOFF_DIR1 -> REVIEW_DIRECTIONAL_REPRESENTATION`

If no aggregate advantage is obtained:

`FAIL_DIR1_SCREEN -> RETAIN_V8N_CWCF1`

Worst-class and Biji Hitam Pecah remain descriptive rather than tuning gates.

## Claim boundary

This is a single-seed representation optimization screen. A positive result
means only that exposing directional Haar bands improved the matched seed-42
CWCF1 comparison. It is not a multi-seed or locked-test claim.

## Test lock

No test image is extracted or evaluated by this screen.
