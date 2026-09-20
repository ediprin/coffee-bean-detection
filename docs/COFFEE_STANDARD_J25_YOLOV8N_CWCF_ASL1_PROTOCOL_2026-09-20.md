# Coffee Standard J25 — YOLOv8n CWCF-ASL1 Screen

Status: **frozen before training**  
Date: 2026-09-20

## Question

Can the current best seed-42 CWCF1 result be improved by changing only the
multi-label attribute-supervision loss from BCE to Asymmetric Loss (ASL)?

Matched comparison:

```
V8N_CWCF1  vs  V8N_CWCF_ASL1
```

## What is frozen

The following are unchanged from `V8N_CWCF1`:

- YOLOv8n P3 detector;
- `coffee-standard-j25-train-siblings-v2`;
- source split and development hashes;
- seed 42;
- 50 epochs;
- image size 640;
- batch size 16;
- patience 15;
- optimizer `auto`;
- semantic-safe augmentation;
- no repeat sampler;
- CWCF cue: Cb, Cr, Haar-L1 detail, Haar-L2 detail;
- zero-initialized classification-only cue adapters;
- 14 fixed J25 attributes;
- `attribute_gain = 0.15`;
- `cue_clip = 4.0`;
- no HNC;
- no explicit composition gate;
- locked test remains closed.

The only experimental change is:

```
attribute BCE  ->  attribute ASL
```

## Frozen ASL

The ASL setting follows the combined configuration reported in the ICCV 2021
paper and used by the authors' released COCO training code:

- positive focusing `gamma_pos = 0`;
- negative focusing `gamma_neg = 4`;
- negative probability margin / clip `m = 0.05`;
- focal weighting is computed without differentiating through the weighting
  factor, matching the released training configuration.

No ASL hyperparameter sweep is authorized for this screen.

For attribute logit `z`, probability `p = sigmoid(z)`, and binary target
`y`, the negative probability branch is shifted by the margin:

```
q_neg = min(1 - p + m, 1)
```

and the positive/negative focusing exponents are asymmetric. Easy negatives are
therefore down-weighted more strongly than positives.

## Important implementation boundary

The original `V8N_CWCF1` does **not** enable the optional
`class_balanced_attributes` switch. ASL1 keeps that setting unchanged.

Therefore this experiment isolates:

```
standard BCE reduction  ->  standard ASL reduction
```

and does not silently add leaf-class reweighting.

## Preflight

Training is authorized only if all gates pass, including:

1. the stored baseline is the completed seed-42 `V8N_CWCF1` protocol;
2. pretrained checkpoint bytes match the baseline contract;
3. data hashes and training schedule match;
4. after excluding ASL-only configuration keys, the CWCF mechanism mapping is
   identical to `V8N_CWCF1`;
5. baseline CWCF configuration still matches the original frozen CWCF1
   mechanism;
6. ASL is exactly `gamma_pos=0, gamma_neg=4, clip=0.05`;
7. candidate and CWCF1 reference model state are bitwise identical before
   training, proving the loss switch does not alter model initialization;
8. the ASL probe loss and gradients are finite and nonzero;
9. locked test is not accessed.

## Reporting

Report seed-42 validation:

- Macro mAP50–95;
- Bottom-3 class mAP50–95;
- Worst-class mAP50–95;
- full classwise AP50–95;
- Biji Hitam Pecah AP50–95.

Primary delta:

```
V8N_CWCF_ASL1 - V8N_CWCF1
```

Current baseline to beat:

- Macro mAP50–95: `0.7098727948936191`;
- Bottom-3: `0.33397510007593945`;
- Worst: `0.04736111111111112`;
- Biji Hitam Pecah: `0.04736111111111112`.

## Frozen screen decision

If both are positive:

- Macro delta > 0;
- Bottom-3 delta > 0;

then:

`PASS_ASL_SCREEN -> RETAIN_ASL_AND_CONTINUE_CWCF_IMPROVEMENT`

If aggregate metrics trade off:

`PARETO_TRADEOFF_ASL -> REVIEW_ASL_BEFORE_NEXT_CHANGE`

If there is no aggregate advantage:

`FAIL_ASL_SCREEN -> RETAIN_V8N_CWCF1`

Worst-class and Biji Hitam Pecah remain descriptive rather than tuning gates.

## Claim boundary

This is a single-seed optimization screen. A positive result supports retaining
ASL for the next CWCF-improvement stage; it is not a multi-seed or locked-test
claim.

## Test lock

No test image may be extracted or evaluated by this screen.
