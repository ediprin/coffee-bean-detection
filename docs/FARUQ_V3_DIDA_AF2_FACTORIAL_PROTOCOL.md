# Faruq-v3 Domain-Invariant Discriminative AF2 Factorial Protocol

Status: **frozen before training**  
Date: 2026-08-17

## Research question

Can the already confirmed AF2 detector improve fine-grained lower-tail
discrimination and source-domain generalization simultaneously when trained
with (a) object-level weak-to-style consistency and (b) native-logit dynamic
hard-negative separation, without changing inference architecture or latency?

## Evidence motivating this study

- AF2 is the lead confirmed Faruq-v3 standalone candidate: three-seed Macro
  mAP50-95 87.94%, Bottom-3 79.37%, Worst 78.15%.
- Against optimization-matched D0FT, AF2 improved Faruq Macro by 1.32 points
  and improved Macro in all three seeds.
- On the leakage-safe Coffee Standard external-development diagnostic, AF2
  improved Macro by 4.08 points and won all three paired seeds, but absolute
  external Bottom-3 and Worst AP remained near zero.
- Prior head, hierarchy, metric-learning, multilevel, direct fusion, and
  distillation attempts do not establish that another inference module should
  be stacked on AF2.

## Dataset and split

- Training and model selection: `faruq-development-v3-grouped` train/val only.
- Test is locked and must not be present in the runtime dataset.
- Coffee Standard v8 is an external-development diagnostic because it has
  already been inspected repeatedly. It is not a new final test.
- No variety-robustness claim is authorized because verified variety metadata
  is unavailable.

## Frozen 2 x 2 design

| Arm | DG style + consistency | FG Top-3 margin |
|---|---:|---:|
| `AF2FT` | no | no |
| `AF2DG` | yes | no |
| `AF2FG` | no | yes |
| `AF2DGFG` | yes | yes |

All arms:

- initialize from the same completed AF2 seed-42 checkpoint;
- execute two forwards per batch;
- average native detection loss across the forwards;
- use the same 50 epochs, optimizer, batch size, seed, augmentation baseline,
  early-stopping rule, and O2M/O2O schedule;
- preserve the exact AF2 inference graph and parameter schema.

For DG-off arms, the second image tensor is an exact clone. For DG-on arms,
the second view receives only geometry-preserving, mild global appearance
transforms before deterministic AF2. No crop, resize, affine, blur, cutout, or
spatial resampling is introduced by this study.

## Objective

The paired native detection term is

```text
L_det_pair = 0.5 * (L_YOLO26(weak, y) + L_YOLO26(style, y)).
```

Branch-specific positive anchors are aggregated by `(image_id, padded_gt_id)`.
The two views are never paired by anchor index. DG uses a detached weak-view
teacher and a style-view student:

```text
L_DG = T^2 * KL(stopgrad(softmax(q_weak/T)) || softmax(q_style/T)).
```

FG acts directly on native class logits aggregated per GT:

```text
L_FG = mean log(1 + sum_{k in Top3(z_not_y)} exp(z_k - z_y + margin)).
```

Top-3 rivals are selected dynamically from training predictions only. No
validation confusion list is used. O2M and O2O use their own native assignment
sets and the native time-varying branch weights.

Frozen constants:

- `lambda_DG = 0.10`;
- `lambda_FG = 0.05`;
- temperature `T = 2.0`;
- margin `m = 0.20`;
- `K = 3` rivals.

## Static authorization gate

Training is forbidden until all conditions pass:

1. four configs form the intended 2 x 2 flags;
2. all candidates have the same state-dict schema and parameter count as AF2;
3. evaluation-mode candidate inference is numerically identical to AF2 after
   the same checkpoint is loaded;
4. DG-off creates an exact cloned view;
5. DG-on preserves BCHW shape and finite `[0,1]` values;
6. appearance transformation has no geometry operation;
7. GT matching uses `(image, GT)` and tolerates different anchor assignments;
8. FG and DG losses and gradients are finite;
9. auxiliary functions receive classification logits only and do not consume
   decoded boxes;
10. no test path is accessed.

## Seed-42 prospective gate

Primary metrics are Macro mAP50-95, Bottom-3 class mAP50-95, and Worst-class
mAP50-95. Hard-confusion error and matched-GT coverage are diagnostic only.

`AF2DGFG` passes screening only if all are true:

1. Macro is not lower than `AF2FT`;
2. Bottom-3 is higher than `AF2FT`;
3. Worst is no more than 1 point below `AF2FT`;
4. Macro is at least 0.5 point higher than both `AF2DG` and `AF2FG`;
5. Bottom-3 is not lower than either `AF2DG` or `AF2FG`;
6. inference parameter count is unchanged.

The effects reported are:

```text
E_DG          = AF2DG   - AF2FT
E_FG          = AF2FG   - AF2FT
E_interaction = AF2DGFG - AF2DG - AF2FG + AF2FT
```

If the joint arm fails, stop without extra seeds or test access. If it passes,
freeze the complete method and authorize paired seeds 123 and 2026 for all
four arms. Cross-domain evaluation happens only after validation confirmation.

## Claim boundary

A successful study supports a training-objective contribution for an AF2
one-stage coffee-defect detector. It does not support robustness to coffee
variety, a target-domain deployment claim, or a new final-test claim without
an untouched external dataset.
