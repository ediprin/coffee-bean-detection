# Faruq-v3 AF2-FFA Bounded Refinement

Status: frozen before bounded training  
Date: 2026-08-22  
Test status: locked; no test access is authorized.

## Evidence motivating this refinement

The completed seed-42 AF2-FFA screen compared an equal-capacity continuation
control (`AF2FFA0`) with the unbounded spectral candidate (`AF2FFA1`). The
candidate changed Macro/Bottom-3/Worst-class mAP50-95 by -0.33/+0.82/+2.87
percentage points. It therefore exposed useful difficult-class signal but also
redistributed performance across classes. In particular, several classes fell
by about four to five points while the previous worst class improved.

This is a bounded refinement, not a retrospective rewrite of the original
screen. The original `FAIL` decision remains preserved.

## Frozen candidate

Only one new arm is trained:

| Arm | Descriptor | Residual multiplier |
|---|---|---|
| `AF2FFAB1` | AF2FFA1 high-frequency ratio | `1 + 0.10*tanh(alpha)*gate` |

`AF2FFAB1` starts from the same AF2 seed-42 checkpoint and uses the same
YOLO26n-P3 detector, AF2 frontend, dataset, 30-epoch continuation schedule,
seed, and classification-only wiring as the completed AF2FFA study. The 10%
cap is frozen before training and is not searched on validation.

The immutable completed reports for `AF2FFA0` and `AF2FFA1` are reused. They
must record seed 42, the identical initial AF2 checkpoint SHA256, no missing
metrics, and `test_images_accessed: false`. Consequently this study trains one
new model rather than repeating two controls.

## Static gates

Before training, the audit must confirm:

1. exact AF2 identity at initialization;
2. unchanged box outputs when the adapter is active;
3. identical parameter count and state schema to AF2FFA1;
4. the only AF2FFA1/AF2FFAB1 configuration change is the fixed gain cap;
5. the multiplier cannot depart from identity by more than 10%;
6. finite nonzero adapter gradients; and
7. no ROI, decoded-box dependency, or test access.

## Seed-42 Pareto decision

`AF2FFAB1` is retained as a deferred multiseed candidate only if all conditions
hold:

- Macro is no more than 0.1 point below AF2FFA0;
- Bottom-3 is at least 0.5 point above AF2FFA0;
- Worst-class AP is at least 1 point above AF2FFA0;
- Macro is higher than AF2FFA1; and
- Bottom-3 and Worst are each no more than 0.5 point below AF2FFA1.

PASS is labeled `RETAIN_PARETO`; it does not automatically authorize expensive
additional seeds. FAIL stops this refinement and retains original AF2. The test
is never opened by this protocol.
