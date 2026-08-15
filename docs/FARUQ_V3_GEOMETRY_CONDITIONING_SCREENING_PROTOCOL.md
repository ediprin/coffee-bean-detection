# Faruq-v3 GEO-C0 vs GEO1 seed42 screening protocol

Status: **frozen before training results are read**.

## Motivation

Validation-only diagnostics found that four recurring size-defined hard pairs are not explained well by simple apparent-scale overlap or boundary proximity. The closing shape audit found that mixed long-side/area evidence is strongly associated with extreme aspect ratio, and extreme shapes have higher pair-error rates in both CPE0 and CIR0. This motivates one narrow causal screen of explicit predicted-box geometry.

## Arms

All arms use the same Faruq-v3 development data, seed 42, D0 checkpoint, 50-epoch continuation schedule, image size, batch size, optimizer policy, augmentation schedule, and native YOLO26 P3/P4/P5 box path.

- **D0FT**: previously completed native D0 continuation; reference only.
- **GEO-C0**: parameter-matched zero-information control. It owns the exact same residual MLP as GEO1 but receives a Bx4xN zero tensor.
- **GEO1**: identical residual MLP receives detached predicted-box geometry `[w_norm, h_norm, area_norm, aspect_ratio]` for every classification location.

`GEO-C0` is a **parameter-matched zero-information control**, not a claim of functional equivalence.

## Geometry and gradient isolation

Geometry is computed from the detector branch's own decoded predicted boxes, not ground truth. The raw predicted box tensor is detached before decoding. Therefore the added classification residual has no direct gradient path into the box-regression outputs. The native box tensor returned to YOLO loss/inference is not modified.

The residual adapter targets only dataset classes whose names encode `kecil`, `sedang`, or `besar`. Other class logits receive zero direct residual.

The final adapter projection is initialized to exactly zero, so GEO-C0 and GEO1 begin with the same native D0 outputs before optimization.

## Frozen architecture

- geometry dimension: 4
- hidden dimension: 32
- activation: SiLU
- final projection: hidden -> `nc`, zero initialized
- max normalized side clamp: 2.0
- max aspect-ratio clamp: 10.0
- no auxiliary loss
- no GT geometry as model input
- no test split

## Primary metrics

- Macro AP50-95
- Bottom-3 class AP50-95
- Worst-class AP50-95
- Size-class mean AP50-95: mean AP50-95 over all validation classes whose labels encode `kecil`, `sedang`, or `besar`

## Frozen validity gate for GEO-C0

Relative to D0FT, GEO-C0 must satisfy all:

1. Macro drop no worse than 1.0 percentage point.
2. Bottom3 drop no worse than 2.0 points.
3. Worst drop no worse than 3.0 points.

Failure means the zero-information control itself is too disruptive for a geometry-specific causal claim.

## Frozen GEO1 retain gate

GEO1 is retained for paired multi-seed confirmation only if GEO-C0 validity passes and all conditions below hold:

1. GEO1 Macro - GEO-C0 Macro >= +0.20 percentage point.
2. GEO1 Bottom3 - GEO-C0 Bottom3 >= -0.50 point.
3. GEO1 Worst - GEO-C0 Worst >= -0.50 point.
4. At least one of Bottom3 or Worst improves over GEO-C0 by >= +0.50 point.
5. GEO1 size-class mean AP50-95 - GEO-C0 size-class mean AP50-95 >= +0.50 point.
6. GEO1 Macro is no worse than D0FT by more than 0.20 point.

No threshold may be moved after seed42 results are observed. A seed42 retain decision authorizes only a paired multi-seed confirmation; it does not authorize locked-test access.
