# DefectosCafeVerde Routed Validation Screen Protocol

Frozen: **2026-09-29**, before native-RGB train audit completion and before
routed validation inference.

## Question

Can the train-only, inference-observable expert router improve native
DefectosCafeVerde validation mAP without ground-truth routing, detector
training, or test access?

## Prerequisite

The `defectoscafeverde-train-routability-v2-rgb-native` audit must return
`AUTHORIZE_SEPARATE_ROUTED_ARCHITECTURE_SCREEN`.  Its full-train minimal subset
and fixed ridge alpha are immutable.  The router is fitted once from its train
records before any validation image is read.

## Runtime architecture

1. Native `D0DIRECT` final detections at confidence 0.001 are anchors.
2. Each selected expert is associated one-to-one to D0 anchors at IoU 0.30.
3. The frozen router receives only the same appearance, predicted class,
   confidence, association, and anchor-IoU features used by the train audit.
4. The selected expert contributes its box, class, and confidence.  If it has
   no associated detection, the system falls back to D0.
5. Unmatched candidate detections are not added.  No score calibration,
   threshold search, extra NMS, or validation fitting is allowed.

## Endpoint calibration

The manually accumulated D0 endpoint must reproduce the historical D0 Macro,
Bottom-3, and Worst-class mAP50-95 within 0.1 percentage point each.  Failure
invalidates the screen rather than changing evaluation code after seeing the
routed result.

## Frozen validation gate

After endpoint calibration, the candidate is retained through either route:

- overall route: Macro gain at least +0.5 point, Bottom-3 non-lower, and Worst
  drop no greater than 0.5 point;
- lower-tail route: Macro drop no greater than 0.1 point, Bottom-3 gain at
  least +0.5 point, and Worst non-lower.

All 12 validation classes must be present.  This one validation screen does
not authorize a test claim.  Failure stops routed fusion; passing retains the
router for a separately frozen confirmation/efficiency study.

## Locks

- detector training: forbidden;
- router fitting from validation: forbidden;
- validation-driven subset, feature, alpha, threshold, association, fallback,
  or postprocess changes: forbidden;
- test extraction/access: forbidden.

