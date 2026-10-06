# DefectosCafeVerde DVSR1 Validation-Screen Protocol

Frozen: **2026-10-06**, after the grouped train-only DVSR routability audit
passed and before DVSR1 accesses validation.

## Question

Does the fixed train-only side-reliability selector improve the source paper's
maximum-confidence rule and the confirmed three-seed DVF1 endpoint on unseen
physical validation pairs?

## Fixed endpoint

- The final selector scale and weights are read unchanged from the passed DVSR
  audit.
- Each view is represented by the same 29 inference-visible features used by
  the audit: 12 native scores, 12 predicted-class one-hot values, top score,
  second score, margin, normalized score entropy, and total score mass.
- The shared antisymmetric scorer chooses Side A or Side B. Exact score ties
  fall back to the paper maximum-confidence choice.
- The selected native side supplies both class and its already-frozen D0 box.
- No class logits are synthesized, no boxes are changed, and no parameter is
  fitted on validation.

## Inputs and comparisons

- immutable DVSR train-only audit JSON;
- immutable DVF1 validation pair cache;
- the exact DVF1 confirmation arm and aggregate decision bound into the audit;
- endpoints: paper maximum confidence, DVSR1, confirmed DVF1 three-seed mean,
  and the non-deployable pair oracle.

Metrics are physical-pair accuracy, macro class accuracy, Bottom-3 class
accuracy, worst-class accuracy, classwise accuracy, and paper-to-DVSR1
rescue/regression counts.

## Frozen decision

The paper-control requirements are mandatory:

1. DVSR1 physical-pair gain over paper is at least 0.5 point;
2. Macro and Bottom-3 are not lower than paper;
3. worst-class loss versus paper is no more than 2 points; and
4. paper and oracle endpoints reproduce the immutable confirmation arm exactly.

DVSR1 passes against confirmed DVF1 through either route:

### Overall route

- physical-pair accuracy exceeds the three-seed DVF1 mean by at least 0.5
  point;
- Macro and Bottom-3 are not lower; and
- worst-class loss is no more than 2 points.

### Lower-tail Pareto route

- physical-pair accuracy is within 0.2 point of the DVF1 mean;
- Macro is not lower;
- Bottom-3 improves by at least 1 point; and
- worst-class accuracy is not lower.

Pass: `PROMOTE_DVSR1_SIDE_RELIABILITY`.

Fail: `RETAIN_CONFIRMED_DVF1`; the result may still be reported as a diagnostic
of train-to-validation reliability transfer, but no threshold or feature may
be retuned on this validation result.

## Locks

- selector fitting or detector/fuser training: forbidden;
- validation access: authorized exactly once by this protocol;
- test extraction or access: forbidden;
- post-result changes to features, alpha, scale, weights, or gates: forbidden;
- the pair oracle is diagnostic only.
