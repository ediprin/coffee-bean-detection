# Auliya fold A — RGBMARGIN1 strongest-rival classification loss

Frozen before training. On the same 119 validation images, tracing the exact
anchor showed that the true class was second in 19/31 raw top-1 mistakes
(5/10 on the six prespecified fine-grained classes). Only 6/19 second-place
true scores were already at least 0.25. That motivates learning a better
separation rather than applying a post-hoc score swap.

## One treatment

- `RGBMARGIN1` uses the unchanged RGB 640 YOLO26s-seg forward, native P3/P4/P5,
  assigner, class BCE, bounding boxes, masks, and native end-to-end schedule.
  No new parameters, features, preprocessing, crop, or inference reranker.
- For each native assigned positive anchor, add
  `0.10 * native_cls_gain * mean(max(0, log(2) + strongest_wrong_logit - true_logit))`
  to its branch's classification loss. The strongest rival is dynamic across
  all 21 classes; no hand-picked validation pairs. Zero with no positive
  anchors. Both one-to-many and one-to-one use the same rule and retain native
  E2E mixing. The margin and coefficient are frozen, not searched.
- At the same outputs, assignment and all non-class loss terms must be
  identical. Training can still change shared features, scores used by the
  assigner, and empirical box/mask quality. Compare detection after training.
- This is not a new loss family. Faruq's ACMC1-HCR used one-to-one hard-rival
  softplus on an adapted detector; DIDA used GT-aggregated top-k separation.
  This screen isolates a per-positive hinge in native Auliya segmentation,
  from official YOLO weights.

## Locked screen

- One seed 0 run, new optimizer, official YOLO26s-seg pretrained asset and
  80-epoch cap/patience 25, matched with the exact historical RGB control.
  Same 676 train and 119 validation images, training arguments, 640 input,
  software/GPU contract. The baseline checkpoint is for comparison only.
- Primary: top-1 class accuracy on exactly the same localized instances of
  the 18 primary classes, paired by box geometry only (IoU ≥ 0.50 and
  confidence ≥ 0.25). Report macro/bottom-3/worst, paired corrections and new
  mistakes, image bootstrap and sensitivity to matching thresholds. Also
  report six prespecified confusion classes, coverage, box AP and mask AP.
- One seed is exploratory. Look for a positive, directionally stable gain
  on matched objects and no material localization or AP regression. Mixed
  results do not promote the method. No test access or validation-based
  tuning of the loss; no automatic extra seeds.
- Colab requires explicit training opt-in, matching CUDA environment and
  passing preflight. Preflight uses synthetic labels and performs no optimizer
  step. The held-out acquisition group A is not extracted or accessed.
