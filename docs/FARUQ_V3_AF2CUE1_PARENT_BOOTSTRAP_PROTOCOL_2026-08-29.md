# AF2CUE1 Paired Parent-Bootstrap Protocol — 2026-08-29

## Status and purpose

Frozen before execution. This is a validation-only post-hoc uncertainty audit
of three already-trained seed-42 checkpoints: `AF2BASE`, `AF2SPDS`, and
`AF2CUE1`. It does not train, tune, or open test data.

The preceding class-composition bootstrap was informative but treated class
AP values as if they were independent even though classes share images. This
audit instead resamples complete independent source-parent clusters recorded
in `faruq_grouped_manifest.json`. Every sibling image belonging to a sampled
parent is repeated together, and the identical cluster sample is applied to
all three arms.

## Frozen implementation

- Validation split: Faruq-v3 grouped, all 294 images and all 21 classes.
- Checkpoints: exact paths and SHA256 values read from the completed arm result
  documents; generic `best.pt` discovery is forbidden.
- Inference: fixed 640 pixels, confidence 0.001, IoU 0.7, maximum 500
  detections, no augmentation, and no test split.
- Bootstrap: 1,000 paired source-parent cluster replicates, RNG seed 20260829.
- A replicate that omits every target of any one of the 21 classes is rejected
  and redrawn, because Bottom-3 and Worst are undefined under incomplete
  ontology coverage.
- Reported metrics: Macro mAP50–95, Bottom-3 class mAP50–95, and Worst-class
  mAP50–95, with point delta, percentile 95% interval, and empirical
  probability of a positive paired delta.

The custom observation evaluator must reproduce the historical Ultralytics
endpoint for each arm within `1e-6` before bootstrap is allowed.

## Interpretation gate

`PARENT_BOOTSTRAP_SUPPORTIVE` requires:

1. AF2CUE1 versus AF2BASE Macro probability-positive at least 95%;
2. AF2CUE1 point Bottom-3 and Worst deltas versus AF2BASE nonnegative; and
3. AF2CUE1 versus AF2SPDS Macro probability-positive at least 95%.

This gate assesses whether the observed Pareto direction is distributed over
source parents. It does **not** revise the frozen `FAIL_KILL_GATE`, establish
independent-dataset superiority, or authorize a test claim. The same reused
validation set already selected these candidates, so any supportive result is
explicitly post-hoc.
