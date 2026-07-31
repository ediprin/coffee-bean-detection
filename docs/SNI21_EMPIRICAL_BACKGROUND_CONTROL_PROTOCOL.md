# SNI-21 Empirical Donor-Background Control

## Research question

Does replacing the procedural background with held-out real validation
background recover the classification performance lost by FC2, while keeping
the foreground cutout, target bbox, canvas size, and checkpoint fixed?

## Arms

- `FC1_repaste_real_fullframe`: existing real-context reference.
- `FC2_repaste_procedural_fullframe`: existing procedural-background arm.
- `FC3_repaste_empirical_fullframe`: the FC1 foreground cutout pasted at the
  same bbox on a different single-object validation image.

For FC3, donor and target images have identical canvas dimensions. Donors are
assigned by a deterministic within-shape derangement so each donor is used
once. The donor object is removed by inpainting an expanded bbox before the
target cutout is pasted. Donor identities never enter training or test.

## Frozen evaluation

- A0 seed-42 checkpoint and inference configuration remain unchanged.
- Existing FC1/FC2 reports and raw predictions are reused.
- FC3 uses the same official Ultralytics metrics.
- Corrected class diagnosis chooses the highest-confidence candidate among
  predictions with IoU at least 0.5.
- No training and no test access.

## Recovery and uncertainty

Official mAP recovery is

`(FC3 - FC2) / (FC1 - FC2)`.

The same quantity is reported for corrected macro top-1 accuracy. Paired
FC3-minus-FC2 top-1 deltas use 10,000 class-stratified bootstrap iterations and
an exact two-sided McNemar test.

Empirical background is marked a supported simulator priority only if:

1. official mAP recovery is at least 50%;
2. the bootstrap 95% interval for macro top-1 delta is entirely above zero;
3. McNemar p is below 0.05.

This remains a development diagnostic. It does not establish performance on
independent real dense scenes.
