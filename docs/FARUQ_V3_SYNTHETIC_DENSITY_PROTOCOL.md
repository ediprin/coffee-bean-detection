# Faruq-v3 Synthetic Density Diagnostic Protocol

Status: frozen after the ACMC locked test, 2026-08-11.

## Purpose

This is a secondary development diagnostic. It asks whether the frozen ACMC1
head retains an advantage over optimization-matched D0FT as synthetic scene
density increases. It cannot change the locked-test `NOT_CONFIRMED` result and
cannot authorize model tuning.

## Why the old B0--B3 artifacts are not reused

The old density library followed an earlier combined-dataset split. Auditing it
against the later Faruq-v3 grouped split found Faruq training-parent overlap in
every arm: 41 parents in B0, 112 in B1, 132 in B2, and 138 in B3. Evaluating
Faruq-v3 checkpoints on those artifacts would be identity-leaked.

## Safe regenerated benchmark

- Source: Faruq-v3 grouped **validation** images only.
- Train/validation parent and exact-hash separation must already pass the
  grouped dataset gate.
- Test is unavailable and is never accessed.
- Parent authority is taken from the Faruq-v3 grouped validation manifest.
- Cutout masks come from the audited/repaired Faruq-v2 COCO polygons for those
  exact parents. Bounding-box foreground estimation is not used.
- Classes: balanced diagnostic prior across all 21 SNI classes.
- Visibility: mild.
- Canvas: 1024 pixels.
- Object scale: fixed 0.025--0.055 long-side fraction.
- Scenes: 100 per condition.
- Density ladder: B0 1--5, B1 10--25, B2 50--100, B3 220--300 objects.

Because these validation identities participated in checkpoint selection, the
benchmark is development-correlated. It is not an independent test and cannot
support a new generalization claim.

## Frozen screening

- Seed: 42 only.
- Models: frozen D0FT and ACMC1 seed-42 checkpoints.
- No training or checkpoint modification.
- Evaluation: official Ultralytics validation at 640 pixels, max_det 500.
- Metrics: macro, bottom-three, and worst-class mAP50-95 plus paired deltas at
  each density.

The result is reported as a density trend. No PASS result opens additional
test access, and no FAIL result permits tuning ACMC after the locked test.
