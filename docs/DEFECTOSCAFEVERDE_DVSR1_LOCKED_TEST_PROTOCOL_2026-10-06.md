# DefectosCafeVerde DVSR1 Locked-Test Protocol

Frozen: **2026-10-06**, after DVSR1 passed its one authorized validation screen
and before any grouped test image is extracted or accessed.

## Purpose

Evaluate final generalization of the already-selected DVSR1 side-reliability
method. Test is for final reporting only and cannot select, tune, or repair a
method.

## Frozen inputs

- the original grouped-physical-v1 archive and manifest;
- the immutable seed-42 D0 detector checkpoint;
- the passed DVSR train-only audit and DVSR1 validation result;
- the three confirmed DVF1 fuser checkpoints (42, 123, and 2026);
- unchanged paper maximum-confidence, DVSR1, and pair-oracle definitions.

Only the grouped `test` split is extracted into the runtime directory. Train
and validation images are forbidden in that directory. The D0 detector builds
one immutable paired test cache; all later endpoints reuse that cache.

## Endpoints and metrics

- paper maximum confidence;
- DVF1 seeds 42, 123, and 2026 plus their mean;
- final deterministic DVSR1;
- non-deployable pair oracle.

Report physical-pair accuracy, Macro class accuracy, Bottom-3 class accuracy,
worst-class accuracy, all classwise accuracies, DVSR1 transitions against the
paper rule, and paired test population statistics.

## Frozen final-confirmation criteria

DVSR1 generalization is confirmed only if all conditions hold:

1. physical-pair accuracy improves over paper;
2. Macro and Bottom-3 are not lower than paper;
3. worst-class accuracy drops by no more than 2 points versus paper;
4. physical-pair accuracy and Macro are not lower than the three-seed DVF1
   test mean; and
5. every provenance, checkpoint, ontology, and no-training gate passes.

Regardless of outcome, next is `REPORT_LOCKED_TEST_AND_STOP`. No additional
feature, threshold, seed, or architecture search is authorized from test.

## Access and claim boundary

- test access is explicitly authorized exactly once by this frozen protocol;
- detector, fuser, and selector fitting are forbidden;
- validation re-access is forbidden;
- pair oracle remains diagnostic and non-deployable;
- claims apply to grouped physical beans with two eligible views.
