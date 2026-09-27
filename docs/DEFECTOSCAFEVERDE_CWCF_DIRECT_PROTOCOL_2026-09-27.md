# DefectosCafeVerde Grouped - DCWCF1 Direct Protocol

Status: **frozen before training**. Date: 2026-09-27.

## Research question

Does the classification-only chromatic-wavelet mechanism retained on J25
transfer to the independent, paper-backed DefectosCafeVerde dataset when
physical-bean identity, detector initialization, training schedule, and
validation endpoint are controlled?

## Evidence motivating the transfer

The DefectosCafeVerde paper reports that its weakest classes are broken, ear,
black, sour, and shell. It specifically attributes residual confusion to:

- black versus broca through partial dark coloration;
- sour versus frozen/normal through color gradients;
- broken versus ear/peaberry through morphological overlap.

These are the two signal families targeted by CWCF1: illumination-reduced
chromatic cues and two-level Haar detail cues. The paper motivates the transfer
but does not guarantee a gain.

## Dataset and split

- Dataset: `defectoscafeverde-grouped-physical-v1`.
- Original images: 4,038; inferred physical groups: 2,069.
- Development data: 2,827 train and 808 validation images.
- Test: 403 images, never extracted into the runtime development root.
- Two sides of one inferred physical bean remain in the same split.
- No generated Roboflow augmentation is used.

The physical grouping remains explicitly inferred from the paper's dual-sided
acquisition description, consecutive filenames, and the frozen visual review.

## Matched arms

| Arm | Detector | Classification input | Auxiliary supervision |
|---|---|---|---|
| `D0DIRECT` | YOLO26n P3-P5 | native RGB | none |
| `DCWCF1` | identical YOLO26n P3-P5 | RGB features conditioned by Cb, Cr, Haar-L1, Haar-L2 | fixed paper-grounded attributes |

Both arms start fresh from the same official `yolo26n.pt`, seed 42, and exact
50-epoch schedule. DCWCF1 leaves every box branch on native RGB features. Its
attribute outputs are training-only and are never composed into inference
scores.

The three shared semantic families are frozen from the paper's reported
confusions: sour/frozen/normal color, black/broca dark, and
broken/ear/peaberry shape. Atomic factors keep all twelve expert labels unique.
No J25 label or checkpoint is used.

## Static gates

- official pretrained SHA is exact;
- target ontology is exactly the 12 frozen Spanish labels;
- train schedules and common initialized native tensors match;
- candidate starts with boxes and scores bitwise equal to D0DIRECT;
- activated cue changes scores but preserves boxes;
- the paper-grounded attribute matrix uniquely encodes all 12 classes;
- auxiliary gradients are finite and reach the classification adapter;
- test is absent and inaccessible.

## Seed-42 decision

Promotion requires either:

1. **Overall:** Macro gain at least +0.5 point, Bottom-3 not lower, and Worst
   drop no greater than 1 point; or
2. **Lower-tail:** Macro drop no greater than 0.2 point, Bottom-3 gain at least
   +1 point, and Worst gain at least +1 point.

If neither route passes, DCWCF1 stops after seed 42. A pass authorizes paired
seeds 123 and 2026, not test evaluation. Only a later paired confirmation may
authorize one final locked-test run.

## Claim boundary

This evaluates transfer of the mechanism to an independent SCA-style
12-class dataset. It is not an SNI evaluation and does not repair the inferred
physical-identity limitation.
