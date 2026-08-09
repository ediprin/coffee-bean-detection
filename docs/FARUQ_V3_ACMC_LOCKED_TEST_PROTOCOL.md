# Faruq-v3 ACMC Single Locked-Test Protocol

Status: frozen after the paired three-seed validation PASS and before any ACMC
test inference, 2026-08-09.

## Purpose

Estimate the final generalization difference between optimization-matched
`D0FT` and `ACMC1`.  No architecture, hyperparameter, threshold, checkpoint,
or acceptance rule may change after the test integrity audit starts.

## Why the Roboflow test cannot be used directly

The original Faruq archive contains parent identities across its nominal
train/valid/test folders.  Earlier repository audit found 48 train--test and 9
valid--test parent overlaps.  Therefore the archive split is not the test
authority.

## Frozen test construction

1. Open only the original Faruq Roboflow v1 test folder.
2. Compare raw SHA-256 and canonical Roboflow parent identity against every
   train/validation row in `faruq_grouped_manifest.json`.
3. Exclude every overlapping parent or exact hash.
4. Keep one lexicographically selected image per remaining test parent to avoid
   pseudo-replication by augmented siblings.
5. Recompute boxes from polygon masks and apply the same frozen orientation
   rule used for Faruq-v3 development: score long side 192 and minimum
   alignment improvement 0.02.
6. Do not inspect model predictions during construction.

The test is eligible only if all gates pass:

- zero development parent/hash overlap;
- one selected image per independent test parent;
- at least 50 independent test images;
- all 21 classes present;
- at least 10 instances and 5 independent parents per class;
- no selected image quarantined for invalid geometry/annotation.

Failure stops inference.  The permitted alternatives are a newly collected
external test or grouped cross-validation; thresholds may not be weakened after
seeing the audit.

## Frozen checkpoints

Seeds are `42`, `123`, and `2026`.  For every seed, evaluate the already
completed paired `D0FT` and `ACMC1` checkpoints.  The test runner records hashes
for the test manifest and all six checkpoints and refuses incompatible cached
reports.

## Metrics and final interpretation

Report per seed and mean +/- sample standard deviation for macro mAP50-95,
bottom-three class mAP50-95, and worst-class mAP50-95, plus all class AP values.
The final ACMC result is `CONFIRMED` only when:

- mean macro delta is positive and positive on at least two of three seeds;
- mean bottom-three delta is non-negative;
- mean worst-class delta is not below -1 point.

Regardless of the outcome, the test is opened once, further tuning is
forbidden, and the result must be reported.
