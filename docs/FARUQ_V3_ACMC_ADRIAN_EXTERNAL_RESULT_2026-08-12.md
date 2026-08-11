# Faruq-v3 ACMC Adrian External-Source Result — 2026-08-12

Status: complete, `DOES_NOT_SUPPORT_EXTERNAL_DIRECTION`.

## Frozen evaluation

- Models: the three D0FT and three ACMC1 checkpoints already frozen by the
  completed Faruq-v3 locked test.
- Seeds: 42, 123, and 2026.
- Data: Adrian-only real validation subset from the combined A0 archive.
- Evaluation only: no training, test access, checkpoint selection, or tuning.
- Checkpoint hashes matched the final locked-test summary.

## Dataset and independence audit

- Images: 158.
- Boxes: 4,462 (28.24 boxes/image).
- Canonical source parents: 8.
- Ground-truth classes: 20 of 21.
- Missing class: `kulit_tanduk_ukuran_kecil`.
- Zero Faruq-train overlap by canonical parent ID, Roboflow derivative ID, and
  cross-manifest image hash.
- Test was neither materialized nor accessed.

The eight-parent support makes this a correlated post-hoc development result,
not an independent population-level test.

## Aggregate result

| Metric | D0FT mean ± SD | ACMC1 mean ± SD | ACMC1 − D0FT | Minimum paired delta | Improved seeds |
|---|---:|---:|---:|---:|---:|
| Macro mAP50-95 | 3.63% ± 0.45% | 3.33% ± 0.41% | **−0.30% ± 0.05%** | −0.35% | 0/3 |
| Bottom-3 mAP50-95 | 0.00% ± 0.00% | ≈0.00% ± ≈0.00% | ≈+0.00% | 0.00% | 2/3* |
| Worst-class mAP50-95 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% | 0.00% | 0/3 |

`*` The two nominal bottom-three improvements are only floating-point-scale
changes (0.00073 and 0.00689 percentage points) while both models remain at
the zero floor. They are not practically meaningful improvements.

Macro deltas by seed were:

- seed 42: −0.280 percentage point;
- seed 123: −0.351 percentage point;
- seed 2026: −0.265 percentage point.

## Decision

ACMC1 was lower than D0FT on the primary external-source metric for all three
seed pairs. The frozen external-direction gate therefore failed. The result
does **not** support a claim that ACMC generalizes from Faruq to Adrian.

## What the result means

The dominant observation is not the small −0.30-point head delta but the
absolute collapse of both models to approximately 3--4% macro mAP50-95. This
indicates severe source/domain mismatch:

- Faruq-v3 development images are sparse (about two objects/image), whereas
  this Adrian subset averages 28.24 objects/image;
- the two sources use different acquisition and annotation pipelines;
- object scale, background, density, and class appearance therefore change
  together.

This result does not by itself prove which of those factors caused the
collapse. It also does not establish that Adrian labels are unusable. An older
combined-source A0 checkpoint reached 40.65% macro mAP50-95 on the same Adrian
validation subset after being trained with Adrian-source data. That comparison
uses a different training distribution and is evidence of domain dependence,
not a controlled architecture comparison.

## Claim boundary

- The Faruq locked-test conclusion remains `NOT_CONFIRMED`.
- The synthetic-density diagnostic remains development-only.
- No cross-source robustness or domain-generalization claim is supported.
- This post-test evaluation does not authorize any additional tuning of the
  frozen ACMC models.

Raw authoritative summary:
`Coffee_Bean_Detection/experiments/faruq-v3-acmc-adrian-external-v1/adrian_external_summary.json`
in the shared Google Drive project.
