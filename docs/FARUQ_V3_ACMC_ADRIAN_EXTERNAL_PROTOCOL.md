# Faruq-v3 ACMC Adrian External-Source Protocol

Status: frozen post hoc after the Faruq-v3 locked test, 2026-08-12.

## Question

Does the directional advantage of ACMC1 over its optimization-matched D0FT
control persist on real Adrian validation images that were not used to train
the Faruq-v3 models?

This is a secondary external-source evaluation. It is not a second final test,
cannot change the Faruq locked-test `NOT_CONFIRMED` conclusion, and cannot
authorize model tuning.

## Frozen models

- Paired seeds: 42, 123, and 2026.
- Baseline: the three completed D0FT checkpoints.
- Candidate: the three completed ACMC1 checkpoints.
- Every checkpoint SHA-256 must exactly match the hashes stored in the final
  Faruq-v3 locked-test summary.
- No retraining, fine-tuning, checkpoint selection, threshold search, or
  architecture change is permitted.

## External data

- Source: `adrian_detection` validation identities from the audited combined
  A0 archive.
- Test is not extracted or read.
- Faruq validation identities in the A0 archive are discarded.
- Canonical SNI-21 class IDs are retained without remapping.
- Classes without Adrian validation ground truth are reported and excluded
  from macro/lower-tail aggregation. They are not assigned artificial zero AP.
- Adrian parent IDs, Roboflow derivative IDs, and cross-manifest hashes must
  have zero overlap with the Faruq-v3 training manifest.

The source tag establishes a different Roboflow project/domain. The overlap
audit is an additional safeguard, not proof that Adrian is representative of
all future coffee imagery.

## Evaluation

- Real validation images only; no synthetic scenes.
- Resolution: 640.
- Confidence floor: 0.001.
- NMS IoU: 0.7.
- `max_det`: 500.
- Metrics: supported-class macro, bottom-three, and worst-class mAP50-95.
- Report every paired seed delta plus mean and sample standard deviation.

Directional status is `SUPPORTS_EXTERNAL_DIRECTION` only if all descriptive
gates pass:

1. mean macro delta is positive;
2. macro improves on at least two of three seed pairs;
3. mean bottom-three delta is non-negative;
4. mean worst-class delta is no lower than -1 percentage point.

Otherwise the status is `DOES_NOT_SUPPORT_EXTERNAL_DIRECTION`. Neither status
changes the frozen locked-test decision or permits further tuning.

## Claim boundary

Adrian validation contains correlated Roboflow derivatives from a limited
number of source parents. Therefore the result is reported as post-hoc
external-source directional evidence, not independent population-level proof
of ACMC superiority.

Runnable notebook:
[`notebooks/Faruq_V3_ACMC_Adrian_External_Colab.ipynb`](../notebooks/Faruq_V3_ACMC_Adrian_External_Colab.ipynb).
