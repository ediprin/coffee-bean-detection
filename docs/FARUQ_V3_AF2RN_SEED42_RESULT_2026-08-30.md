# Faruq-v3 AF2RN Seed-42 Result — 2026-08-30

## Decision

**FAIL; retain original AF2 and stop AF2RN.** The result is not a threshold
near-miss: AF2RN was lower than the frozen AF2 control on all three headline
metrics and failed every performance gate. Test remained closed.

| Model | Macro mAP50–95 | Bottom-3 | Worst class |
|---|---:|---:|---:|
| AF2C | **88.20%** | **80.04%** | **79.35%** |
| AF2RN | 85.11% | 75.33% | 72.16% |
| AF2RN − AF2C | **−3.09 pp** | **−4.71 pp** | **−7.18 pp** |

The frozen criteria and observed outcomes were:

- Macro gain at least +0.50 point: **FAIL**;
- Bottom-3 not lower: **FAIL**;
- Worst-class drop no more than 1.00 point: **FAIL**;
- all 21 validation classes present: **PASS**;
- test not opened: **PASS**.

## Interpretation

AF2RN tested the hypothesis that the natural radial magnitude profile was a
nuisance that should be divided out before forming AF2's angular density. The
large, consistent degradation rejects that hypothesis for this dataset and
protocol. Absolute and/or cross-radius spectral strength is not disposable
background here; it carries information that the original AF2 angular
statistic uses for fine-grained coffee-defect discrimination. Annulus-median
normalization removed useful signal, with the largest damage appearing in the
lower tail.

This result does not imply that every radial treatment is impossible. It does
show that **per-annulus median normalization before angular accumulation is
incompatible with the successful AF2 statistic under the frozen setup**. The
result therefore closes AF2RN specifically and strengthens the decision to
retain original AF2 rather than continue modifying this operator after seeing
validation.

## Research boundary

- Do not run AF2RN seeds 123 or 2026.
- Do not tune annulus width, normalization statistic, threshold, or gamma from
  this validation result.
- Do not rescue AF2RN through fusion with previous modules.
- Do not open or reuse test for AF2RN.
- Preserve the failure as a negative ablation supporting the final AF2 design.

Frozen protocol:
`docs/FARUQ_V3_AF2_RADIAL_NORMALIZED_ANGULAR_DENSITY_PROTOCOL_2026-08-29.md`.

Machine-readable evidence:
`docs/evidence/FARUQ_V3_AF2RN_SEED42_2026-08-30.json`.
