# AF2FFAB2 paired three-seed confirmation result

Status: **completed -- PASS; retain AF2FFAB2 as a validated Pareto
refinement. Test remained locked.**

## Frozen comparison

Each seed compares the gradient-matched, bounded spectral adapter
`AF2FFAB2` against the zero-information continuation control `AF2FFA0`.
Both arms start from the same seed-matched original AF2 checkpoint and use
the same 30-epoch continuation schedule. This isolates the spectral feature
signal from continuation optimization.

## Per-seed result

| Seed | Arm | Macro mAP50-95 | Bottom-3 | Worst class |
|---:|---|---:|---:|---:|
| 42 | AF2FFA0 | 88.89% | 80.84% | 77.53% |
| 42 | **AF2FFAB2** | **88.89%** | **82.11%** | **80.49%** |
| 123 | AF2FFA0 | 86.89% | 74.73% | 70.57% |
| 123 | **AF2FFAB2** | **88.41%** | **78.03%** | **71.46%** |
| 2026 | AF2FFA0 | 85.52% | 77.48% | 75.99% |
| 2026 | **AF2FFAB2** | **88.33%** | **81.41%** | **78.95%** |

## Three-seed aggregate

| Metric | AF2FFA0 mean ± SD | AF2FFAB2 mean ± SD | Mean delta | Minimum delta | Improved seeds |
|---|---:|---:|---:|---:|---:|
| Macro | 87.10 ± 1.70% | **88.54 ± 0.30%** | **+1.44 points** | +0.003 | 3/3 |
| Bottom-3 | 77.68 ± 3.06% | **80.52 ± 2.18%** | **+2.83 points** | +1.27 | 3/3 |
| Worst class | 74.69 ± 3.66% | **76.96 ± 4.83%** | **+2.27 points** | +0.89 | 3/3 |

Every frozen criterion passed. The aggregate decision is `PASS`, with next
action `RETAIN_AF2FFAB2_AS_VALIDATED_PARETO_REFINEMENT`. Validation retained
all 21 classes and no test images were opened.

## Interpretation and boundary

The result supports a causal claim that the bounded feature-frequency signal
improves the matched continuation control: all three primary metrics improve
in every seed. It also reduces Macro variation relative to that control.

This confirmation does **not** establish universal superiority over the
pre-continuation original AF2 checkpoint because that is not the frozen causal
comparison. As descriptive context, original AF2 previously averaged 87.94%
Macro, 79.37% Bottom-3, and 78.15% Worst class. AF2FFAB2 is therefore higher
by about +0.60/+1.15 points on Macro/Bottom-3 but lower by about -1.19 points
on Worst class. AF2FFAB2 is the validated Macro/lower-tail Pareto refinement;
original AF2 remains the conservative choice when the single worst class is
the sole objective.

Protocol:
`docs/FARUQ_V3_AF2_FFA_GRADIENT_MATCHED_PAIRED_PROTOCOL_2026-08-22.md`.

Raw evidence:
`docs/evidence/FARUQ_V3_AF2_FFA_B2_PAIRED_CONFIRMATION_2026-08-22.json`.

Drive source:
`experiments/faruq-v3-af2-ffa-gradient-matched-paired-v1/val_reports/af2_ffa_b2_paired_confirmation.json`.
