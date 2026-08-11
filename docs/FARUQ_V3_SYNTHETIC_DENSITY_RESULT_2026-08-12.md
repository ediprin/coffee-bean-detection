# Faruq-v3 Synthetic Density Diagnostic Result

Date: 2026-08-12  
Status: complete, development-only, no training and no test access.

## Frozen comparison

- Models: frozen D0FT and ACMC1 seed-42 checkpoints.
- Source: Faruq-v3 grouped validation parents only.
- Cutouts: audited repaired COCO polygons.
- Prior: balanced diagnostic.
- Visibility: mild.
- Scenes: 100 per condition.
- Evaluation: 640 pixels, `max_det=500`.

| Condition | Objects | D0FT Macro mAP50-95 | ACMC1 Macro mAP50-95 | Delta | D0FT Bottom-3 | ACMC1 Bottom-3 | Delta | D0FT Worst | ACMC1 Worst | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 1-5 | 4.12% | 5.55% | +1.44 pp | 0.16% | 0.26% | +0.09 pp | 0.00% | 0.00% | +0.00 pp |
| B1 | 10-25 | 8.07% | 8.09% | +0.02 pp | 1.15% | 1.69% | +0.54 pp | 0.45% | 1.14% | +0.69 pp |
| B2 | 50-100 | 3.42% | 3.79% | +0.38 pp | 0.26% | 0.38% | +0.13 pp | 0.00% | 0.06% | +0.06 pp |
| B3 | 220-300 | 1.28% | 1.39% | +0.12 pp | 0.03% | 0.03% | +0.01 pp | 0.00% | 0.00% | +0.00 pp |

Mean unweighted macro delta is +0.49 percentage point. The macro delta is
positive in 4/4 conditions, with a minimum of +0.02 point.

## Interpretation

ACMC1 never falls below D0FT in the four paired synthetic conditions. This is
directionally consistent with the validation and locked-test comparisons, but
the density advantage is small after B0 and does not establish density
robustness. Absolute mAP collapses for both models, including in B0, and the
ordering B1 > B0 is not a monotonic density response. Synthetic composition
domain shift therefore dominates this benchmark.

This result may be reported only as a secondary stress-test observation:

- ACMC1's relative direction does not reverse across the synthetic ladder;
- the advantage attenuates and is not evidence of operational dense-scene
  performance;
- the benchmark is validation-correlated, seed-42 only, and has no independent
  real-dense reference;
- the locked-test conclusion remains `NOT_CONFIRMED`;
- no further tuning or test access is authorized from this result.

Raw report:
`experiments/faruq-v3-synthetic-density-v1/synthetic_density_seed42_summary.json`
in the shared Drive artifact root.

