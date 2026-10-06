# DefectosCafeVerde DVSR1 Validation Result

Completed: **2026-10-06**.

## Decision

**PASS — PROMOTE_DVSR1_SIDE_RELIABILITY.**

The fixed train-only selector was applied once to 376 eligible validation
physical pairs. No selector, detector, or fuser fitting was executed.

| Endpoint | Pair accuracy | Macro | Bottom-3 | Worst |
|---|---:|---:|---:|---:|
| Paper maximum confidence | 94.41% | 92.85% | 80.45% | 76.19% |
| Confirmed DVF1 mean | 95.48% | 93.85% | 83.35% | 76.19% |
| **DVSR1** | **96.28%** | **95.74%** | **89.38%** | **86.36%** |
| Pair oracle | 97.34% | 96.96% | 90.26% | 86.36% |

Against confirmed DVF1, DVSR1 gained 0.80 point physical-pair accuracy, 1.89
points Macro, 6.03 points Bottom-3, and 10.17 points worst-class accuracy. It
rescued nine paper errors and regressed two paper-correct pairs.

The largest class gains over the paper rule were `agrio` (+15.15 points),
`helado` (+14.29), and `partido` (+12.50). `triangulo` declined by 2.86 points
and `broca` by 4.35 points. These validation observations are frozen and must
not be used to retune DVSR1.

Both the overall and lower-tail decision routes passed. Endpoint calibration
was exact, all twelve validation classes were present, and test remained
unopened.

## Artifact

`experiments/defectoscafeverde-dvsr1-v1/DVSR1_validation_result.json`
