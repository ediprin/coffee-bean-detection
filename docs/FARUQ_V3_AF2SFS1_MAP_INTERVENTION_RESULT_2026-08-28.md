# Faruq-v3 AF2SFS1 Full-mAP Intervention Result — 2026-08-28

## Decision

The completed seed-42 gain is **optimization-mediated**, not a direct
inference-time selector benefit. `AF2SFS1` normal inference and adapter-bypass
inference are effectively identical, while both retain the detector-weight
change learned during training.

| State | Macro mAP50–95 | Bottom-3 | Worst | Macro AP50 | Macro AP75 |
|---|---:|---:|---:|---:|---:|
| AF2CTRL | 89.00% | 83.88% | 83.55% | 91.31% | 91.31% |
| AF2SFS1 normal | **89.96%** | **84.19%** | 83.25% | 92.18% | 92.14% |
| AF2SFS1 bypass | **89.96%** | **84.19%** | 83.25% | **92.19%** | 92.14% |
| AF2SFS1 spatial only | 89.96% | 84.03% | 82.78% | **92.20%** | **92.16%** |
| AF2SFS1 frequency only | **89.96%** | **84.20%** | **83.28%** | 92.18% | 92.14% |

## Exact decomposition

| Component | Macro | Bottom-3 | Worst | AP50 | AP75 |
|---|---:|---:|---:|---:|---:|
| Total: normal − AF2CTRL | +0.95 | +0.31 | -0.29 | +0.87 | +0.83 |
| Direct selector: normal − bypass | -0.01 | 0.00 | 0.00 | -0.01 | -0.01 |
| Optimization-mediated: bypass − AF2CTRL | **+0.96** | **+0.31** | -0.29 | **+0.88** | **+0.84** |

All values are percentage points. Additivity residual is zero. The large
per-class gains and regressions are likewise retained under adapter bypass;
they are not caused by the active selector at inference.

Checkpoint drift is concentrated more strongly in the feature extractor
(relative L2 0.06224) and regression head (0.07214) than in the classification
head (0.02143). This supports a training-trajectory explanation: the temporary
adapter changes the solution reached by the native AF2 detector.

## Scientific consequence

`AF2SFS1` must not be described as an inference enhancement. It is evidence
for a **training-time structural scaffold / temporary over-parameterization**
direction. A successor experiment may therefore place temporary scaffolds on
P3/P4/P5, but validation and deployment must use the native AF2 path with all
scaffolds bypassed or stripped.

This remains a one-seed mechanistic result. Test was not accessed.

Raw summary:
`experiments/faruq-v3-af2-complement-v1/root_cause/map_intervention/af2sfs1_map_intervention.json`.

Frozen protocol:
`docs/FARUQ_V3_AF2SFS1_MAP_INTERVENTION_PROTOCOL_2026-08-28.md`.
