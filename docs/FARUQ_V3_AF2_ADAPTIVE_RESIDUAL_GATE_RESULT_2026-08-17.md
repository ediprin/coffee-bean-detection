# Faruq-v3 Raw-Preserving Adaptive AF2 Result

Date: 2026-08-17  
Decision: **FAIL -- stop without illumination screening, extra seeds, or test**

## Protocol and execution

The prospectively frozen protocol is
`docs/FARUQ_V3_AF2_ADAPTIVE_RESIDUAL_GATE_PROTOCOL.md`. The static gate passed
before training: both arms had the same architecture, state schema, schedule,
and 2,512,457 parameters; the gate added 467 parameters. Both arms started at
the fixed AF2 output. `AF2R0` received zero conditioning information and
`AF2R1` received the six illumination/recovery maps.

Both arms completed 30 requested continuation epochs on grouped Faruq-v3
development data using seed 42. Validation contained all 21 classes. Neither
arm accessed test images.

## Clean-validation results

| Model | Role | Macro mAP50-95 | Bottom-3 | Worst class |
|---|---|---:|---:|---:|
| Frozen AF2 | pre-study reference | 88.20% | 80.04% | 79.35% |
| `AF2R0` | equal-parameter zero-information control | **89.55%** | **84.30%** | **83.97%** |
| `AF2R1` | illumination-conditioned candidate | 88.93% | 83.16% | 82.57% |

`AF2R1 - AF2R0` was -0.62 Macro, -1.13 Bottom-3, and -1.40 Worst-class
points. `AF2R1` remained above the original frozen AF2 by +0.73, +3.12, and
+3.22 points respectively, but the matched control improved even more.

## Frozen gate

Only the three fixed-AF2 preservation criteria passed. All three causal
criteria against the matched control failed:

- Macro did not gain at least 0.5 point over `AF2R0`;
- Bottom-3 was lower than `AF2R0`;
- Worst-class AP dropped more than one point from `AF2R0`.

The study therefore stops at seed 42. The paired illumination screen, seeds
123/2026, and test access are not authorized.

## Interpretation and artifact status

Continued matched fine-tuning from AF2 improved the clean-validation metrics,
but supplying illumination/recovery information to the adaptive gate harmed
all three primary metrics relative to the zero-information control. The result
rejects the proposed conditioning mechanism; it does not reject fixed AF2 or
its previously confirmed in-domain and Coffee Standard results.

The Kaggle logs show that `best.pt`, `last.pt`, and both arm result JSON files
were written. The interactive Kaggle session was subsequently stopped before
an output version was persisted, so this record does not claim that those
checkpoint files remain retrievable. Exact final metrics and per-class AP were
recovered from the completed logs and frozen in:
`docs/evidence/FARUQ_V3_AF2R_SCREENING_2026-08-17.json`.

