# Faruq-v3 AF2 Channel-Calibration Result

Date: 2026-08-17  
Decision: **FAIL -- stop without extra seeds or test**

## Protocol and execution

The prospectively frozen protocol is
`docs/FARUQ_V3_AF2_CHANNEL_CALIBRATION_PROTOCOL.md`. The static audit passed
before training. `AF2FT30` continued the completed AF2 seed-42 checkpoint for
30 epochs without added parameters. `AF2CAL3` used the identical checkpoint,
data, schedule, and detector, but added exactly three trainable RGB residual
scales initialized to reproduce AF2 exactly.

Both arms completed on grouped Faruq-v3 development data. Validation contained
all 21 classes. Training was executed; test images were not accessed.

## Validation results

| Model | Role | Macro mAP50-95 | Bottom-3 | Worst class |
|---|---|---:|---:|---:|
| Frozen AF2 | pre-study reference | 88.20% | 80.04% | 79.35% |
| AF2R0 | prior zero-information reference | **89.55%** | **84.30%** | **83.97%** |
| `AF2FT30` | matched continuation control | 89.00% | 83.88% | 83.55% |
| `AF2CAL3` | three-scalar candidate | 88.77% | 83.72% | 83.00% |

`AF2CAL3 - AF2FT30` was -0.23 Macro, -0.17 Bottom-3, and -0.55
Worst-class points. The candidate also remained 0.78 Macro, 0.58 Bottom-3,
and 0.97 Worst-class points below AF2R0.

Both continuation arms improved over frozen AF2, but that comparison does not
isolate the calibration mechanism. Against the matched continuation control,
the three-scalar calibrator improved none of the three primary metrics.

## Frozen gate

The candidate failed four of six criteria:

- no Macro gain of at least 0.5 point over `AF2FT30`;
- Bottom-3 was lower than `AF2FT30`;
- Macro was not within 0.5 point of AF2R0;
- the complete causal decision therefore failed.

Worst-class degradation relative to `AF2FT30` stayed within the one-point
tolerance, and Bottom-3/Worst remained within one point of AF2R0. These partial
passes do not override the prospectively frozen joint gate.

## Interpretation

The AF2R0 improvement is **not explained by input-independent per-channel
residual scaling**. Continued optimization itself accounts for much of the
gain over frozen AF2, while the additional three-channel calibration slightly
hurts the matched metrics. AF2CAL3 is closed at seed 42; seeds 123/2026 and test
evaluation are not authorized.

This negative result does not invalidate fixed AF2, whose earlier paired
in-domain and Coffee Standard evidence remains unchanged. The downloaded ZIP
contains both best/last checkpoints, logs, validation reports, and the decision
report, but model artifacts are not committed to Git.

Frozen evidence:
`docs/evidence/FARUQ_V3_AF2_CHANNEL_CALIBRATION_SCREENING_2026-08-17.json`.
