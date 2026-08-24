# Faruq-v3 Multilevel Fusion CM512 Protocol

Version: `v1.0.0`

Frozen: 2026-08-02, before PCA-512 fitting

## Question

At the native 512-dimensional capacity of the strongest single-level P5
descriptor, does P3+P4+P5 retain a validation advantage when both
representations use the identical train-only PCA and ridge-probe pipeline?

This is one preregistered capacity control, not a rank sweep. The value 512 is
fixed because it equals the native P5 descriptor dimension.

## Fixed setup

- Input artifacts: the completed Faruq-v3 predicted-ROI transfer report and its
  immutable train/validation feature caches.
- Checkpoint identity: must match the D0 SHA-256 stored in the transfer report.
- Regions: matched D0 predicted boxes already frozen by the preceding audit.
- Representations:
  - P5 mean+maximum descriptor, native 512 dimensions;
  - P3+P4+P5 mean+maximum descriptor, native 896 dimensions.
- Capacity transform: independent train-only PCA to exactly 512 dimensions for
  both representations.
- Probe: identical class-balanced ridge least squares, regularization 0.01.
- No detector inference, detector training, validation tuning, rank sweep, or
  test access.

## Required metrics

- train and validation Macro-F1;
- validation balanced accuracy, bottom-3 F1, worst F1, and top-3 accuracy;
- train-to-validation Macro-F1 gap;
- fusion deltas against P5 for Macro-F1, bottom-3, worst-class, and top-3;
- per-class validation F1 for the fusion.

## Frozen gate

Return `AUTHORIZE_MULTILEVEL_HEAD_PROTOCOL` only when all are true:

1. fusion validation Macro-F1 improves by at least two percentage points;
2. fusion bottom-3 F1 does not decrease;
3. fusion worst-class F1 does not decrease by more than one point;
4. fusion top-3 accuracy does not decrease;
5. fusion reaches at least 75% validation Macro-F1; and
6. fusion reaches at least 50% validation bottom-3 F1.

Passing authorizes writing a capacity-matched detector-head protocol and static
architecture audit only. It does not authorize training, more seeds, or test
access. Failure stops this multilevel-head route.
