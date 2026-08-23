# Faruq-v3 AF2_ORIENT + CMC0 Composition Screening Protocol

Date: 2026-08-23
Status: **FROZEN AFTER IMPLEMENTATION REPAIR — NO RESULT YET**

## Question

Does adding the non-spatial classification-capacity control `CMC0` to the
already frozen `AF2_ORIENT` input transform improve the seed-42 grouped
validation result without sacrificing the difficult-class tail?

This is an **exploratory composition screen**, not a confirmed-mechanism claim.
Two prior facts must remain explicit:

- `AF2_ORIENT` improved Macro in 2/3 paired seeds but failed its frozen
  lower-tail confirmation gate; original AF2 remains the retained spectral
  frontend.
- `CMC0` explained much of the seed-42 STB gain, but the paired STB-vs-CMC0
  causal confirmation also failed its frozen Macro gate. CMC0 is therefore a
  strong capacity control, not an independently confirmed thesis method.

The purpose of this branch is only to test whether those two frozen components
are compatible when composed; a positive seed-42 result would still require
seed-matched confirmation.

## Architecture

```text
RGB image
  -> AF2_ORIENT (unsigned 180-degree angular folding)
  -> YOLO26n backbone/neck
  -> native localization path
  -> CMC0 zero-gated, non-spatial channel-mixing blocks on classification path
  -> native YOLO26 output
```

`AF2_ORIENT` remains parameter-free. `CMC0` is inserted only in the
classification path; localization uses the original feature tensors.

## Frozen configuration

- detector: `configs/coffee_fg/models/yolo26n-p3.yaml`
- classes: 21
- seed: 42 only for screening
- training schedule: 50 epochs, image size 640, batch 16, patience 15
- AF2_ORIENT: patch 32, stride 16, gamma 0.1, 180 bins, 180-degree period,
  one radial band
- CMC0: two token-wise channel-mixing blocks per P3/P4/P5 classification level,
  identity residual gate initialized to zero
- evaluation: grouped Faruq-v3 validation only
- test split: unavailable / forbidden

## Initialization and matched parent

The candidate **must start from the same seed-42 D0 checkpoint** used by the
frozen `AF2_ORIENT` seed-42 run. Generic `yolo26n.pt` is not an authorized
scientific initialization for this experiment.

The matched parent is the existing machine-readable result:

`experiments/faruq-v3-af2-isolated-seed42-v1/val_reports/AF2_ORIENT_seed42_result.json`

The runner verifies that the parent result and candidate use the same D0
checkpoint SHA-256, seed, operator definition, validation split, and test lock.

## Static gate before training

Training is blocked unless `static_audit.json` is `PASS`. The audit checks:

1. AF2_ORIENT actually changes the input tensor.
2. All CMC0 residual gates initialize at exactly zero.
3. At zero CMC0 gate, the combined model reproduces AF2_ORIENT localization
   and classification tensors from the same D0 checkpoint with full-model
   numerical tolerance `max_abs_diff <= 1e-4` (the established AF2 FFT/IFFT
   composition-audit convention).
4. Activating CMC0 changes classification scores while preserving localization
   tensors within the same `1e-4` tolerance.
5. Three P3/P4/P5 classification levels are wrapped.
6. No test data is accessed.

This establishes the relevant starting-function identity **conditional on
AF2_ORIENT** rather than comparing unrelated full models.

## Seed-42 decision gate

Primary comparison:

`AF2_ORIENT_CMC0 - AF2_ORIENT`

The candidate passes through either of two pre-frozen routes.

### Route A — superiority

- Macro gain >= **+0.20 percentage point**
- Bottom-3 drop <= **0.50 point**
- Worst-class drop <= **1.00 point**

### Route B — tail Pareto

- Macro drop <= **0.10 point**
- Bottom-3 gain >= **+0.50 point**
- Worst-class gain >= **+1.00 point**

All 21 validation classes must be present. No hyperparameter tuning is allowed
after this screen.

## Decision

- `PASS` -> authorize seed-matched 123/2026 confirmation of the composition.
- `FAIL` -> stop this composition; do not tune AF2 orientation, CMC0 depth,
  gate initialization, or training schedule on the same validation evidence.

A `PASS` would mean only that the **composition is worth confirmation**. It
would not retroactively validate AF2_ORIENT as superior to original AF2, nor
validate the STB spatial-causal claim.
