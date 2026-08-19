# WAV1 mechanism-factorization numerical fix — 2026-08-19

## Trigger

The first Kaggle preflight stopped **before any factorization arm training** because
`tests/test_wav1_factorization.py::test_constant_image_has_zero_effect_for_new_causal_controls`
failed on that runtime. A spatially constant RGB image produced tiny backend-dependent
floating-point residuals in a new causal cue. Plain min-max normalization then amplified
that numerical residue into a non-zero [0,1] gate.

The run therefore produced **no HP1/WAV_L1/WAV_L2/WAV_RAWFUSE validation result** and
created no post-result tuning opportunity.

## Correction

Only the **new causal controls** now use `stable_minmax_spatial`:

- normalization is computed in float32;
- a spatial range within `64 * float32 machine epsilon`, relative to `max(|cue|, 1)`,
  is treated as numerical flatness and mapped exactly to zero;
- otherwise ordinary per-image/per-channel min-max normalization is used.

This threshold is derived from machine precision and was not chosen from validation
performance.

## Frozen reference boundary

`WAV1_REF` is unchanged and still delegates bitwise to the previously confirmed WAV1
implementation. The shared `coffee_detector.afab.operator.minmax_spatial` and
`afab_gate` are also unchanged, so prior experiments are not redefined.

## Scientific scope

The frozen mechanism questions, training schedule, seed-42 screening scope, and locked-test
boundary remain unchanged. This is a pre-training numerical correctness fix, not a new arm,
new hyperparameter search, or post-result method revision.
