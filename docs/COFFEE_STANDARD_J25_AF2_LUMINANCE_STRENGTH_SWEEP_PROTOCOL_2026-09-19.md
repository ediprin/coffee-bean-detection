# Coffee Standard J25 AF2LUMSAFE Inference-Strength Diagnostic

Status: **frozen before diagnostic evaluation**

Date: 2026-09-19

## Motivation

The fresh seed-42 `AF2LUMSAFE` screen improved Macro mAP50-95 from 60.54% to
65.89% and Bottom-3 from 19.55% to 25.70% relative to `D0DIRECT`, but
`Biji Hitam Pecah` decreased from an already-low 1.90% AP to 0.00%. During
training, `AF2LUMSAFE` sampled AF2 strength uniformly between raw RGB and full
luminance AF2. Validation used only full strength. This diagnostic tests
whether the isolated failure is caused by that train-inference strength
mismatch.

## Frozen evaluation

The completed `AF2LUMSAFE_seed42` checkpoint is not modified or retrained. It
is evaluated on the unchanged 68-image, 25-class validation partition at:

```text
lambda = 0.00, 0.25, 0.50, 0.75, 1.00
```

The model input remains:

```text
x_lambda = x * (1 + lambda * G_Y)
```

where `G_Y` is the unchanged shared Rec.709 luminance AF2 gate. `lambda=0`
disables the frontend at inference without changing trained detector weights;
`lambda=1` must reproduce the completed historical endpoint within numerical
tolerance.

For every strength, the report includes Macro, Bottom-3, Worst-class, all
25 class AP values, and specifically `Biji Hitam Pecah` AP. The summary reports
the metric winners and the non-dominated strength frontier. It does not select
a deployable operating point or authorize a test claim.

## Claim boundary

- validation only;
- no optimization step and no gradient update;
- no checkpoint mutation;
- no locked-test extraction or model evaluation;
- any favorable strength is post-screen validation evidence and requires a
  separately frozen confirmation before a final claim;
- failure to rescue `Biji Hitam Pecah` attributes the zero-AP result away from
  AF2 inference strength and toward class support, confusion, or ontology.

Output root:
`experiments/coffee-standard-j25-af2-luminance-strength-sweep-v1`.
