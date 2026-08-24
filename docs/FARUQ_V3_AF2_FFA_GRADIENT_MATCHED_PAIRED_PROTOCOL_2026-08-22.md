# AF2FFAB2 paired three-seed confirmation protocol

Status: frozen before seed 123/2026 confirmation training.

## Research question

Does the gradient-matched ±10% frequency adapter provide a repeatable
lower-tail benefit beyond an optimization- and capacity-matched continuation
control?

## Paired design

Seeds are 42, 123, and 2026. Each seed compares:

- `AF2FFA0`: zero-information adapter continuation control;
- `AF2FFAB2`: gradient-matched bounded spectral adapter.

For seeds 123 and 2026, both arms start from the corresponding completed AF2
checkpoint, use the same Faruq-v3 grouped development split and frozen 30-epoch
schedule, and differ only in whether the adapter receives the fixed spectral
descriptor. The seed-42 reports are reused. Four new training runs are
authorized: two arms for each of the two new seeds.

This pairing is mandatory. Comparing AF2FFAB2 after continuation directly to
the pre-continuation AF2 checkpoint would confound the spectral mechanism with
the extra optimization schedule.

## Aggregate gate

AF2FFAB2 passes as a validated Pareto refinement only if all conditions hold:

1. mean Macro loss versus AF2FFA0 is no more than 0.1 percentage point;
2. Macro is non-inferior within 0.1 point on at least two of three seeds;
3. mean Bottom-3 delta is positive and improves on at least two seeds;
4. mean Worst-class delta is positive and improves on at least two seeds.

The gate tests repeatable tail improvement while preserving global accuracy;
it does not require an arbitrary +0.5 Macro gain because the seed-42 candidate
was explicitly selected as a Pareto, not Macro-only, refinement.

## Locks

- Seed 123/2026 training requires the immutable seed-42 `RETAIN_PARETO`
  decision.
- Every arm requires a static audit tied by SHA256 to its seed-matched AF2
  checkpoint.
- Validation must retain all 21 classes.
- Test is not restored, read, or authorized.
- No hyperparameter, schedule, threshold, or model change is allowed after
  observing either new seed.

