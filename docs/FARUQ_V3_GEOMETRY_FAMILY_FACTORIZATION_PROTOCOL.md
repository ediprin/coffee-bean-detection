# Faruq-v3 exact-capacity geometry family-factorization protocol

Status: **frozen before GEO-SHARED60 / GEO-FAM35x3 training results are read**.

## Scientific status

The completed GEO1 three-seed validation confirmed that predicted-box geometry can improve YOLO26 classification relative to an equal-parameter zero-information control. A posthoc decomposition then showed reproducible family heterogeneity: the mean GEO1 effect was negative for `kulit_kopi` in 3/3 seeds and positive for `kulit_tanduk` in 3/3 seeds.

Because that decomposition used the same development validation data and seeds 42/123/2026, this experiment is an **exploratory architecture validation**, not an independent confirmation. It must not be described as new held-out evidence and it does not authorize locked-test access by itself.

## Question

Does separating the geometry-to-logit mapping by object family improve over a single shared geometry mapping when the two candidates have exactly the same added trainable parameter count?

## Frozen arms

Both arms start from the exact seed-specific native D0 checkpoint and use the same predicted-box geometry:

`g = [w_norm, h_norm, area_norm, aspect_ratio]`

Raw predicted box outputs are detached before decode. No GT geometry or GT family label is model input. The native YOLO box tensor is returned unchanged.

Only the nine size-defined logits are modified:

- `kulit_kopi`: kecil / sedang / besar
- `kulit_tanduk`: kecil / sedang / besar
- `tanah_batu_ranting`: kecil / sedang / besar

### GEO-SHARED60

One shared adapter:

`4 -> 60 -> 9`

Parameter count:

`(4*60 + 60) + (60*9 + 9) = 849`.

### GEO-FAM35x3

Three independent family adapters, each:

`4 -> 35 -> 3`

Parameter count per family:

`(4*35 + 35) + (35*3 + 3) = 283`.

Total:

`3 * 283 = 849`.

Thus both candidates add **exactly 849 trainable parameters**. Matching parameter count does not imply functional equivalence; it isolates shared-vs-family-factorized mapping more cleanly than comparing to GEO1's original 853-parameter adapter.

All final projections are zero initialized, so both arms begin from the same native D0 function.

## Frozen training

Seeds: `42, 123, 2026`.

For every seed both candidates use:

- the same D0 checkpoint;
- 50 continuation epochs;
- YOLO26 P3/P4/P5 architecture unchanged outside the classification residual;
- image size 640;
- batch 16;
- optimizer `auto`;
- `close_mosaic=10`;
- deterministic seed-specific training;
- the same Faruq-v3 development train/val split;
- no locked test.

No family routing network, no family GT, no class-specific hand weights, no stacking, and no additional tuning are allowed.

## Primary metrics

Paired deltas are `GEO-FAM35x3 - GEO-SHARED60` for:

- Macro AP50-95
- Bottom-3 class AP50-95
- Worst-class AP50-95
- SizeMean AP50-95 over the same nine size-defined classes

The analysis also reports family means for `kulit_kopi`, `kulit_tanduk`, and `tanah_batu_ranting` from validation per-class AP50-95.

## Frozen exploratory retain criteria

The family-factorized candidate is retained for final-stage review only if all conditions hold across seeds 42/123/2026:

1. Mean Macro delta >= +0.20 percentage point.
2. Macro improves in at least 2 of 3 seeds.
3. Mean Bottom3 delta is non-negative.
4. Mean Worst delta is non-negative.
5. Mean SizeMean delta >= +0.50 percentage point.
6. SizeMean improves in at least 2 of 3 seeds.
7. At least one of mean Bottom3 or mean Worst gain >= +0.50 percentage point.
8. Mean `kulit_kopi` family delta >= +0.50 percentage point (directly tests reduction of the observed negative-transfer family).
9. Mean `kulit_tanduk` family delta >= -0.50 percentage point (factorization must not erase the family that benefited most from generic GEO1).
10. Static preflight confirms exactly 849 added parameters for each arm, identical target class indices, native-identical initial boxes/scores, and zero-initialized final projections.

These are exploratory candidate-selection criteria, not statistical significance tests.

## Decision semantics

- `RETAIN_FAMILY_FACTORIZATION_FOR_FINAL_STAGE_REVIEW`: family factorization is a viable optimized candidate, still requiring a separate final-stage decision before any locked-test use.
- `KEEP_SHARED_GEOMETRY_STRUCTURE`: exact-capacity family factorization did not justify replacing a shared geometry mapping. Do not tune routing/family adapters further from these same validation results.

No criterion may be changed after results are observed.