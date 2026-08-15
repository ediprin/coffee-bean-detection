# Faruq-v3 GEO-C0 vs GEO1 paired three-seed confirmation protocol

Status: **frozen before seed 123/2026 confirmation results are read**.

## Purpose

Seed-42 screening retained explicit predicted-box geometry after a parameter-matched zero-information control. This confirmation asks whether that GEO1 advantage is reproducible across the frozen seeds 42, 123, and 2026.

This stage remains validation-only. It does not authorize locked-test access.

## Frozen arms and seeds

Seeds: `42, 123, 2026`.

For every seed:

- **D0FT**: native YOLO26 continuation reference.
- **GEO-C0**: the same geometry residual MLP as GEO1, but fed a zero-information Bx4xN tensor.
- **GEO1**: the same residual MLP fed detached decoded predicted-box geometry `[w_norm, h_norm, area_norm, aspect_ratio]`.

Seed 42 is reused from the completed screening result. Seeds 123 and 2026 use the already completed D0 and D0FT paired-control artifacts from `faruq-v3-acmc-paired-confirmation-v1`; their provenance must be verified before GEO training. Only GEO-C0 and GEO1 are newly trained for those seeds.

## Architecture and optimization freeze

The GEO architecture, target size-class mask, geometry transforms, clamps, zero initialization, D0 start, optimizer policy, augmentation schedule, image size, batch size, epoch budget, and validation split are exactly those used by the seed-42 screening protocol.

No GT geometry is model input. Geometry is decoded from the detector's own predicted boxes after detaching raw box outputs, so the added classification residual has no direct gradient path into box regression.

No architecture tuning, threshold tuning, class-specific routing, family-aware adapter, or stacking is allowed during this confirmation.

## Static/provenance preflight

Before training seeds 123 and 2026:

1. The seed-42 screening summary must be `RETAIN`, validation-only, and test-locked.
2. Each reused D0FT run must retain a manifest whose `weights_override_sha256` equals the exact paired D0 checkpoint SHA256.
3. The seed-specific static GEO audit must PASS: GEO-C0/GEO1 parameter counts identical, initial native boxes/scores identical, final residual projections zero, and target mask matches size-defined classes.
4. Development data must not expose a test split.

Any preflight failure blocks training.

## Metrics

Primary paired deltas are GEO1 minus GEO-C0 for:

- Macro AP50-95
- Bottom-3 class AP50-95
- Worst-class AP50-95
- Size-class mean AP50-95, computed over the same size-defined validation classes used in seed-42 screening

D0FT is a control-validity reference, not the causal comparator for GEO1.

## Frozen per-seed GEO-C0 validity

For each seed, GEO-C0 relative to D0FT must satisfy all:

1. Macro drop no worse than 1.0 percentage point.
2. Bottom3 drop no worse than 2.0 points.
3. Worst drop no worse than 3.0 points.

All three seeds must PASS this validity check. If the zero-information control is unstable enough to fail on any seed, a geometry-specific causal claim is not confirmed.

## Frozen three-seed GEO confirmation gate

Confirmation passes only if all conditions hold:

1. Mean GEO1−GEO-C0 Macro gain >= +0.20 percentage point.
2. Macro improves in at least 2 of 3 seeds.
3. Mean Bottom3 delta is non-negative.
4. Bottom3 improves in at least 2 of 3 seeds.
5. Mean Worst delta is non-negative.
6. Worst improves in at least 2 of 3 seeds.
7. Mean SizeMean gain >= +0.50 percentage point.
8. SizeMean improves in at least 2 of 3 seeds.
9. At least one of mean Bottom3 or mean Worst gain is >= +0.50 percentage point.
10. All three GEO-C0 validity checks PASS.

No criterion may be changed after seed 123/2026 results are observed.

## Decision

- `PASS` -> geometry conditioning is confirmed on Faruq-v3 validation across the frozen three seeds and may proceed to a separately authorized final stage.
- `FAIL` -> stop the geometry causal claim without locked-test access or post-hoc architecture tuning.

A PASS is still evidence within one dataset/model family, not proof of universal effectiveness or physical-size measurement.