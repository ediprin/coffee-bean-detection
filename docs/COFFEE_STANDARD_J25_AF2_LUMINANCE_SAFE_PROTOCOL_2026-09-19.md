# Coffee Standard J25 AF2LUM-SAFE Final-Package Screen

Status: **frozen before training**
Date: 2026-09-19

## Question

Can the chromaticity-preserving AF2 luminance result be improved without a new
spectral transform by (1) exposing the detector to the full raw-to-enhanced
continuum during training, (2) preserving apparent object scale, and (3)
sampling at source-identity rather than derivative-image level?

## Evidence and scope

The matched seed-42 J25 results are frozen references:

| Arm | Macro mAP50-95 | Bottom-3 | Worst |
|---|---:|---:|---:|
| `D0DIRECT` | 60.5381% | 19.5543% | 1.8967% |
| `AF2DIRECT` | 60.5725% | 18.2777% | 0.0000% |
| `AF2LUMDIRECT` | 61.5006% | 19.2226% | 1.6500% |

`AF2LUMDIRECT` isolated RGB-channel interference, but it did not dominate the
native lower tail.  Its largest remaining deficits are concentrated in
size-defined husk classes.  The old schedule used mosaic, random scale, and
uniform derivative sampling even though the 695 train images represent only
315 source identities.

This study is a **single final-package screen**, not a causal ablation.  If it
advances the Pareto frontier, its training components must be ablated later.

## Candidate

`AF2LUMSAFE` starts fresh from the same official `yolo26n.pt` checkpoint and
the same deterministically initialized 25-class YOLO26n-P3 detector.

The unchanged legacy AF2 statistic is computed from Rec.709 luminance.  One
gate is shared by all RGB channels.  During training only:

```text
x_safe = x * (1 + lambda * G_Y),  lambda ~ Uniform(0, 1) per image
```

At validation and inference, `lambda = 1`.  The frontend remains
parameter-free.  All strengths preserve RGB chromaticity because the same
scalar field multiplies all channels.

The training schedule remains 50 epochs, seed 42, image size 640, batch 16,
and optimizer `auto`, but uses ontology-safe augmentation:

- mosaic disabled;
- random scale disabled;
- hue shift disabled;
- saturation/value jitter limited to 0.2;
- horizontal flip and translation 0.05 retained;
- random erasing, mixup, cutmix, and copy-paste disabled.

Train sampling uses source identities recovered in the frozen dataset.  Class
frequency is computed only from train identities.  A standard square-root
repeat factor uses the median positive train-identity frequency as its
threshold, capped at 4.0.  Derivative siblings split their identity's total
weight equally.  Validation is never resampled.

## Static and runtime gates

- official pretrained SHA is exact;
- initial detector keys, tensors, and parameter count equal native D0;
- zero strength is exactly raw RGB;
- unit strength is exactly deterministic `AF2LUMDIRECT`;
- intermediate strength is finite, bounded between the two endpoints, and
  chromaticity-preserving;
- evaluation always uses unit strength;
- train sampler sees exactly 695 derivatives, 315 identities, and all 25
  classes;
- each identity's derivative weights sum back to its identity repeat factor;
- an epoch-keyed sampler reproduces the same sequence after resume;
- no test directory, YAML key, image, or model evaluation is accessed.

## Decision without arbitrary point threshold

The completed candidate is compared descriptively with fixed `D0DIRECT` and
`AF2LUMDIRECT` references.

- `PARETO_ADVANCE`: no reference dominates the candidate, and the candidate
  dominates at least one reference across Macro, Bottom-3, and Worst.
- `PARETO_TRADEOFF`: no reference dominates the candidate, but it does not
  dominate a reference.
- `DOMINATED_STOP`: D0 or AF2LUM dominates the candidate.

Dominance requires every metric to be no lower and at least one to be higher.
No locked test or extra seed is automatically authorized.  A Pareto advance
first authorizes a component ablation under the same development split.

Output root:
`experiments/coffee-standard-j25-af2-luminance-safe-v1`.
