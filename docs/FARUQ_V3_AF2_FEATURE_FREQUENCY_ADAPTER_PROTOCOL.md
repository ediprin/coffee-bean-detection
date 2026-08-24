# Faruq-v3 AF2 Feature-Frequency Classification Adapter

Status: frozen before training  
Date: 2026-08-22  
Test status: locked; no test access is authorized by this protocol.

## Question

Can channel-wise frequency evidence from the P3/P4/P5 feature pyramids improve
fine-grained classification over an equally optimized AF2 control while leaving
YOLO26 localization unchanged?

The motivation is the completed AF2 mechanism diagnosis: raw proposal
accessibility was unchanged (-0.06 percentage point), while conditional top-1
classification increased by 8.12 points. Consequently this experiment targets
the classification pathway, not proposal generation or box regression.

## Frozen arms

Both arms start from the same seed-matched AF2 checkpoint, retain the original
parameter-free AF2 image frontend, use the same YOLO26n-P3 detector, and run the
same 30-epoch continuation schedule.

| Arm | Classification conditioning |
|---|---|
| `AF2FFA0` | capacity-matched zero-information control |
| `AF2FFA1` | fixed high-frequency energy ratio from each P3/P4/P5 channel |

Each level has learnable channel-wise scale, bias, and residual amplitude. The
residual amplitude is initialized to zero, making both arms exactly AF2 at the
first forward pass. Regression receives the untouched pyramid tensors. There is
no ROIAlign, second crop, decoded-box feedback, or second-stage classifier.

The fixed radial cutoff is 0.35 of the normalized FFT radius. It is a
predeclared architectural constant and must not be tuned using validation.
The adapter adds fewer than 1% of AF2 detector parameters.

## Dataset and evaluation

- Dataset: Faruq-v3 grouped development split.
- Train: 1,665 images; validation: 294 images; all 21 classes present.
- Test must not be extracted or accessed.
- Seed-42 screening first. Seeds 123 and 2026 are authorized only after PASS.
- Metrics: Macro mAP50-95, Bottom-3 class mAP50-95, Worst-class mAP50-95.

## Static gates

Before training, the audit must prove:

1. identity-start boxes and scores are bitwise equal to AF2;
2. an activated adapter changes scores but preserves boxes bitwise;
3. both arms have identical parameter counts and state schemas;
4. only the conditioning source differs;
5. the zero control receives no spectral information;
6. the candidate receives finite spectral information and gradients;
7. added parameters are below 1%; and
8. test access is false.

## Seed-42 decision

`AF2FFA1` is retained only if all are true against `AF2FFA0`:

- Macro mAP50-95 gain is at least +0.5 percentage point;
- Bottom-3 does not decrease; and
- Worst-class decreases by no more than 1 point.

FAIL stops the direction without test or extra seeds.

## Three-seed confirmation

After screening PASS, run paired controls and candidates at seeds 123 and 2026
from their seed-matched AF2 checkpoints. Confirmation PASS requires:

- mean Macro gain at least +0.5 point;
- Macro improves in at least 2/3 seeds;
- mean Bottom-3 is not lower and improves in at least 2/3 seeds; and
- mean Worst-class decrease is no more than 1 point.

Only after confirmation PASS may external, synthetic-density, and illumination
diagnostics be run without training. The already reused Faruq test remains
post-hoc evidence and must not be reopened for tuning.
