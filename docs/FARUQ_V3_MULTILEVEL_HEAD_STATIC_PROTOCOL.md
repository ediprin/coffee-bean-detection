# Faruq-v3 Capacity-Matched Multilevel Head Static Protocol

Version: `v1.0.1`

Frozen: 2026-08-02, before architecture injection

## Authorized mechanism

The CM512 probe authorizes one integrated candidate-conditioned classification
branch. The native YOLO26 D0 box and classification convolutions remain intact.
For selected one-to-one candidates, the new branch extracts P3, P4, and P5 ROI
descriptors and adds a residual leaf-class logit before native postprocessing.

This is one serialized detector and one inference graph. It does not save crops
or invoke a separate classifier process.

## Capacity-matched variants

- `MHC0`: P5-only capacity control.
- `MHF1`: P3+P4+P5 residual fusion.

Both variants contain exactly the same modules, parameters, tensor dimensions,
and state-dict schema:

1. ROIAlign 3 x 3 on P3, P4, and P5;
2. channel-wise mean and maximum descriptors;
3. LayerNorm for all three levels;
4. learned P5 side expansion from 512 to 384 dimensions;
5. an 896-to-512 projection;
6. LayerNorm, SiLU, and a 512-to-21 residual classifier.

The only difference is parameter-free:

- `MHC0`: side context equals the learned P5 expansion;
- `MHF1`: the normalized P3+P4 descriptor is added to that same side context.

## Static audit

Before any dataset training, verify:

- identical MHC0/MHF1 parameter counts and state-dict schemas;
- no native box/class branch replacement or weight mutation during injection;
- unchanged YOLO train/eval output contracts;
- numerically D0-equivalent output when residual inference weight is zero,
  using `rtol=0` and the strict absolute tolerance `1e-7`; raw native branch
  tensors must remain bitwise equal;
- numerically MHC0/MHF1-equivalent output under the same tolerance when the
  residual is disabled;
- non-identical MHC0/MHF1 logits when the residual is enabled;
- finite branch gradients from a direct ROI classification loss;
- exact state-dict round trip;
- added parameters no greater than 30% of D0;
- MHF1 CPU smoke latency no more than 25% above MHC0 under the same small
  synthetic input. This is a wiring gate, not the final T4 benchmark.

## Boundary

The static audit performs no optimization step, dataset access, or test access.
PASS authorizes freezing a one-seed validation training protocol. It does not
authorize training itself, additional seeds, or test evaluation.

## Amendment log

`v1.0.0` required bitwise equality for the final decoded detections. Its first
static run failed only because independently executed CPU decode operations
differed by `5.82e-11`; all raw native box, class, and feature tensors were
already bitwise equal. Before any dataset access or optimization, `v1.0.1`
replaced that unsuitable final-output gate with `rtol=0, atol=1e-7`, retained
bitwise equality for every raw native tensor, and required the measured maximum
absolute difference to be recorded. No accuracy result informed this amendment.
