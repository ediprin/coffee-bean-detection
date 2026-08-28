# Faruq-v3 AF2 Signal-Preservation Deep Supervision Protocol

Date frozen: 2026-08-29  
Status: implementation complete; static audit required before training  
Test status: locked

## Research question

Can training-only multilevel reconstruction preserve the useful AF2 signal in
P3/P4/P5 without changing the native AF2 detection path at inference?

This study follows the failed `AF2MTS1` screen. `AF2MTS1` changed the feature
tensors consumed by Detect and reduced Macro/Bottom-3/Worst by 0.73/6.52/8.27
points. AF2-SPDS instead uses read-only auxiliary branches: Detect receives the
original P3/P4/P5 tensors bitwise unchanged.

## Arms and causal controls

| Arm | Auxiliary target | Purpose |
|---|---|---|
| `AF2BASE` | none (zero connected loss) | matched decoder capacity/compute control |
| `AF2RGBDS` | original RGB input | generic reconstruction/deep-supervision control |
| `AF2SPDS` | `AF2(x) - x` | AF2 signal-preservation treatment |

All arms use the same three 1x1 decoders (P3/P4/P5 to three channels), the same
AF2 parent checkpoint, AF2 configuration, optimizer schedule, seed, and data.
The only changed variable is the auxiliary target. Auxiliary gain is frozen at
0.10 before validation is observed.

## Training and inference contract

- Dataset: leakage-safe Faruq-v3 grouped development data; train and validation only.
- Initialization: completed AF2 seed-42 checkpoint.
- Screen: seed 42, 30-epoch matched continuation, image size 640, batch 16.
- Native YOLO detection loss remains unchanged except for adding the auxiliary
  reconstruction scalar to the classification loss slot.
- Auxiliary decoders read P3/P4/P5 and never replace, add to, gate, or mutate
  features passed to Detect.
- RGB and AF2-signal targets are detached; gradients flow through decoder and
  detector features, never through target construction.
- At inference auxiliary decoders are not executed. They are structurally
  removable, and removal must preserve detector output exactly and restore the
  native AF2 state schema.
- No ROI, second crop, decoded-box dependency, test access, or validation-tuned
  hyperparameter search is allowed.

## Static gates

Training is authorized only if all arms have:

1. identical AF2 configuration, model YAML, schedule, parameter count, and
   auxiliary state schema;
2. exact initial AF2 detector output;
3. exact output and native state schema after stripping auxiliary heads;
4. finite loss and gradients; nonzero treatment gradients and exactly-zero
   matched-control gradients;
5. active AF2 signal distinct from RGB;
6. no test access.

## Seed-42 decision

Let deltas be against `AF2BASE`. `AF2SPDS` must satisfy either:

- strong-Macro route: Macro +0.5 point, Bottom-3 non-lower, Worst drop no more
  than 0.5 point; or
- lower-tail route: Macro drop no more than 0.1 point, Bottom-3 +0.5 point, and
  Worst +0.5 point.

In addition, cue-specific evidence is mandatory: `AF2SPDS` must beat
`AF2RGBDS` on at least two of Macro/Bottom-3/Worst, while its Macro may not be
more than 0.1 point lower. This prevents generic deep supervision from being
misreported as AF2-signal preservation.

PASS freezes a paired seed-123/2026 confirmation protocol. FAIL retains the
original AF2 and stops this direction without opening test.

## Scientific basis

The design combines three established principles without copying a detector
architecture wholesale: training-only auxiliary heads/deep supervision,
reconstruction supervision for retaining frequency information, and AF2's
input-frequency cue. The contribution tested here is their controlled,
removable integration for fine-grained coffee-defect detection.

