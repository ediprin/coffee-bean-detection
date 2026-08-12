# Faruq-v3 AGSF Synthesis Protocol v1

Status: frozen before training. Date: 2026-08-12.

## Research question

Can a single classification-only YOLO26 pathway combine the strongest
spatial-context mechanism from breadth screening with ambiguity-conditioned
multilevel correction, and can a separately encoded AF2 frequency residual
improve that core only when selected by a learned gate?

This is a synthesis hypothesis, not a claim that stacking retained modules is
automatically beneficial. AF12 already demonstrated destructive interaction,
so the frequency extension is isolated by an additive control and a
parameter/schema-matched gated arm.

## Fixed data and controls

- Dataset: Faruq-v3 grouped development, 21 SNI classes.
- Training: grouped train only; evaluation: grouped validation only.
- Test must not be extracted, opened, or used.
- Seed: 42 for synthesis screening.
- Initialization: exact audited D0 checkpoint.
- Schedule: 50 epochs, 640 pixels, batch 16, patience 15, identical across arms.
- Primary reference: completed STB1 seed-42 report.
- Metrics: Macro mAP50-95, Bottom-3 class mAP50-95, Worst-class mAP50-95.

The frozen STB1 reference is Macro 0.886694, Bottom-3 0.836394, Worst
0.808137. Synthesis is required to beat this reference, not merely D0FT.

## Architecture boundary

All arms retain the native YOLO26 backbone, neck, end-to-end assignment, box
heads, and one-to-many/one-to-one execution. Box branches consume the original
P3/P4/P5 features. No ROIAlign, crop classifier, proposal top-k, or decoded-box
classification is permitted.

For level `l`:

```text
S_l = STB(P_l)
C_l = ambiguity-conditioned Select(Align(S_3), Align(S_4), Align(S_5))
R_f = AF2(X) - X
Q_l = E_l(resize(R_f))
z_l = z_l_native(STB(P_l)) + H(z_l_native) * Delta_l
b_l = b_l_native(P_l)
```

The three predeclared arms are:

- `SYN0`: STB plus ambiguity-conditioned multilevel correction; no frequency cue.
- `SYN1`: SYN0 plus unconditional additive AF2 residual cue.
- `SYN2`: SYN0 plus learned spatial gate for the AF2 residual cue.

SYN1 and SYN2 instantiate identical frequency encoder and gate schemas and
must have identical parameter counts. The SYN1 gate output is intentionally
inactive; this controls the presence of the learned selection policy without
changing model size.

The actual checkpoint static audit fixes the following capacity accounting:

- D0: 2,511,990 parameters;
- STB1 architecture: 4,589,201 parameters;
- SYN0: 4,638,114 parameters (+48,913 / +1.07% versus STB1);
- SYN1/SYN2: 4,644,072 parameters (identical to each other).

Therefore a SYN0 gain cannot be described as capacity-free. Parameter count,
FP32 size, and same-device latency must accompany any retained result. If the
gain is marginal, a post-screen capacity control is required before a
mechanistic claim.

All STB residual gates and class-correction layers start at zero, so every arm
must be exactly D0 before learning. AF2 is restricted to the correction path;
it must not alter native box tensors.

## Static gate

Before data access or training, all conditions must pass:

1. Native D0 head state is preserved bitwise.
2. Zero-start one-to-one boxes and scores equal D0.
3. Activated correction changes scores but preserves boxes.
4. STB, class-correction, and applicable frequency/gate gradients are finite.
5. SYN1 and SYN2 have identical parameter counts and state schemas.
6. No ROIAlign, proposal top-k, or box decode is used before classification.

## Fail-fast stages

### Stage 1: core

Train only SYN0. `STB1 vs SYN0` passes only when:

- Macro gain is at least +0.5 percentage point;
- Bottom-3 does not decrease;
- Worst-class decreases by no more than 1 point.

If this fails, stop. Do not train SYN1/SYN2 and do not open test.

### Stage 2: frequency

Only after Stage 1 passes, train SYN1 and SYN2. SYN2 is retained only when the
same gate passes against STB1, SYN0, and SYN1. Otherwise the complete AGSF
direction stops without additional seeds or test.

## Claim boundary

A seed-42 pass authorizes only confirmation on missing seeds. It does not
authorize a final superiority claim or test access. Because validation has
already supported breadth selection, no architecture, gate, loss, threshold,
or frequency setting may be changed after these results are observed.
