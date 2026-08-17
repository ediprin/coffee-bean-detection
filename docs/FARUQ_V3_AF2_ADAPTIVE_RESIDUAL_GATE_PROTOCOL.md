# Faruq-v3 Raw-Preserving Adaptive AF2 Protocol

Status: **frozen before training**  
Date: 2026-08-17

## Research question

Can a lightweight illumination-conditioned gate retain AF2's frequency-domain
benefit on clean fine-grained coffee defects while suppressing harmful AF2
residuals when the input evidence says the fixed filter is unreliable?

This question follows directly from the completed AF2 illumination screen:
fixed AF2 helped warm/cool shifts but lost to D0FT under most exposure,
contrast, and shadow conditions. DIDA-AF2 already showed that adding DG/FG
training losses does not solve the problem. This study changes the AF2
mechanism itself and adds no auxiliary loss.

## Mechanism

For normalized input `x`, ordinary AF2 produces a fixed recovered residual:

```text
r(x) = x * normalize(AF2_recover(x))
AF2(x) = x + r(x)
```

The proposed front end is:

```text
g(x) = 1 + tanh(G(phi(x)))
AF2R(x) = x + g(x) * r(x)
```

`phi(x)` contains six spatially aligned low-level channels: luminance, local
mean luminance, local contrast, relative illumination, red-blue temperature,
and AF2 recovery strength. `G` is a 3x3 Conv--SiLU--1x1 Conv gate with eight
hidden channels and three output channels. Its final projection is initialized
to exactly zero, so `g=1` and both arms initially reproduce AF2 exactly.

The gate is bounded to `[0,2]`: it can suppress the AF2 residual toward the raw
image or amplify it, but it cannot remove the raw path. AF2 recovery remains
deterministic. There is no ROI Align, proposal selection, decoded-box input,
second detector, or training-only auxiliary objective.

## Paired arms

| Arm | Gate input | Purpose |
|---|---|---|
| `AF2R0` | six all-zero maps | parameter- and optimization-matched control |
| `AF2R1` | six illumination/recovery maps | proposed adaptive candidate |

The arms use identical model YAML, AF2 settings, gate parameter count,
state-dict schema, training schedule, seed, data, and AF2 initialization. Only
the information supplied to the gate differs.

## Data, initialization, and schedule

- Dataset: leakage-safe `faruq-development-v3-grouped` train/validation.
- Test must not be restored or read.
- Seed-42 AF2 checkpoint initializes both arms.
- Screening seed: 42 only.
- 30 requested continuation epochs, patience 10, image size 640, batch 16,
  optimizer `auto`, deterministic training, and validation-only selection.
- The two arms may run concurrently in separate Colab accounts because they
  own separate output directories and training locks.

## Static authorization gate

Training is allowed only when the audit verifies:

1. identical arm architecture, parameter count, state schema, and schedule;
2. fewer than 1,000 added parameters;
3. both arms initially reproduce the completed AF2 output within `1e-6`;
4. both gates initially equal exactly one;
5. AF2R0 receives zero information and AF2R1 receives nonzero cues;
6. active gate output is bounded, changes the AF2 input, and has finite
   gradients;
7. test access remains false.

## Seed-42 clean-validation gate

AF2R1 passes only if all are true:

1. Macro gain over AF2R0 is at least +0.5 percentage point;
2. Bottom-3 is not lower than AF2R0;
3. Worst is no more than 1 point below AF2R0;
4. Macro is not lower than the frozen AF2 seed-42 result;
5. Bottom-3 and Worst are each no more than 1 point below frozen AF2.

Failure stops the study without illumination evaluation, extra seeds, or test.
A PASS authorizes only the paired illumination screen.

## Paired illumination gate

The already frozen nine stress conditions and clean-normalized estimand from
the AF2 illumination protocol are reused without changing a threshold. AF2R1
must, relative to AF2R0:

1. preserve clean Macro;
2. have positive mean Macro robustness advantage;
3. have positive Macro advantage in at least 6/9 conditions;
4. preserve mean Bottom-3 robustness;
5. keep mean Worst robustness advantage above -1 point.

Only a PASS may authorize seeds 123/2026 under a separately executable paired
confirmation. Test remains unavailable throughout this study.

## Claim boundary

A successful result supports an end-to-end, trainable AF2 reliability gate for
this grouped development dataset and the frozen synthetic illumination suite.
It does not by itself establish real-lux robustness, varietal robustness,
locked-test superiority, or deployment readiness. Efficiency must be reported
with parameter count, FP32 size, and same-device latency before making a
lightweight claim.
