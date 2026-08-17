# Faruq-v3 AF2 Channel-Calibration Factorization Protocol

Status: **frozen before training**
Date: 2026-08-17

## Research question

The completed AF2R screen found that the zero-information control (`AF2R0`)
outperformed both fixed AF2 and the illumination-conditioned candidate. This
study asks whether that gain is caused by additional continuation training or
by the control's ability to learn an input-independent RGB scale for the AF2
residual.

## Mechanism and arms

For fixed AF2 residual `r_c(x)`, the proposed minimal calibrator is:

```text
s_c = 1 + tanh(alpha_c)
y_c = x_c + s_c * r_c(x)
```

There are exactly three trainable logits `alpha_c`, one for each RGB channel.
They initialize to zero, so all scales equal one and inference initially
reproduces fixed AF2 exactly. The scale is bounded to `[0,2]`. It receives no
image, illumination, label, proposal, ROI, or decoded-box input.

| Arm | Added parameters | Purpose |
|---|---:|---|
| `AF2FT30` | 0 | continuation-training control |
| `AF2CAL3` | 3 | channel-wise AF2 residual calibration |

The arms use the same seed-42 AF2 checkpoint, grouped Faruq-v3 train/val,
model YAML, AF2 configuration, seed, and 30-epoch schedule. Test remains
unavailable.

## Static authorization gate

Training is allowed only if the executable audit verifies:

1. identical model YAML, AF2 configuration, and training schedule;
2. `AF2FT30` has exactly the source AF2 parameter count;
3. `AF2CAL3` adds exactly three parameters and one state key;
4. both arms reproduce the AF2-enhanced input bitwise at initialization;
5. detector outputs agree within `1e-4` at initialization;
6. the initial scales equal one exactly;
7. an active scale is bounded, changes the AF2 input, and receives finite
   gradients;
8. all source checkpoint tensors transfer and test access is false.

## Seed-42 validation gate

`AF2CAL3` passes as a causal model improvement only if all are true:

1. Macro gain over `AF2FT30` is at least +0.5 percentage point;
2. Bottom-3 is not lower than `AF2FT30`;
3. Worst-class AP drops no more than one point from `AF2FT30`;
4. Macro is no more than 0.5 point below the completed `AF2R0` result;
5. Bottom-3 and Worst-class AP are each no more than one point below `AF2R0`.

A PASS authorizes a paired three-seed confirmation against `AF2FT30`. A FAIL
stops without extra seeds or test. If `AF2FT30` alone reaches the AF2R0 region
and `AF2CAL3` fails to beat it, the AF2R0 gain is attributed to optimization,
not a new calibration mechanism.

## Claim boundary

A seed-42 PASS supports only a lightweight in-domain calibration direction.
It does not establish multi-seed superiority, external-domain robustness,
illumination robustness, locked-test performance, or deployment readiness.
