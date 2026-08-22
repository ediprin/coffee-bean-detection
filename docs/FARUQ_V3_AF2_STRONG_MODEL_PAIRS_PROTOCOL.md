# Faruq-v3 Direct AF2 + Strong-Model Pairs Protocol

Status: **FROZEN BEFORE TRAINING**  
Date: 2026-08-23

## Question

Can the historical parameter-free AF2 input frontend complement the three
strongest non-enhancement seed-42 mechanisms: STB1, IGEM1, and SAF1?

## Arms

- `AF2STB1`: AF2 before the unchanged STB1 detector.
- `AF2IGEM1`: AF2 before the unchanged IGEM1 detector and loss.
- `AF2SAF1`: AF2 before the unchanged SAF1 detector.

Every candidate starts from the same D0 seed-42 checkpoint and uses the exact
standalone 50-epoch schedule. AF2 is active during training and inference. It
has no trainable parameters and no ROI, crop, decoded-box, or second-stage
dependency. The standalone checkpoint is evaluated again on the same
validation loader rather than copying a historical headline.

## Gate

The first screen is seed 42 only. A pair is `RETAIN_STRICT_SUPERIOR` when all
three headline metrics are non-lower and at least one improves by 0.2 point.
It is `RETAIN_PARETO` when Macro drops by at most 0.1 point, both tail metrics
are non-lower, and Bottom-3 improves by at least 0.5 point or Worst by at least
1 point. This deliberately avoids a rigid +0.5 Macro-only rule.

Only retained pairs may enter paired multi-seed confirmation. Faruq test and
all external datasets remain unopened during this screen.
