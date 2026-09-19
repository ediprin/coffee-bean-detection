# Coffee Standard J25 SAFE-AUG0 Control

Status: **frozen before training**

Date: 2026-09-19

`SAFEAUG0` is a single fresh seed-42 causal control. It starts from the same
official `yolo26n.pt`, native YOLO26n P3 model, J25 train-siblings-v2 split,
and 50-epoch schedule as `SAFED0`. It retains the exact semantic-safe
augmentation dictionary but removes identity repeat sampling. It contains no
AF2 frontend during training or inference.

The comparison estimates:

- semantic-safe augmentation effect: `SAFEAUG0 - D0DIRECT`;
- sampler effect under fixed augmentation: `SAFED0 - SAFEAUG0`;
- whether `Biji Hitam Pecah` recovers when the sampler is removed.

Report Macro, Bottom-3, Worst-class, and target-class AP. This is single-seed
component evidence. No other arm, extra seed, or locked-test evaluation is
authorized by this protocol.

