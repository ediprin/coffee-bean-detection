# Coffee Standard J25 CWCFHNC1 Protocol — 2026-09-20

Status: **frozen before training**.

## Research question

Can the clean CWCF1 gain be retained while a training-only conditional
classification objective repairs the unresolved `Biji Hitam Pecah` confusion?

## Evidence and exclusion of repeated mechanisms

- CWCF1 improved Macro by 1.24 points and Bottom-3 by 2.92 points over its
  matched SAFEAUG0 control, but `Biji Hitam Pecah` fell from 1.86% to 1.49%.
- All six validation targets of that class are localizable. The observed
  bottleneck is class selection rather than proposal accessibility.
- CWCF2's explicit inference-time attribute composition was ineffective; its
  learned gates were nearly zero and the zero-gate endpoint remained below
  CWCF1. Explicit score composition is therefore not repeated.
- Repeat sampling is excluded because it previously collapsed the target class.

## Frozen candidate

`CWCFHNC1` starts fresh from the official YOLO26n checkpoint with exactly the
CWCF1 architecture, J25 train-siblings-v2 development data, seed 42, safe
augmentation, no repeat sampler, and 50-epoch schedule.

The only change is a training-only conditional cross-entropy with gain `0.05`
over the frozen confusion family:

1. `Biji Hitam Pecah`;
2. `Biji Hitam Penuh`;
3. `Biji Hitam Sebagian`;
4. `Biji Pecah`.

For assigned objects in this family, the loss asks the native 25-class logits
to discriminate within the four-class subset. Classes present in a minibatch
receive equal aggregate weight. Objects outside the family receive exactly
zero contribution. No inference parameter, score rewrite, sampler, ROI, or
second-stage crop is added; the native box path remains unchanged.

## Seed-42 screen

The candidate is retained for confirmation if, relative to historical CWCF1
under the frozen matched protocol:

- Macro mAP50–95 does not fall by more than 0.5 point;
- Bottom-3 does not fall by more than 0.5 point;
- `Biji Hitam Pecah` AP improves by at least 0.5 point;
- all 25 validation classes are present and test remains closed.

The target-class condition is primary because this experiment is explicitly a
bottleneck repair, not another broad architecture search. Failure stops the
mechanism without extra seeds or test evaluation.

## Claim boundary

This is one fresh seed-42 validation screen. Historical CWCF1 is the matched
reference, but the comparison is not multi-seed confirmation and cannot support
a locked-test claim.
