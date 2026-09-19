# Coffee Standard J25 CWCF2 Protocol — 2026-09-19

Status: **frozen before training**.

## Motivation

CWCF1 improved Macro by 1.24 points and Bottom-3 by 2.92 points over the matched SAFEAUG0 control, with gains concentrated in scale, skin, and foreign-material classes. Its `Biji Hitam Pecah` AP nevertheless fell by 0.37 points. The post-screen audit found that CWCF1's attribute predictions were training-only auxiliary outputs and were never explicitly recombined into leaf-class scores.

## Frozen change

CWCF2 retains CWCF1's RGB-native localization path, four chromatic-wavelet cues, official YOLO26n initialization, J25 train-siblings development data, safe augmentation, no repeat sampler, seed 42, and 50-epoch schedule.

Only two changes are allowed:

1. Per-object attribute BCE is balanced so each leaf class present in a batch receives equal aggregate weight.
2. Attribute predictions are converted into a fixed 25-class Bernoulli compatibility score and added to native classification logits through one bounded, zero-initialized scalar per pyramid level. The maximum absolute gain is 0.5.

The compatibility matrix is fixed before training. `Biji Hitam Pecah`, `Biji Hitam Penuh`, and `Biji Pecah` therefore encode `black+broken`, `black+not-broken`, and `not-black+broken`, respectively. Native logits remain responsible for distinctions not uniquely represented by these attributes.

## Static gates

- official checkpoint and schedule match CWCF1/SAFEAUG0;
- initial boxes and scores are bitwise identical to CWCF1/native output;
- activating the composition gate changes scores but not boxes;
- the matching black-broken code scores above its two primitive alternatives;
- cue, auxiliary, and composition gradients are finite;
- test is not extracted or accessed.

## Reporting

CWCF2 is a single fresh seed-42 validation screen. It is compared descriptively against D0DIRECT, SAFEAUG0, and CWCF1. No test or multi-seed claim is authorized by this protocol.
