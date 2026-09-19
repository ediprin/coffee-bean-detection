# Coffee Standard J25 CWCF1 Protocol — 2026-09-19

Status: **frozen before training**.

## Research question

Can a classification-only chromatic-wavelet path improve J25 fine-grained defect recognition while leaving the native YOLO26n localization path untouched?

## Evidence motivating the design

- J25 diagnostics localize all six validation instances of `Biji Hitam Pecah`, but the final class is wrong. The immediate bottleneck is therefore classification rather than proposal accessibility.
- The legacy RGB AF2 frontend is not superior to the matched native detector on J25.
- Luminance-only AF2 improves Macro modestly, which supports channel interference as a plausible issue, but it does not repair the failed class.
- Safe augmentation improves the matched native control without collapsing the failed class. Repeat sampling is excluded because it changed exposure and collapsed that class.

## Frozen arm

`CWCF1` starts fresh from the official YOLO26n checkpoint and uses the exact `SAFEAUG0` 50-epoch schedule, seed 42, train-siblings J25 development split, and no repeat sampler.

The native RGB backbone and box branches are unchanged. Only classification features receive a zero-initialized residual conditioned on four fixed cues:

1. luminance-reduced blue chroma;
2. luminance-reduced red chroma;
3. level-1 Haar detail energy;
4. level-2 Haar detail energy.

Training adds object-level binary supervision derived deterministically from the 25 leaf labels. In particular, `Biji Hitam Pecah` is represented as the conjunction `black + broken`; `Biji Hitam Penuh` is `black` without `broken`, and `Biji Pecah` is `broken` without `black`. These are derived targets, not new manual annotations.

## Preflight gates

- exact official pretrained checkpoint;
- model and training schedule identical to `SAFEAUG0`;
- initial native boxes and scores bitwise equal;
- activated cue changes scores but not boxes;
- finite nonzero compositional-loss gradients;
- J25 test not extracted or accessed.

## Reporting boundary

This is a single-seed validation screen. Results are reported descriptively against `D0DIRECT` and the matched `SAFEAUG0` control. No locked-test or multi-seed claim is made from this run.

## Design references

- Finder et al., *Wavelet Convolutions for Large Receptive Fields*, ECCV 2024: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7137_ECCV_2024_paper.php
- Tokmakov et al., *Learning Compositional Representations for Few-Shot Recognition*, ICCV 2019: https://openaccess.thecvf.com/content_ICCV_2019/html/Tokmakov_Learning_Compositional_Representations_for_Few-Shot_Recognition_ICCV_2019_paper.html

These papers motivate multiscale frequency cues and attribute-factorized representations; they do not establish that CWCF1 will improve this dataset. That remains the purpose of the frozen screen.
