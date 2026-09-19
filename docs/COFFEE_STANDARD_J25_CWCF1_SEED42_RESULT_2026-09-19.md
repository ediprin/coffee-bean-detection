# Coffee Standard J25 CWCF1 Seed-42 Result — 2026-09-19

Status: completed validation-only screening. Test was not opened.

## Matched results

| Model | Macro mAP50–95 | Bottom-3 | Worst |
|---|---:|---:|---:|
| D0DIRECT | 60.54% | 19.55% | 1.90% |
| SAFEAUG0 | 62.40% | 21.76% | **1.86%** |
| **CWCF1** | **63.64%** | **24.69%** | 1.49% |

CWCF1 minus the matched safe-augmentation control is **+1.24 points Macro**, **+2.92 points Bottom-3**, and **−0.37 points Worst**. Relative to D0DIRECT it is **+3.11**, **+5.13**, and **−0.41 points**, respectively.

## Interpretation

CWCF1 is a positive Pareto result, not a universal improvement. The classification-only chromatic-wavelet and compositional supervision improves average and lower-tail validation performance under fresh matched training, while the single worst class remains slightly lower. This makes CWCF1 the strongest clean no-repeat-sampler architectural screen currently available on J25, but not evidence of test performance or seed stability.

The next authorized operation is validation-only classwise attribution. No additional training or test evaluation is implied by this result.

## Classwise attribution

The validation-only audit shows that the gain is concentrated in morphology and scale-related groups rather than in the intended black-broken conjunction:

| Attribute group | CWCF1 minus SAFEAUG0 |
|---|---:|
| large | +5.16 points |
| small | +3.37 points |
| skin | +3.12 points |
| foreign | +2.89 points |
| medium | +2.64 points |
| broken | +0.58 points |
| black | +0.08 points |

The largest class gains are `Kulit Kopi Ukuran Besar` (+15.12), `Kulit Tanduk Ukuran Kecil` (+14.13), and `Ranting Ukuran Besar` (+10.12 points). This pattern is consistent with the multiscale-detail path helping morphology/scale discrimination, but it is descriptive and does not isolate the wavelet component causally.

`Biji Hitam Pecah` remains the worst class: 1.90% for D0DIRECT, 1.86% for SAFEAUG0, and 1.49% for CWCF1. Against SAFEAUG0, the target drops 0.37 points while the mean change of its black-only and broken-only primitives is +0.71 points. Therefore the current training-only attribute head does **not** demonstrate successful compositional transfer to the conjunction. Its predictions are auxiliary and are not explicitly recombined into the native class logits at inference.

Revised conclusion: CWCF1 supports the usefulness of chromatic-wavelet conditioning for morphology and lower-tail performance, while the explicit `black + broken` bottleneck remains unresolved.
