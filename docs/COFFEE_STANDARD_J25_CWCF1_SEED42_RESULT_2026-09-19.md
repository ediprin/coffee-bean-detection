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
