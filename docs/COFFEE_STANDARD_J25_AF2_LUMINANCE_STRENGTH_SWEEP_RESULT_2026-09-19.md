# Coffee Standard J25 AF2LUMSAFE Inference-Strength Result

Date: 2026-09-19

Status: **completed; diagnostic only; test not opened**

The completed seed-42 `AF2LUMSAFE` checkpoint was evaluated on the unchanged
validation set without training at five inference strengths.

| Strength | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class | `Biji Hitam Pecah` |
|---:|---:|---:|---:|---:|
| 0.00 | **66.57%** | **28.05%** | 0.00% | 0.00% |
| 0.25 | 66.45% | 26.89% | 0.00% | 0.00% |
| 0.50 | 66.30% | 26.55% | 0.00% | 0.00% |
| 0.75 | 66.16% | 26.05% | 0.00% | 0.00% |
| 1.00 | 65.89% | 25.70% | 0.00% | 0.00% |

Macro and Bottom-3 decrease monotonically as AF2 strength increases. The best
descriptive endpoint is therefore `lambda=0`, but the zero-AP target is
unchanged at every strength. This means full-strength inference mismatch is
not the cause of the `Biji Hitam Pecah` failure. It also means the package's
large gain cannot be attributed to using the AF2-transformed image at
inference; semantic-safe augmentation, identity-aware sampling, and possibly
training-time stochastic AF2 regularization remain entangled.

The next step is the frozen validation-only target root-cause audit. It checks
whether the zero is caused by absent localization, wrong-class assignment, or
final ranking/selection. It performs no training and does not open the locked
test.

