# Coffee Standard J25 AF2 Luminance-Isolation Result

Date: 2026-09-19
Status: **completed seed-42 validation diagnostic; test not opened**

## Result

All arms below use the same reconstructed J25 train-siblings development
artifact, official YOLO26n initialization, 50-epoch schedule, seed 42, and
68-image validation partition. `AF2LUMDIRECT` differs from `AF2DIRECT` only in
computing one AF2 gate from Rec.709 luminance and sharing it across RGB.

| Arm | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| `D0DIRECT` | 60.5381% | 19.5543% | 1.8967% |
| `AF2DIRECT` | 60.5725% | 18.2777% | 0.0000% |
| `AF2LUMDIRECT` | **61.5006%** | 19.2226% | 1.6500% |

`AF2LUMDIRECT - AF2DIRECT`:

- Macro: **+0.9282 points**;
- Bottom-3: **+0.9449 points**;
- Worst class: **+1.6500 points**.

`AF2LUMDIRECT - D0DIRECT`:

- Macro: **+0.9625 points**;
- Bottom-3: **-0.3316 points**;
- Worst class: **-0.2467 points**.

## Interpretation

The matched result supports the narrow hypothesis that independent RGB AF2
gates mix useful structure with channel-specific chromatic variation on J25.
Sharing a luminance-derived gate removes most of the lower-tail damage and
raises Macro. It does not fully dominate the native detector: Bottom-3 and
Worst remain slightly below `D0DIRECT`. Therefore this is evidence for a real
failure mechanism, not evidence that AF2 alone is sufficient on every coffee
dataset.

This is a one-seed validation diagnostic. It does not authorize a J25 test
evaluation, a cross-dataset robustness claim, or a final superiority claim.
The next prospective experiment is frozen separately as `AF2LUMSAFE`.

## Research lock

- Training split: 695 derivatives from 315 assigned source identities.
- Validation split: 68 source representatives with all 25 classes.
- Locked test images accessed: **false**.
- Additional seed authorized by this result: **false**.
