# Coffee Standard J25 SAFE-D0 Seed-42 Result

Date: 2026-09-19

Status: **completed; Pareto tradeoff; test not opened**

| Model | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class |
|---|---:|---:|---:|
| `D0DIRECT` | 60.54% | 19.55% | 1.90% |
| `SAFED0` | **66.84%** | 26.86% | 0.00% |
| `AF2LUMSAFE`, lambda=0 | 66.57% | **28.05%** | 0.00% |

Relative to `D0DIRECT`, `SAFED0` gained 6.30 Macro points and 7.31 Bottom-3
points while losing 1.90 Worst-class points. Relative to `SAFED0`, stochastic
AF2 exposure followed by raw-RGB inference changed Macro by -0.27 points and
Bottom-3 by +1.19 points. Therefore the large aggregate improvement belongs
primarily to the safe training policy, not AF2. AF2 contributes only a small
single-seed Pareto shift toward the lower tail.

`Biji Hitam Pecah` remained at 0.00% under both `SAFED0` and `AF2LUMSAFE` at
lambda zero. Removing AF2 did not recover the target. The next no-training
audit examines target support, sampler exposure, co-occurrence, and raw/final
confusion before choosing one component ablation.

