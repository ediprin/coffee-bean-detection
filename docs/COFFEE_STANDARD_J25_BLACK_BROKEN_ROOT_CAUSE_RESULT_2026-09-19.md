# Coffee Standard J25 `Biji Hitam Pecah` Root-Cause Result

Date: 2026-09-19

Status: **completed; validation only; test not opened**

The target has six validation instances. At IoU 0.50, both `D0DIRECT` and
`AF2LUMSAFE` at inference strength zero localized all six instances in their
raw top-500 candidates and in low-confidence final detections. Localization
accessibility and matched recall were therefore 100% for both models.

| Model | Stage | Correct class | Wrong class | Conditioned class accuracy |
|---|---|---:|---:|---:|
| `D0DIRECT` | raw top-500 | 2/6 | 4/6 | 33.33% |
| `D0DIRECT` | final, conf 0.001 | 0/6 | 6/6 | 0.00% |
| `AF2LUMSAFE`, lambda=0 | raw top-500 | 0/6 | 6/6 | 0.00% |
| `AF2LUMSAFE`, lambda=0 | final, conf 0.001 | 0/6 | 6/6 | 0.00% |

The failure is not a box-localization deficit. `D0DIRECT` contains two
class-correct raw decisions that disappear from its final predictions, while
`AF2LUMSAFE` is already wrong at the raw class decision. Because AF2 is turned
off at inference for this endpoint, the latter shift was acquired during
training. A fresh `SAFED0` control is therefore frozen to distinguish the
semantic-safe data policy from stochastic-AF2 training exposure.

