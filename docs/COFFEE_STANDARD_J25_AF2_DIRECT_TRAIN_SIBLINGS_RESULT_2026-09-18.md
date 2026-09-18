# Coffee Standard J25 AF2 Direct Train-Siblings Result

Status: **completed; stop after seed 42**

Date: 2026-09-18

The amended matched screen used 695 train images (315 source identities), 68
validation identities, and a manifest-only locked test. Both arms started from
the same official YOLO26n pretrained detector state and used the same 50-epoch
schedule. Test images were not extracted or evaluated.

| Arm | Macro mAP50-95 | Bottom-3 | Worst-class |
|---|---:|---:|---:|
| `D0DIRECT` | 60.5381% | 19.5543% | 1.8967% |
| `AF2DIRECT` | 60.5725% | 18.2777% | 0.0000% |
| AF2 minus D0 | +0.0344 pp | -1.2765 pp | -1.8967 pp |

Neither the overall route nor the lower-tail route passes. AF2 is effectively
tied on Macro and worse on the lower tail. The larger leakage-safe training set
substantially improves both arms over the medoid-only regime, but does not make
the RGB-independent AF2 frontend superior on J25.

Decision: `STOP_AFTER_SEED42`. No extra seed and no locked-test access.

