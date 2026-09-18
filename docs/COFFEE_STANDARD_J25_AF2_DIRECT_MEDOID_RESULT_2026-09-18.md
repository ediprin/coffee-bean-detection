# Coffee Standard J25 AF2 Direct Medoid-Only Result

Date: 2026-09-18

## Result

The first matched seed-42 comparison used one visual medoid per recovered
source identity, including training. It therefore contained only 315 training
images for 25 classes.

| Arm | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| D0DIRECT | 46.15% | 8.10% | 0.00% |
| AF2DIRECT | 44.79% | 6.73% | 0.00% |
| AF2 - D0 | -1.36 points | -1.37 points | 0.00 points |

The prospectively frozen promotion gate failed. No extra seed and no locked
test evaluation are authorized from this result.

## Interpretation and amendment boundary

This remains a valid negative result for the **medoid-only low-data regime**.
It is not erased or relabeled. A post-result data-policy review found that the
builder had also discarded augmentation siblings assigned wholly to the
training split. Keeping those siblings in train does not create cross-split
identity leakage and restores 695 training images while leaving the 68-image
validation and 68-identity locked-test assignments unchanged.

The sibling-preserving rerun is a disclosed protocol amendment applied
identically to D0DIRECT and AF2DIRECT. It is not a continuation, checkpoint
reuse, model change, validation split change, or test opening.

- Training images accessed: 315
- Validation images accessed: 68
- Locked-test images accessed: false
- Decision: `STOP_AFTER_SEED42_MEDOID_ONLY`
