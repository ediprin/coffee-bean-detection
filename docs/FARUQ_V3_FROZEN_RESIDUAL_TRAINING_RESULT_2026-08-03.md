# Faruq-v3 Frozen-D0 Multilevel Residual Training Result

Date: 2026-08-03

Protocol: `faruq-v3-frozen-residual-v1`

Decision: **FAIL — stop without test or additional seeds**

FRM1 was trained on the leakage-safe Faruq-v3 development train split for ten
epochs at seed 42. It was evaluated on validation only. Test remained
unavailable and was not accessed.

## Result

| Model | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| D0 | 79.97% | 68.72% | 65.09% |
| FRM1 | 79.89% | 68.65% | 65.13% |
| FRM1 minus D0 | -0.075 points | -0.075 points | +0.042 points |

The frozen gate required Macro improvement of at least 0.5 points and no
bottom-3 decrease. Neither condition passed. The worst-class preservation
condition passed, but all conditions were required.

## Implementation and optimization evidence

- The static audit bound the run to D0 SHA-256
  `0c458841b84bedce4e0ddada6a5773f6a5ac8a91dad084a4a5f24e89f04e6367`.
- All native D0 parameters were frozen; only 668,953 refiner/gate parameters
  were trainable.
- The best checkpoint has nonzero residual classifier weights and a changed
  gate (`linear.bias = -1.3994`, versus the initial logit for 0.01). Therefore
  the null result is not attributable to a completely inactive refiner.
- Training validation mAP50-95 was highest at epoch 1 (79.91%) and did not
  exceed D0's separately evaluated 79.97%.

## Interpretation boundary

This is a clean negative result for the conservative strategy “freeze D0 and
learn only a confidence-gated multilevel residual.” It shows that preserving
D0 removes the destructive regression observed in prior end-to-end multilevel
heads, but does not create measurable correction headroom on Faruq-v3
validation. It does not establish that multilevel features are useless in
general, nor authorize tuning this same method after its predeclared gate
failed.

Raw artifacts:

- `experiments/faruq-v3-frozen-residual-v1/val_reports/frozen_residual_seed42_decision.json`
- `experiments/faruq-v3-frozen-residual-v1/FRM1_seed42/results.csv`
