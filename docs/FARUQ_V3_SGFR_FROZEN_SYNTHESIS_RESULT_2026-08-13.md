# Faruq-v3 SGFR Frozen Residual Synthesis Result

Date: 2026-08-13  
Protocol: `faruq-v3-sgfr-frozen-synthesis-v1`  
Decision: **FAIL — stop before AF2 frequency stage and test**

## Seed-42 validation result

| Model | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| STB1 reference | 88.67% | 83.64% | 80.81% |
| SGC0 continued-training control | **89.07%** | **84.52%** | **82.37%** |
| SGI1 frozen IGEM residual | 88.71% | 84.08% | 81.67% |

SGI1 minus STB1 was +0.04 Macro, +0.44 Bottom-3, and +0.85 Worst-class
points. It failed because the frozen Macro threshold required at least +0.5
point. More importantly, SGI1 was lower than the optimization-matched SGC0 by
0.36 Macro, 0.44 Bottom-3, and 0.70 Worst-class points. The apparent lower-tail
improvement over STB1 therefore did not establish an IGEM residual effect.

The protocol stopped before SGF2, so the AF2 frequency branch was never
trained. Validation only, seed 42 only, and test was not accessed.

Raw artifacts:

- `experiments/faruq-v3-sgfr-frozen-synthesis-v1/static_audit.json`
- `experiments/faruq-v3-sgfr-frozen-synthesis-v1/val_reports/SGC0_seed42_val.json`
- `experiments/faruq-v3-sgfr-frozen-synthesis-v1/val_reports/SGI1_seed42_val.json`
- `experiments/faruq-v3-sgfr-frozen-synthesis-v1/val_reports/sgfr_geometry_seed42_decision.json`

