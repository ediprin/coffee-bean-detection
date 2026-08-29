# Faruq-v3 AF2-SPDS Seed-42 Result

Status: completed on the development validation split.  
Test: not accessed.

## Headline metrics

| Arm | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| `AF2BASE` | 88.89% | 80.84% | 77.53% |
| `AF2RGBDS` | 88.64% | 83.62% | 82.61% |
| `AF2SPDS` | 88.65% | 84.67% | 83.44% |

`AF2SPDS - AF2BASE` was -0.24/+3.83/+5.91 points for
Macro/Bottom-3/Worst. `AF2SPDS - AF2RGBDS` was +0.01/+1.05/+0.82 points.

## Interpretation and decision

The AF2-specific target contributed lower-tail information beyond generic RGB
reconstruction, but the frozen lower-tail route allowed at most a 0.1-point
Macro loss. The observed 0.24-point Macro loss therefore produced
`FAIL_KILL_GATE`; it must not be relabelled as a protocol PASS after seeing the
result.

Code inspection after that decision found that the old target `AF2(x)-x` is
mathematically `RGB * normalized_recovery_gate`, so it is not an isolated
frequency target. The auxiliary gain also remained 0.10 through the last
training update. Those two distinct defects motivate the separately frozen
`AF2CUE1` and `AF2DECAY1` refinement arms.

Raw decision artifact:
`experiments/faruq-v3-af2-spds-v1/val_reports/af2_spds_seed42_decision.json`.
