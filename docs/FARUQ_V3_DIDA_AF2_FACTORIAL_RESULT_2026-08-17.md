# Faruq-v3 DIDA-AF2 Factorial Result

Date: 2026-08-17  
Decision: **FAIL -- stop without test or extra seeds**

## Protocol

The prospectively frozen protocol is
`docs/FARUQ_V3_DIDA_AF2_FACTORIAL_PROTOCOL.md`. The four arms used the same
AF2 seed-42 initialization, dataset, seed, schedule, two-forward compute, and
inference graph. Only the DG and FG training-objective flags differed. The
static implementation gate passed before training.

## Seed-42 validation results

| Arm | DG | FG | Macro mAP50-95 | Bottom-3 | Worst class |
|---|---:|---:|---:|---:|---:|
| `AF2FT` | no | no | **87.68%** | **78.37%** | 75.12% |
| `AF2DG` | yes | no | 87.05% | 76.31% | 73.13% |
| `AF2FG` | no | yes | 87.61% | 78.37% | **75.29%** |
| `AF2DGFG` | yes | yes | 86.92% | 75.60% | 73.65% |

No validation class was missing ground truth. Test was not accessed.

## Factorial effects

Effects below are absolute proportions; point changes are shown in
parentheses.

| Effect | Macro | Bottom-3 | Worst |
|---|---:|---:|---:|
| DG (`AF2DG - AF2FT`) | -0.00624 (-0.62 pt) | -0.02057 (-2.06 pt) | -0.01995 (-1.99 pt) |
| FG (`AF2FG - AF2FT`) | -0.00065 (-0.06 pt) | -0.00001 (~0.00 pt) | +0.00165 (+0.16 pt) |
| Joint vs control | -0.00763 (-0.76 pt) | -0.02773 (-2.77 pt) | -0.01469 (-1.47 pt) |
| DG x FG interaction | -0.00074 (-0.07 pt) | -0.00715 (-0.71 pt) | +0.00361 (+0.36 pt) |

The joint arm also lost 0.14 Macro point to `AF2DG` and 0.70 point to
`AF2FG`. Its Bottom-3 was lower than both single-factor arms.

## Frozen gate

All seven acceptance criteria failed:

- joint Macro was lower than control;
- joint Bottom-3 was lower than control;
- joint Worst dropped more than one point;
- joint Macro did not gain at least 0.5 point over DG;
- joint Macro did not gain at least 0.5 point over FG;
- joint Bottom-3 was lower than DG;
- joint Bottom-3 was lower than FG.

## Interpretation and boundary

Under this frozen implementation, weak-to-style consistency harmed in-domain
fine-grained discrimination. The dynamic Top-3 margin alone was effectively
neutral: it slightly improved the worst class but did not improve Macro or
Bottom-3. Combining FG with DG did not recover the DG loss and produced a
negative interaction for Macro and Bottom-3.

This rejects the specific DIDA-AF2 objective, not AF2 itself and not all
domain-generalization or margin-learning methods. The protocol requires the
study to stop at seed 42. Seeds 123/2026 and test evaluation are unauthorized.

Raw reports remain in the shared project artifact tree:

`experiments/faruq-v3-dida-af2-factorial-v1/val_reports/`

