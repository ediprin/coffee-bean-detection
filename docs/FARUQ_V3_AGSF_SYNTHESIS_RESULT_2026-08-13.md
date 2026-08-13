# Faruq-v3 AGSF Synthesis Result

Date: 2026-08-13  
Protocol: `faruq-v3-agsf-synthesis-v1`  
Decision: **FAIL — stop before frequency stages and test**

## Seed-42 validation result

| Model | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| STB1 reference | 88.67% | 83.64% | 80.81% |
| SYN0 core synthesis | 86.98% | 79.69% | 78.76% |
| SYN0 minus STB1 | -1.69 points | -3.95 points | -2.06 points |

SYN0 failed all three frozen criteria: it did not gain at least 0.5 Macro
point, did not preserve Bottom-3, and exceeded the permitted one-point
Worst-class drop. The run therefore stopped before the AF2 frequency arms
SYN1/SYN2. The result rejects the proposed AGSF core construction on this
screening; it is not evidence that every individual source mechanism is
ineffective.

Validation only, seed 42 only, and test was not accessed.

Raw artifacts:

- `experiments/faruq-v3-agsf-synthesis-v1/static_audit.json`
- `experiments/faruq-v3-agsf-synthesis-v1/val_reports/SYN0_seed42_val.json`
- `experiments/faruq-v3-agsf-synthesis-v1/val_reports/agsf_core_seed42_decision.json`

