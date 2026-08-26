# Faruq-v3 AF2 Class-Selective DLRBC Result — 2026-08-26

## Decision

**STOP_AFTER_SEED42.** Do not run extra seeds or open test.

The 20-epoch `AF2CSD1_seed42` screen completed against the frozen
`AF2DIRECT_seed42` parent. The selective residual produced a small Pareto
movement but did not satisfy the prospectively frozen selected-class effect
gate.

| Model | Macro mAP50–95 | Bottom-3 mAP50–95 | Worst-class mAP50–95 |
|---|---:|---:|---:|
| AF2DIRECT | 80.79% | 69.58% | 66.95% |
| AF2CSD1 | 80.86% | 69.80% | 66.95% |
| Delta | +0.07 point | +0.22 point | +0.00 point |

Mean AP across the train-selected classes increased by 0.23 point. The frozen
minimum was 0.50 point, so this criterion failed. All other gates passed:
Macro did not fall more than 0.1 point, Bottom-3 did not fall, Worst-class did
not fall more than 0.5 point, all 21 validation classes were present, and test
was not opened.

## Interpretation

The engineering isolation worked: the class-selective residual preserved the
AF2 lower tail while producing a small positive Macro and Bottom-3 movement.
However, the independently observed train-only DLRBC complementarity did not
translate into a sufficiently large validation effect. This is evidence that
global DLRBC's useful signal is too weak or too protocol-specific to justify
promotion through this selective route.

This result is scoped to the direct-from-pretrained AF2 protocol. Its 80.79%
AF2DIRECT baseline must not be substituted for historical AF2 results around
88%, which use a different training/initialization contract.

Protocol:
`docs/FARUQ_V3_AF2_CLASS_SELECTIVE_DLRBC_PROTOCOL_2026-08-26.md`.

Raw evidence:
`docs/evidence/FARUQ_V3_AF2_CLASS_SELECTIVE_DLRBC_SEED42_2026-08-26.json`.
