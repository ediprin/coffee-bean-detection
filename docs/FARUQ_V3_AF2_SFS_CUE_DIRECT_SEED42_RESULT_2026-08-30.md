# Faruq-v3 AF2-SFS-CUE Direct Seed-42 Result — 2026-08-30

## Decision

**STOP_AFTER_SINGLE_ARM.** The direct-from-pretrained combination did not
produce the frozen promotion signal. Do not run a same-runtime control,
component ablation, extra seed, or test evaluation for this candidate.

| Model | Macro mAP50–95 | Bottom-3 | Worst class |
|---|---:|---:|---:|
| historical `AF2DIRECT` | 80.79% | **69.58%** | **66.95%** |
| `AF2SFSCUE1` | **80.84%** | 65.57% | 58.60% |
| descriptive delta | +0.06 point | −4.02 points | −8.36 points |

The candidate preserved aggregate Macro almost exactly, but redistributed the
errors sharply toward the lower tail. It failed both frozen promotion routes:
the Macro gain was below 0.50 point, Bottom-3 was lower, and Worst-class AP
fell far beyond the allowed tolerance.

## Interpretation

This result rejects the specific hypothesis that the previously promising
continuation mechanisms `AF2SFS1` and `AF2CUE1` can simply be combined and
activated from the first optimization step. The result does not invalidate
their earlier matched-continuation evidence. It shows that their optimization
effect is regime-dependent: before a coffee-domain representation has
matured, the combined selector and auxiliary targets preserve mean AP while
causing severe negative transfer to tail classes.

The appropriate follow-up is therefore not a component ablation of this
failed direct arm. Any future use of auxiliary spectral supervision must treat
its activation schedule or staged coffee-domain adaptation as part of the
method and compare against an equal-budget matched control.

## Scope

- Seed: 42 only.
- Source: frozen official 80-class `yolo26n.pt`.
- Maximum/completed epochs: 50/50.
- Coffee-trained parent: none.
- Evaluation: grouped Faruq-v3 validation only.
- Test accessed: false.
- Comparison status: historical matched-protocol screening reference, not a
  same-runtime causal comparison.

Protocol:
`docs/FARUQ_V3_AF2_SFS_CUE_DIRECT_PROTOCOL_2026-08-30.md`.

