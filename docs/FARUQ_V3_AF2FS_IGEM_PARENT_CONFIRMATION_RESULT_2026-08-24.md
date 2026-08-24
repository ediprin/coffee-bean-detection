# Faruq-v3 AF2FS + IGEM Frozen-Parent Paired Confirmation Result

Date: 2026-08-24  
Branch: `codex/af2-igem-parent-confirmation`  
Decision: **REJECT**  
Next: **STOP_AF2FS_IGEM_RESIDUAL_ROUTE**  
Evaluation: Faruq-v3 grouped development validation only  
Locked test: **not opened**

## Protocol

This result closes the experiment frozen in `docs/FARUQ_V3_AF2FS_IGEM_PARENT_CONFIRMATION_PROTOCOL_2026-08-24.md`.

Each seed uses its canonical seed-matched frozen `AF2FS` parent. The AF2 frontend, YOLO backbone, neck, native classification/localization heads, and parent BatchNorm state remain frozen. Only `model.23.residual.*` is trainable.

Matched arms:

- `AF2IGEM0`: zero-information residual receiving zero tensors shaped like frozen P3/P4/P5;
- `AF2IGEM1`: feature-conditioned residual receiving real frozen P3/P4/P5.

The architecture, capacity, initialization, optimization, auxiliary mask objective, and training schedule are matched. The intended treatment difference is only zero versus real feature conditioning.

## Three-seed aggregate

| Metric | AF2FS parent | AF2IGEM0 control | AF2IGEM1 candidate | Candidate - control |
|---|---:|---:|---:|---:|
| Macro mAP50-95 | 87.6195% | 87.6195% | 87.6427% | **+0.0231 pp** |
| Bottom-3 mAP50-95 | 78.2931% | 78.2931% | 78.1417% | **-0.1513 pp** |
| Worst-class mAP50-95 | 75.1328% | 75.1328% | 74.7546% | **-0.3782 pp** |

Across-seed SD for the candidate was 0.4780 pp Macro, 1.9748 pp Bottom-3, and 3.0670 pp Worst-class.

## Per-seed candidate-minus-control deltas

| Seed | Macro | Bottom-3 | Worst-class |
|---:|---:|---:|---:|
| 42 | -0.0088 pp | -0.3143 pp | -1.0635 pp |
| 123 | +0.0227 pp | -0.0044 pp | -0.1132 pp |
| 2026 | +0.0555 pp | -0.1354 pp | +0.0421 pp |

Macro improved in 2/3 seeds, Bottom-3 improved in 0/3, and Worst-class improved in 1/3.

## Frozen-gate decision

### Parent safety — PASS

All three required safety criteria passed:

- mean Macro candidate-minus-parent >= -0.20 pp;
- mean Bottom-3 candidate-minus-parent >= -1.00 pp;
- mean Worst candidate-minus-parent >= -1.00 pp.

### Route A: aggregate superiority — FAIL

- Macro mean gain >= +0.20 pp: **FAIL** (`+0.0231 pp`);
- Macro improves in at least 2/3 seeds: PASS;
- Bottom-3 mean loss at most 0.50 pp: PASS;
- Worst mean loss at most 1.00 pp: PASS.

Because every Route-A condition was required, Route A fails.

### Route B: lower-tail Pareto improvement — FAIL

- Macro mean loss at most 0.10 pp: PASS;
- Bottom-3 mean gain >= +0.50 pp: **FAIL** (`-0.1513 pp`);
- Bottom-3 improves in at least 2/3 seeds: **FAIL** (`0/3`);
- Worst mean gain >= +1.00 pp: **FAIL** (`-0.3782 pp`);
- Worst improves in at least 2/3 seeds: **FAIL** (`1/3`).

Therefore the frozen decision is:

`RETAIN = parent safety AND (Route A OR Route B) = FALSE`

**Final decision: REJECT.**

## Validity checks

The reported final run states:

- all 21 validation classes present: true;
- canonical parent binding verified: true;
- serialized frozen-parent state verified: true;
- test not opened: true.

The zero-information control reproduced the canonical AF2FS aggregate metrics essentially exactly, supporting its role as a capacity/optimization control rather than a competing learned-information treatment.

## Interpretation

Under this exact parent-preserving formulation, real frozen AF2 P3/P4/P5 conditioning did **not** provide useful enough complementary classification information beyond the matched zero-information residual control. The mean Macro change was only +0.0231 pp and both lower-tail metrics declined.

This result closes only the tested formulation:

`frozen AF2FS parent + classification-only IGEM residual + P3/P4/P5 feature conditioning`.

It does not invalidate standalone AF2 or standalone IGEM evidence, and it must not be generalized to all possible AF2/IGEM formulations.

No threshold is revised after observing the result. No further tuning of this route is authorized by this protocol. The locked test remains closed.

## Reported Kaggle artifacts

The completed Kaggle run reported:

- `af2-igem-parent-confirmation-state.zip` — 266,782 bytes;
- `af2-igem-parent-confirmation-output.zip` — 260,570 bytes.

These filenames are recorded as run provenance; the archives themselves are not committed by this document.

Repository evidence snapshot:
`docs/evidence/FARUQ_V3_AF2FS_IGEM_PARENT_CONFIRMATION_2026-08-24.json`.
