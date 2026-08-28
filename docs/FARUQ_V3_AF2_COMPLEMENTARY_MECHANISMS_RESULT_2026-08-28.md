# Faruq-v3 AF2 Complementary Mechanisms Result — 2026-08-28

## Decision

**PASS at seed 42. Retain `AF2SFS1` and freeze a separate paired-confirmation
protocol.** Do not open test. `AF2FS1` and `AF2BHCL1` are rejected and must not
receive additional seeds.

All four arms used the same completed AF2 seed-42 parent and the frozen matched
30-epoch continuation contract. The decision compares every candidate with
`AF2CTRL`, not with the historical pre-continuation AF2 endpoint.

| Arm | Macro mAP50–95 | Bottom-3 mAP50–95 | Worst-class mAP50–95 | Decision |
|---|---:|---:|---:|---|
| `AF2CTRL` | 89.00% | 83.88% | **83.55%** | matched control |
| `AF2FS1` | 89.13% | 82.55% | 81.10% | REJECT |
| **`AF2SFS1`** | **89.96%** | **84.19%** | 83.25% | **RETAIN** |
| `AF2BHCL1` | 85.03% | 74.33% | 73.15% | REJECT |

## Frozen comparisons against AF2CTRL

| Candidate | Δ Macro | Δ Bottom-3 | Δ Worst | Strict Macro gate | Lower-tail Pareto gate |
|---|---:|---:|---:|---:|---:|
| `AF2FS1` | +0.12 point | -1.33 points | -2.45 points | FAIL | FAIL |
| **`AF2SFS1`** | **+0.95 point** | **+0.31 point** | **-0.29 point** | **PASS** | FAIL |
| `AF2BHCL1` | -3.98 points | -9.55 points | -10.39 points | FAIL | FAIL |

`AF2SFS1` satisfies the prospectively frozen strict route: its Macro gain is
at least 0.5 point, Bottom-3 is not lower, and its Worst-class reduction is
within the maximum 1.0-point tolerance. Its 770 added parameters amount to
approximately 0.031% of the 2,511,990-parameter detector control.

## Interpretation and scope

The seed-42 evidence supports adding a compact shared-P3 space/frequency
selector to AF2. Unlike the failed classification-only and naive fusion
directions, this mechanism adapts the shared feature consumed by both native
box and classification branches. The result is a valid screening success, not
yet a three-seed confirmation or test claim.

The next authorized action is to freeze a paired seed-123/2026 protocol that
compares `AF2SFS1` with seed-matched `AF2CTRL` controls. Training additional
seeds before that protocol is frozen, or opening test, is not authorized by
this result.

Protocol:
`docs/FARUQ_V3_AF2_COMPLEMENTARY_MECHANISMS_PROTOCOL_2026-08-28.md`.

Raw evidence:
`docs/evidence/FARUQ_V3_AF2_COMPLEMENT_SEED42_2026-08-28.json`.

Drive decision artifact:
`experiments/faruq-v3-af2-complement-v1/val_reports/af2_complement_seed42_decision.json`.
