# DefectosCafeVerde Train-Only Side-Reliability Routability Result

Completed: **2026-10-06**.

## Decision

**PASS — AUTHORIZE_DVSR1_VALIDATION_SCREEN.**

The frozen five-fold grouped OOF audit evaluated 1,266 physical train pairs.
The learned swap-equivariant side selector improved the source paper's
maximum-confidence rule in every fold and recovered 91.67% of the available
pair-oracle headroom.

| Endpoint | Pair accuracy | Delta vs paper |
|---|---:|---:|
| Paper maximum confidence | 96.76% | — |
| Maximum margin | 96.84% | +0.08 pp |
| Minimum entropy | 95.81% | -0.95 pp |
| **OOF learned selector** | **98.50%** | **+1.74 pp** |
| Pair oracle | 98.66% | +1.90 pp |

The learned selector produced 24 rescues and two regressions, for 22 net
rescues. Fold-level net changes were `+1`, `+3`, `+6`, `+7`, and `+5`.
All frozen authorization gates passed. No detector training, validation access,
or test access occurred.

## Class behavior

The largest net rescue counts were `agrio` (+14), `helado` (+5), `elefante`
(+2), `partido` (+2), and `concha` (+1). Small regressions occurred for
`oreja` (-1) and `broca` (-1). The result therefore supports a learned
reliability mechanism rather than a universal confidence or entropy heuristic.

## Interpretation boundary

This is grouped train-only OOF evidence. It establishes that side reliability
is learnable without synthesizing a class or modifying a detector box, but it
does not establish validation superiority over confirmed DVF1. Exactly one
frozen DVSR1 validation screen is authorized next. Test remains unopened.

## Artifact

The raw JSON is stored outside Git under the shared Drive project:

`experiments/defectoscafeverde-dvsr-routability-v1/dvsr_routability.json`
