# Coffee Standard J25 AF2LUMSAFE Seed-42 Result

Date: 2026-09-19

Status: **PARETO_TRADEOFF; classwise review completed; test not opened**

## Headline result

| Arm | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| `D0DIRECT` | 60.5381% | 19.5543% | 1.8967% |
| `AF2LUMDIRECT` | 61.5006% | 19.2226% | 1.6500% |
| `AF2LUMSAFE` | **65.8872%** | **25.7021%** | 0.0000% |

Relative to `D0DIRECT`, the final package improved Macro by 5.35 points and
Bottom-3 by 6.15 points, while Worst decreased by 1.90 points. Relative to
`AF2LUMDIRECT`, Macro improved by 4.39 points and Bottom-3 by 6.48 points,
while Worst decreased by 1.65 points. No reference dominates the candidate,
and the candidate does not dominate either reference; the frozen decision is
therefore `PARETO_TRADEOFF`.

## Classwise localization of the tradeoff

The zero-AP class is `Biji Hitam Pecah`. It was already nearly unresolved by
the references: 1.90% for `D0DIRECT` and 1.65% for `AF2LUMDIRECT`. The other
two classes comprising the candidate's Bottom-3 average 38.55%, so the zero is
an isolated class failure rather than broad lower-tail collapse.

Notable candidate deltas against `D0DIRECT` include:

- `Biji Hitam Penuh`: +27.58 points;
- `Kulit Kopi Ukuran Kecil`: +16.57 points;
- `Brown Yellow -Sour Bean-`: +10.54 points;
- `Kulit Kopi Ukuran Besar`: +7.13 points;
- `Biji Hitam Sebagian`: +6.59 points;
- `Biji Normal`: -11.02 points;
- `Biji Berlubang Satu`: -4.15 points;
- `Biji Hitam Pecah`: -1.90 points.

These observations show redistribution across defect decisions, but do not by
themselves identify whether stochastic AF2 strength, source-identity sampling,
or the ontology caused the isolated zero.

## Next diagnostic

A validation-only inference-strength sweep is frozen separately. It evaluates
the same completed checkpoint without training at five AF2 strengths. The
diagnostic tests whether the training distribution over AF2 strength conflicts
with full-strength validation. It cannot produce a tuned test claim, and the
locked test remains closed.
