# Faruq-v3 Geometry Conditioning Screening Result

Date: 2026-08-15  
Protocol: `faruq-v3-geometry-conditioning-screening-v1`  
Evaluation: Faruq-v3 validation only, seed 42  
Test opened: **No**

## Decision

**RETAIN — authorize paired three-seed GEO confirmation.**

GEO1 passed every frozen retain criterion against the parameter-matched
zero-information control GEO-C0. This result authorizes only paired validation
confirmation on additional seeds. It does not authorize test access.

## Results

| Model | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 | Size-class mean mAP50-95 |
|---|---:|---:|---:|---:|
| D0FT | 86.6887% | 74.9809% | 72.0224% | n/a |
| GEO-C0 | 87.0901% | 75.5103% | 70.3567% | 86.7242% |
| **GEO1** | **87.3120%** | **80.6079%** | **78.7054%** | **88.3342%** |

GEO1 minus GEO-C0:

- Macro: **+0.2219 percentage point**
- Bottom-3: **+5.0977 points**
- Worst-class: **+8.3488 points**
- Nine size-labelled classes mean: **+1.6100 points**

GEO1 minus D0FT:

- Macro: **+0.6233 point**
- Bottom-3: **+5.6271 points**
- Worst-class: **+6.6830 points**

## Frozen-gate audit

GEO-C0 validity passed. Relative to D0FT, its Macro and Bottom-3 increased,
while Worst-class decreased by 1.6658 points, within the predeclared maximum
3-point drop.

All GEO1 retain criteria passed:

1. Macro gain over GEO-C0 was at least +0.20 point.
2. Bottom-3 was preserved and materially improved.
3. Worst-class was preserved and materially improved.
4. At least one tail metric improved by at least +0.50 point.
5. Size-class mean improved by at least +0.50 point.
6. Macro was not worse than D0FT by more than 0.20 point.

## Capacity and interpretation

GEO-C0 and GEO1 each contain 2,512,843 parameters, only **853 parameters**
more than native D0. GEO1 and GEO-C0 have the same residual MLP capacity; the
experimental difference is whether the adapter receives detached predicted-box
geometry or a zero-information tensor.

The result therefore supports the narrow seed-42 claim that predicted geometry
contains useful information for the size-labelled SNI classes and substantially
improves lower-tail validation performance. It does not yet establish
multi-seed stability, physical-size inference, or locked-test generalization.

## Seed-42 candidate context

These rows are a descriptive same-validation snapshot. A higher single-seed
score is not by itself a causal or final selection result.

| Model | Macro | Bottom-3 | Worst | Current evidence status |
|---|---:|---:|---:|---|
| STB1 | 88.669% | 83.639% | 80.814% | high absolute score; mechanism failed its capacity-control gate |
| AF2 | 88.197% | 80.043% | 79.347% | passed paired three-seed validation |
| IGEM1 | 88.008% | 82.183% | 82.078% | passed paired three-seed validation |
| FT1 | 87.719% | 80.663% | 80.236% | seed-42 screening result |
| ACMC1 | 87.624% | 80.405% | 79.490% | passed paired validation; locked test not confirmed |
| SAF1 | 87.337% | 81.330% | 80.336% | seed-42 screening result |
| **GEO1** | **87.312%** | **80.608%** | **78.705%** | retained; paired three-seed confirmation pending |
| GEO-C0 | 87.090% | 75.510% | 70.357% | valid parameter-matched control |
| D0FT | 86.689% | 74.981% | 72.022% | matched-training baseline |

## Next action

Run a paired GEO-C0 versus GEO1 confirmation for seeds 42, 123, and 2026,
reusing seed 42. Keep the test split locked. The three-seed result must be
reported separately and must not retroactively change this frozen seed-42 gate.

## Evidence source

Authoritative runtime report:

`Coffee_Bean_Detection/experiments/faruq-v3-geometry-conditioning-screening-v1/val_reports/geometry_conditioning_seed42_decision.json`

Repository evidence snapshot:

`docs/evidence/FARUQ_V3_GEOMETRY_CONDITIONING_RESULT_2026-08-15.json`
