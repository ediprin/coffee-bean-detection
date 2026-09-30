# DefectosCafeVerde FSRC1 Seed-42 Result

Completed: **2026-09-30**.

## Decision

**FAIL - STOP_FSRC1.** Do not run additional seeds and do not open test.

The completed D0DIRECT detector remained frozen. FSRC1 trained only its
1,281-parameter frequency-guided reliability calibrator from 6,903 train
candidates (48.65% positive). All dataset, endpoint, box, class, monotonic
score-suppression, validation-class, detector-freeze, and test-lock gates
passed.

| Model | Macro mAP50-95 | Bottom-3 | Worst class |
|---|---:|---:|---:|
| `D0DIRECT` | 91.6197% | 86.7287% | 84.5206% |
| `FSRC1` | 91.5383% | 86.6283% | 84.0173% |
| Delta | **-0.0814 pp** | **-0.1005 pp** | **-0.5033 pp** |

FSRC1 improved seven classes by at most 0.26 point, left `partido` unchanged,
and reduced `negro` by 0.50 point and `normal` by 1.14 points. Neither the
strong-Macro route, lower-tail route, nor strict Pareto route passed.

## Interpretation

The failure is not an implementation or contract failure. Boxes and predicted
class identities were exactly preserved and scores never increased. The
train-only frequency/context reliability signal did not generalize into a
better validation ranking. Suppression-only post-hoc calibration is therefore
not the missing mechanism for the already strong native endpoint.

The result closes frozen-D0 frequency reliability calibration. It does not
justify another calibrator variant, another seed, or test evaluation.
