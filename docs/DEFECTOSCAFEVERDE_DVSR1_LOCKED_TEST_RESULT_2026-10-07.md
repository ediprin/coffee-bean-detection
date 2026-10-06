# DefectosCafeVerde DVSR1 Final Locked-Test Result

Completed: **2026-10-07**.

## Final decision

**FINAL_DVSR1_GENERALIZATION_CONFIRMED.**

DVSR1 remained superior on the untouched grouped physical-bean test split.
No detector, fuser, or selector fitting was executed during this evaluation.
Only test was present in the runtime dataset, and every frozen provenance and
checkpoint gate passed.

| Endpoint | Physical-pair accuracy | Macro class accuracy | Bottom-3 class accuracy | Worst-class accuracy |
|---|---:|---:|---:|---:|
| Paper maximum confidence | 92.73% | 94.08% | 80.55% | 64.71% |
| Confirmed DVF1 mean | 92.73% | 94.08% | 80.55% | 64.71% |
| **DVSR1** | **95.15%** | **95.99%** | **85.92%** | **76.47%** |
| Pair oracle | 95.76% | 96.48% | 85.92% | 76.47% |

Against both the paper rule and three-seed DVF1 test mean, DVSR1 gained:

- **+2.42 points** physical-pair accuracy;
- **+1.91 points** Macro class accuracy;
- **+5.37 points** Bottom-3 class accuracy; and
- **+11.76 points** worst-class accuracy.

The paper rule correctly classified 153 of 165 eligible physical pairs,
DVSR1 classified 157, and pair oracle classified 158. DVSR1 therefore captured
four of five available oracle-headroom pairs (**80%**). Relative to paper it
produced five rescues, one regression, 152 pairs correct under both methods,
and seven pairs wrong under both.

## Classwise result

| Class | Paper | DVSR1 | Delta |
|---|---:|---:|---:|
| agrio | 64.71% | **76.47%** | **+11.76 pp** |
| oreja | 90.91% | **100.00%** | **+9.09 pp** |
| normal | 86.96% | **91.30%** | **+4.35 pp** |
| caracolillo | 96.43% | **100.00%** | **+3.57 pp** |
| negro | 100.00% | 100.00% | +0.00 pp |
| broca | 100.00% | 100.00% | +0.00 pp |
| helado | 90.00% | 90.00% | +0.00 pp |
| elefante | 100.00% | 100.00% | +0.00 pp |
| seca | 100.00% | 100.00% | +0.00 pp |
| partido | 100.00% | 100.00% | +0.00 pp |
| triangulo | 100.00% | 100.00% | +0.00 pp |
| concha | **100.00%** | 94.12% | **-5.88 pp** |

The `concha` regression is retained as final locked-test evidence and must not
be used to tune the selector.

## Evidence across splits

| Split | Paper pair accuracy | DVSR1 pair accuracy | Delta |
|---|---:|---:|---:|
| Grouped train OOF | 96.76% | **98.50%** | **+1.74 pp** |
| Validation | 94.41% | **96.28%** | **+1.86 pp** |
| Final locked test | 92.73% | **95.15%** | **+2.42 pp** |

The direction persisted across grouped train OOF, validation, and final test.
DVF1 seeds 42, 123, and 2026 all reproduced the paper endpoint exactly on
test, whereas DVSR1 changed the selected native side and retained a material
gain.

## Population and claim boundary

The test source contained 403 images in 213 physical groups. The frozen paired
endpoint accepted 165 groups. Forty-eight groups were rejected before scoring:

- 23 did not contain exactly two views;
- 10 had disagreeing paired labels; and
- 15 did not contain exactly one object per view.

Claims therefore apply to eligible dual-view physical beans with two
label-consistent, single-object views. Pair oracle is non-deployable and is
reported only as diagnostic headroom.

## Final method statement

DVSR1 is a deterministic, swap-equivariant side-reliability selector over the
frozen D0 detector outputs. It uses the complete per-side score distribution
rather than maximum confidence alone, then selects one existing native class
and its corresponding frozen detector box. It does not synthesize a class,
regress a new box, retrain the detector, or fit on validation/test.

The study terminates with `REPORT_LOCKED_TEST_AND_STOP`. No additional
architecture, feature, threshold, or seed search is authorized from this test
result.

## Raw artifact

`experiments/defectoscafeverde-dvsr1-locked-test-v1/DVSR1_locked_test_result.json`
