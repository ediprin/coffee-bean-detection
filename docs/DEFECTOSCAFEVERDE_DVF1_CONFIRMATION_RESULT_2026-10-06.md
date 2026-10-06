# DefectosCafeVerde DVF1 Three-Seed Confirmation Result

Completed: **2026-10-06**.

## Decision

**PASS — PROMOTE_DVF1_DUAL_VIEW_FUSION.**

The fixed 2,844-parameter DVF1 symmetric fuser improved the source paper's
maximum-confidence dual-view rule in all three fuser seeds. The seed-42 D0
detector evidence, train/validation pair caches, architecture, objective, and
100-epoch schedule were unchanged. The detector was not loaded or trained for
seeds 123 and 2026, and test was not accessed.

| Metric | Paper mean | DVF1 mean | DVF1 std | Mean delta | Minimum seed delta | Improved seeds |
|---|---:|---:|---:|---:|---:|---:|
| Physical-pair accuracy | 94.4149% | **95.4787%** | 0.2171% | **+1.0638 pp** | +0.7979 pp | 3/3 |
| Macro class accuracy | 92.8456% | **93.8504%** | 0.2094% | **+1.0048 pp** | +0.7497 pp | 3/3 |
| Bottom-3 class accuracy | 80.4473% | **83.3514%** | 0.0000% | **+2.9040 pp** | +2.9040 pp | 3/3 |
| Worst-class accuracy | 76.1905% | **76.1905%** | 0.0000% | **+0.0000 pp** | +0.0000 pp | 0/3; nonlower 3/3 |

## Per-seed endpoints

| Seed | Pair accuracy | Pair delta | Macro | Macro delta | Bottom-3 | Bottom-3 delta | Worst | Worst delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 95.4787% | +1.0638 pp | 93.8478% | +1.0022 pp | 83.3514% | +2.9040 pp | 76.1905% | +0.0000 pp |
| 123 | 95.2128% | +0.7979 pp | 93.5953% | +0.7497 pp | 83.3514% | +2.9040 pp | 76.1905% | +0.0000 pp |
| 2026 | 95.7447% | +1.3298 pp | 94.1082% | +1.2626 pp | 83.3514% | +2.9040 pp | 76.1905% | +0.0000 pp |

The paper endpoint was identical for all seeds: 94.4149% physical-pair,
92.8456% Macro, 80.4473% Bottom-3, and 76.1905% worst-class accuracy. The
non-deployable pair oracle remained 97.34% physical-pair accuracy. DVF1 thus
captured approximately 36% of the available pair-oracle headroom over the
paper rule.

## Frozen decision gates

All gates passed:

- mean physical-pair gain was at least 0.5 point;
- physical-pair accuracy improved in 3/3 seeds;
- mean Macro was not lower and improved in 3/3 seeds;
- mean Bottom-3 was not lower and improved in 3/3 seeds;
- mean worst-class accuracy did not decrease;
- all input reports retained `test_images_accessed=false`; and
- the aggregation step performed no training.

## Interpretation

DVF1 replaces the paper's fixed maximum-confidence selection with an
order-invariant learned fusion of the two native detector score vectors. It
keeps a native detector box, adds no detector or box regressor, and leaves the
seed-42 D0 detector frozen. The result supports promoting DVF1 as the selected
dual-view method: its average gain is positive, its lower-tail improvement is
identical across all three fuser seeds, and its worst class is never harmed.

The confirmation concerns **fuser optimization stability**, not detector-seed
or dataset-resampling stability. All three fusers use the same frozen D0
evidence and the same validation physical pairs. The result is therefore a
reused-validation confirmation and not a locked-test claim. It applies only
when both physical sides of a bean are available at inference.

## Artifacts

The raw arm reports, checkpoints, and aggregate decision are stored outside
Git under the shared Drive project:

`experiments/defectoscafeverde-dvf1-confirmation-v1/`

The aggregate report is:

`DVF1_three_seed_confirmation.json`

Test remained unopened.
