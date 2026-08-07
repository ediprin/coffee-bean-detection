# SNI-21 held-out synthetic density benchmark

**Version:** 1.0
**Frozen:** 30 July 2026
**Status:** development benchmark setup; no training and no test access

## Research purpose

This benchmark diagnoses where a frozen detector fails as scene density rises:

1. proposal/localization accessibility;
2. fine-grained wrong-class errors after localization;
3. output-candidate saturation;
4. visibility/occlusion sensitivity.

It is not a substitute for an independent real dense benchmark.

## Identity boundary

Training uses only A0 real-train identities. Synthetic benchmark scenes use
only crop records with `generated_split=val`.

The setup must stop when:

- a selected crop or parent identity also appears in train or test metadata;
- the validation object library contains a non-validation asset;
- canonical SNI-21 names/order are incomplete;
- a reuse bound is exceeded.

Reading split and identity fields from `manifest.csv` is permitted for leakage
auditing. The setup never opens a test image or materializes a test scene.

Old A1/A2 scenes are excluded from evaluation because they use train
identities. They remain training arms, generator references, and placement
audit artifacts.

## Density ladder

| Code | Objects per scene | Role |
|---|---:|---|
| B0 | 1--5 | synthetic count-matched sparse diagnostic |
| B1 | 10--25 | low/medium density |
| B2 | 50--100 | high density |
| B3 | 220--300 | dense 300 g scene-count hypothesis |

Object scale is fixed across B0--B3. This isolates object count from scale.
Consequently B0 is count-matched, not guaranteed to match the real sparse
scale distribution. R0 real-validation remains the real anchor.

## Staged conditions

Generation is staged to avoid unnecessary work:

1. `core`: empirical source prior + mild visibility;
2. `clear`: empirical source prior + clear visibility comparator;
3. `balanced`: class-balanced + mild visibility diagnostic;
4. `severe`: empirical source prior + severe visibility stress test;
5. `all`: all conditions above.

“Empirical source prior” means the class distribution of the validation crop
package. It is not claimed to represent industrial prevalence.

Scenes generated with the same density and seed share the observable selection
and geometry plan. Visibility/placement policy is the intended difference.

Mild visibility is the primary occlusion condition. Severe visibility is a
stress test because whole-instance visibility cannot prove that the
class-defining defect remains visible.

## Reuse and uncertainty

Every scene stores all `source_asset_id` and `source_parent_id` values. Reuse
limits are computed before generation from class capacity and may be overridden
only before result inspection.

The setup writes `resampling_units.json` containing:

- scene bootstrap units;
- source-asset clusters;
- source-parent clusters.

Performance uncertainty is not computed during generation because predictions
do not yet exist. After inference, metric deltas must be recomputed with paired
scene bootstrap and grouped source-identity resampling/weighting. Repeated
placements are not independent samples.

## Evaluation matrix

The same frozen checkpoints are evaluated on R0 and generated B conditions:

```text
delta_real    = CoffeeFG(R0) - YOLO26(R0)
delta_density = CoffeeFG(Bk) - YOLO26(Bk)
```

For paired clear/mild conditions:

```text
occlusion_interaction =
  [CoffeeFG(mild) - YOLO26(mild)]
  - [CoffeeFG(clear) - YOLO26(clear)]
```

Required diagnostics:

- mAP50--95, mAP50, precision, and recall;
- macro, bottom-three, worst, and per-class AP;
- proposal accessibility versus K;
- localized classification accuracy and wrong-class count;
- missed and duplicate detections;
- count MAE, signed count bias, and exact-count accuracy;
- AP by object size, density, and visibility;
- `max_det` 100/300/500 only on B2/B3;
- sparse and dense latency reported separately.

## Setup command

Run only the first stage:

```bash
python -u -m coffee_detector.run_sni21_density_benchmark_setup \
  --crop-dataset-root /content/drive/MyDrive/coffee-sni-instance-crop-v1 \
  --output-root /content/drive/MyDrive/coffee-bean-detection/sni21-density-benchmark-v1 \
  --stage core \
  --scenes-per-condition 200 \
  --seed 42 \
  --shard-cache-root /content/sni21-shard-cache
```

This creates only validation-derived synthetic scenes. It does not train a
detector, run inference, or access a test image.

Do not generate later stages until the core audit and visual sheets are
accepted.

## Claim boundary

Allowed:

- controlled synthetic density/visibility diagnosis on held-out validation
  identities;
- localization versus classification bottleneck analysis;
- relative robustness of frozen checkpoints across synthetic density.

Not allowed:

- final test performance;
- real dense or conveyor performance;
- physical validation of a 300 g sample;
- treating repeated synthetic placements as independent real observations.
