# Coffee Standard v8 External Robustness Result

Date: 2026-08-16

## Dataset provenance and audit

The external source is Roboflow Universe project
`tes-rcphs/coffee-detection-with-standard`, version 8, exported as YOLOv8.
The public export contained 993 images, 13,926 boxes, and 25 labels.

The original Roboflow split is **not valid for model comparison**:

- 0 exact byte-identical image groups crossed splits;
- 14 Roboflow parent identities crossed train/validation/test;
- some parents had up to 48 augmented siblings distributed across splits;
- several labels were absent or nearly absent from validation or test;
- no malformed YOLO label rows were found.

Therefore no reported result below uses the original Roboflow split.

## Leakage-safe external benchmark

Only one deterministic representative was selected per Roboflow parent. Seven
labels without a direct SNI-21 equivalent were excluded rather than force
mapped: moldy bean, silver-skin bean, sour bean, gravel, and three twig-size
labels. Eighteen directly equivalent SNI classes remained.

| Item | Value |
|---|---:|
| Source derivative images | 993 |
| Source parent identities | 267 |
| Selected independent identities with mapped boxes | 148 |
| Evaluated instances | 3,989 |
| Directly mapped classes | 18 |
| Training on target data | No |

The three combined `tanah_batu_ranting_*` labels have no direct target ground
truth and are excluded from the metric aggregation. This is an external
post-hoc diagnostic, not a replacement locked test.

## Eleven-model seed-42 evaluation

| Model | Macro mAP50-95 | Bottom-3 | Worst | Recall | Macro delta vs D0FT |
|---|---:|---:|---:|---:|---:|
| AF2 | **17.06%** | **0.65%** | 0.07% | 29.83% | **+4.47** |
| CPE0 | 15.82% | 0.49% | **0.09%** | 30.44% | +3.23 |
| CPE7 | 15.21% | 0.27% | 0.00% | 30.01% | +2.62 |
| ACMC1 | 14.52% | 0.43% | 0.01% | 29.60% | +1.93 |
| STB1 | 14.45% | 0.36% | 0.01% | 30.23% | +1.86 |
| AF1 | 14.31% | 0.22% | 0.00% | 25.48% | +1.72 |
| IGEM1 | 14.04% | 0.39% | 0.00% | 30.19% | +1.45 |
| LPS1 | 14.04% | 0.29% | 0.01% | **30.46%** | +1.45 |
| GEO1 | 13.92% | 0.45% | 0.00% | 27.75% | +1.33 |
| SAF1 | 13.77% | 0.36% | 0.03% | 29.41% | +1.18 |
| D0FT | 12.59% | 0.32% | 0.01% | 26.33% | control |

All architectural candidates increased seed-42 Macro over D0FT, but absolute
lower-tail AP remained near zero. The benchmark therefore diagnoses severe
cross-dataset shift; it does not demonstrate deployment-ready target-domain
performance.

## AF2 versus D0FT paired three-seed confirmation

No model was trained or tuned on Coffee Standard. Existing seed 42, 123, and
2026 checkpoints were evaluated once.

| Metric | D0FT mean ± SD | AF2 mean ± SD | Paired delta mean ± SD | Minimum delta | Improved seeds |
|---|---:|---:|---:|---:|---:|
| Macro mAP50-95 | 11.43 ± 1.18% | **15.51 ± 1.76%** | **+4.08 ± 0.62** | +3.36 | 3/3 |
| Bottom-3 | 0.25 ± 0.22% | **0.35 ± 0.28%** | +0.10 ± 0.24 | -0.14 | 2/3 |
| Worst class | 0.01 ± 0.02% | **0.04 ± 0.03%** | +0.02 ± 0.04 | approximately 0.00 | 2/3 |

Decision: **PASS** under the frozen post-hoc gate. AF2 improved Macro in every
seed and preserved mean lower-tail behavior.

## Consolidated interpretation

- In-domain Faruq validation: AF2 remains the descriptive Macro leader among
  confirmed three-seed architectural candidates (87.94%, +1.32 points over
  D0FT, improved 3/3 seeds).
- Coffee Standard external: AF2 improved Macro by +4.08 points on average,
  improved 3/3 seeds, without target-domain training.
- Adrian external: AF2 was the best tested direction, but the dataset has only
  eight parent identities and all models collapsed; treat that evidence as
  weak.
- Synthetic density: SAF1 and IGEM1 were the strongest average directions;
  AF2 improved 3/4 conditions but was not the synthetic-density leader.

The defensible claim is that AF2 improves **target-free cross-dataset
robustness directionally and consistently**, not that it solves the target
domain. Absolute Coffee Standard lower-tail AP remains inadequate.

## Authoritative Drive artifacts

- `evidence/coffee-detection-with-standard-v8/dataset_audit.json`
- `evidence/coffee-detection-with-standard-v8/external_benchmark_summary.json`
- `experiments/coffee-standard-v8-retained-external-v1/coffee_standard_retained_external_summary.json`
- `experiments/coffee-standard-v8-af2-paired-v1/coffee_standard_af2_paired_summary.json`

