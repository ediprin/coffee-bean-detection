# Faruq-v3 AF2 vs CLAHE classical-enhancement control

Date frozen: 2026-08-23

Status: **PROTOCOL FROZEN BEFORE CLAHE RESULTS**

## Research question

Does the validated AF2 gain reflect information beyond a generic classical local-contrast enhancement effect?

This experiment is not authorized to tune CLAHE after observing validation performance.

## Frozen arms

All arms use the same Faruq-v3 grouped development split, YOLO26n-P3 detector, seed-matched D0 starting checkpoint, 50-epoch schedule, augmentations, optimizer policy, image size, batch size, and validation metrics.

1. `D0FT` — existing optimization-matched continued-training control.
2. `CLAHE_LAB` — classical enhancement control.
3. `AF2` — existing validated adaptive directional-frequency frontend.

### CLAHE_LAB transform

The primary classical control is transferred literally from the preprocessing configuration reported by Guruprakash et al. (2026), DeeppestNet:

- input RGB;
- convert RGB -> LAB;
- apply CLAHE **only to L (luminance)**;
- `clipLimit = 3.0`;
- `tileGridSize = 8 x 8`;
- merge unchanged A/B chroma channels;
- convert LAB -> RGB.

The transform is deterministic, contains no learnable parameters, and is applied on every model forward so training and validation see the same frontend placement as AF2.

Gamma correction is deliberately excluded from the primary experiment. Ruan et al. (2025) used CLAHE and Gamma correction jointly on LiDAR-derived low-light vehicle images, so including Gamma here would introduce a second treatment and a less direct domain transfer. It may be tested only as a separately frozen follow-up if the primary CLAHE result warrants it.

## Seeds and starting checkpoints

Frozen seeds: `42, 123, 2026`.

Each CLAHE run starts from the same seed-matched D0 checkpoint used to construct the corresponding D0FT/AF2 comparison lineage. No seed is added or removed after results are observed.

## Metrics

Primary validation metrics:

- Macro mAP50-95;
- Bottom-3 class mAP50-95;
- Worst-class mAP50-95.

For every metric, report all three per-seed values, paired deltas, mean, standard deviation, minimum paired delta, and number of seed wins.

## Frozen questions and decision rules

### Q1 — Does generic CLAHE enhancement help over continued training?

Compare `CLAHE_LAB - D0FT`.

`PASS` requires all of:

- Macro mean gain >= +0.50 point;
- Macro improves in at least 2/3 seeds;
- Bottom-3 mean is not lower;
- Worst-class mean drop is no worse than -1.00 point.

### Q2 — Does AF2 show advantage beyond this CLAHE baseline?

Compare `AF2 - CLAHE_LAB`.

`PASS` requires all of:

- AF2 Macro mean advantage >= +0.50 point;
- AF2 Macro wins in at least 2/3 seeds;
- AF2 Bottom-3 mean is not lower;
- AF2 Worst-class mean is not lower.

A Q2 PASS supports only the bounded claim that AF2 adds value beyond this frozen paper-derived CLAHE baseline on Faruq-v3 validation. It does not establish superiority over every possible image-enhancement method.

### Q3 — Does CLAHE clearly outperform AF2?

The symmetric `CLAHE_LAB - AF2` rule uses the same +0.50 Macro margin, 2/3 seed wins, and non-lower Bottom-3/Worst means.

If neither Q2 nor Q3 passes, the result is `NO_DIRECTIONAL_SUPERIORITY_ESTABLISHED`; do not call the methods equivalent without a dedicated equivalence design.

## Leakage and test policy

- Development `test` must not exist under the active data root.
- Dataset audit must pass before training.
- Existing AF2/D0FT three-seed validation evidence may be read as frozen references.
- The locked test is not opened, evaluated, or used for tuning.

## Implementation

- Config: `configs/classical/CLAHE_LAB_yolo26n.yaml`
- Frontend: `src/coffee_detector/classical_enhancement/`
- Runner: `src/coffee_detector/experiments/run_faruq_v3_af2_clahe_control.py`
- Colab: `notebooks/Faruq_V3_AF2_CLAHE_Control_Colab.ipynb`

No CLAHE parameter search is authorized under this protocol.
