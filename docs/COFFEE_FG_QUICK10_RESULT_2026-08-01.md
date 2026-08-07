# CoffeeFG-YOLO26 Quick-10 Result — 2026-08-01

## Status

**STOP.** CoffeeFG ROI refinement failed the frozen one-seed validation gate.
Do not run the 50-epoch schedule, additional seeds, or the locked test split.

This is a negative screening result, not final test evidence.

## Frozen setup

- Dataset: real A0 SNI-21 development data.
- Train: 8,011 images and 20,959 boxes.
- Validation: 416 images and 4,969 boxes.
- Classes: 21.
- Image size: 640.
- Seed: 42.
- Schedule: quick-10, 10 epochs per model.
- Evaluation: validation only; test remained locked.
- Output: `Coffee_Bean_Detection/experiments/coffee-fg-quick10-v1`.

Models:

| Code | Detector foundation | Refinement |
|---|---|---|
| D0Q | YOLO26n P3-P5 | none |
| D1Q | YOLO26n P2-P5 | none |
| R0Q | YOLO26n P3-P5 | first-order P3/P4 ROI refiner |
| R1Q | YOLO26n P3-P5 | capacity-matched bilinear P3/P4 ROI refiner |

## Foundation diagnostic

The validation diagnostic selected 500 candidates and reported:

- P2 accessibility gain: **-0.2616 percentage points**;
- recommended foundation: **D0/P3-P5**;
- proposal accessibility sufficient: **true**;
- oracle class headroom sufficient: **true**;
- P2 materially improves accessibility: **false**;
- classification refinement rational for screening: **true**.

Therefore P2 and the R2/R3 branch were stopped. Only R0Q/R1Q were authorized.

## Validation result

| Comparison | Macro AP50-95 | Bottom-3 AP50-95 | Worst AP50-95 | Decision |
|---|---:|---:|---:|---|
| D0Q -> R0Q | 40.78% -> 32.00% (-8.77 pp) | 10.09% -> 7.02% (-3.08 pp) | 0.00% -> 0.00% (-0.00 pp) | FAIL |
| D0Q -> R1Q | 40.78% -> 29.92% (-10.86 pp) | 10.09% -> 4.29% (-5.80 pp) | 0.00% -> 0.01% (+0.01 pp) | FAIL |
| R0Q -> R1Q | 32.00% -> 29.92% (-2.09 pp) | 7.02% -> 4.29% (-2.73 pp) | 0.00% -> 0.01% (+0.01 pp) | FAIL |

The `+0.01` percentage-point worst-class change is practically zero and does
not offset the large macro and lower-tail degradation.

## Interpretation and decision

The diagnostic established that localized candidates still contained class
headroom, but it did **not** establish that ROI refinement would exploit it.
Both authorized refiners degraded validation performance substantially.
The capacity-matched bilinear refiner was also worse than the first-order
control, so the result does not support a second-order interaction claim.

Frozen decision:

1. D0Q remains the best quick-10 model in this study.
2. P2 is rejected for this protocol.
3. First-order and bilinear ROI refinement are rejected.
4. Do not expand CoffeeFG to more seeds or test.
5. Preserve this result as evidence that classification headroom alone is not
   sufficient justification for a candidate-level ROI refiner.

## Artifact provenance

- Diagnostic: `val_reports/diagnostic_seed42.json`.
- Decision: `val_reports/coffee_fg_decision.json`.
- Per-model validation reports: `val_reports/D0Q_seed42_val.json`,
  `val_reports/R0Q_seed42_val.json`, and `val_reports/R1Q_seed42_val.json`.
- Test accessed: **false**.

