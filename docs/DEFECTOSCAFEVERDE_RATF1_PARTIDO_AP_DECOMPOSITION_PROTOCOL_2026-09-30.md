# DefectosCafeVerde RATF1 `partido` AP-Decomposition Protocol

Date frozen: 2026-09-30

## Motivation

The first root-cause audit showed that all 131 validation `partido` instances
remain localizable at IoU 0.50. RATF1 also improved final localization-
conditioned top-1 accuracy from 97.71% to 99.24%, although its `partido`
AP50-95 fell by 2.15 percentage points. Consequently, the regression cannot be
attributed simply to missing objects or a loss of final top-1 recognition.

This second audit isolates whether the AP loss is associated with confidence
ranking, stricter-IoU box quality, duplicate/false-positive behavior, the
inference-time RATF class residual, or the representation learned during
training.

## Frozen endpoints

- `D0DIRECT`: completed seed-42, 50-epoch matched native checkpoint.
- `RATF1_ACTIVE`: completed seed-42, 50-epoch RATF1 checkpoint.
- `RATF1_ZERO_RESIDUAL`: the same RATF1 checkpoint with only every
  `class_residual` weight and bias set to zero in memory.

The zero-residual endpoint is an inference intervention, not a trained model.

## Evaluation

One validation dataloader pass evaluates all three endpoints. The audit uses
the same Ultralytics-compatible class-aware matching and ten IoU thresholds
(0.50 through 0.95) as the existing calibrated evaluator. It reports:

- `partido` AP50, AP75, AP50-95, and AP at every IoU threshold;
- endpoint precision, recall, F1, and best-F1 confidence;
- TP/FP/FN counts at IoU 0.50 and 0.75;
- matched-box IoU and TP/FP confidence summaries;
- false positives categorized as target duplicates, overlap with another
  ground-truth class, or background/localization errors;
- full validation Macro, Bottom-3, and Worst-class calibration against the
  frozen D0DIRECT and RATF1 result documents.

## Interpretation rules

- Recovery after zeroing the residual implicates the inference-time residual;
  incomplete recovery indicates a mixed residual and learned-training-path
  effect.
- Preserved AP50 with degraded high-IoU AP implicates high-IoU localization or
  ranking rather than basic object accessibility.
- No recovery after zeroing, together with AP50 degradation, implicates the
  learned training path, confidence ranking, duplicates, or false positives.

These are diagnostic attributions, not new model-performance claims.

## Locks

- Validation only; test is not extracted or accessed.
- No parameter update, training, tuning, or checkpoint write.
- Dataset audit, endpoint result contracts, checkpoint SHA256 values, 12-class
  ontology, and historical endpoint metrics must pass before attribution.
- A failed calibration gate invalidates the audit rather than silently
  producing a conclusion.
