# Faruq-v3 Operational Audit Result — 2026-08-02

## Scope

- Frozen YOLO26n D0 seed-42 checkpoint.
- Faruq-v3 grouped validation: 294 images and 526 objects.
- No training and no test access.
- The corrected result was recomputed from stored v1 counts; inference was not
  repeated.

## Recorded correction

The v1 selector maximized correct-decision recall without penalizing false
positives. It therefore selected native output at confidence 0.01 and emitted
`postprocessing_sufficient=True`. That decision is invalid and must not be
cited.

Version 2 selects by correct-decision F1 and separates two questions:

1. whether post-processing improves the operating point; and
2. whether fine-grained classification error remains material.

## Corrected comparison

| Metric | Native, conf 0.25 | Class-agnostic NMS, conf 0.05 | Delta |
|---|---:|---:|---:|
| Proposal accessibility | 63.69% | 96.39% | +32.70 points |
| Correct-decision precision | 53.83% | 57.06% | +3.23 points |
| Correct-decision recall | 40.11% | 55.32% | +15.21 points |
| Correct-decision F1 | 45.97% | 56.18% | +10.21 points |
| Conditional top-1 class accuracy | 62.99% | 57.40% | -5.59 points |
| Predictions per image | 1.33 | 1.73 | +0.40 |

The selected operating point produced 291 correct decisions from 510 retained
detections for 526 ground-truth objects.

## Decision

```text
PASS_POSTPROCESSING_CLASSIFICATION_UNRESOLVED
```

Class-agnostic NMS plus a lower confidence threshold materially improves the
operational precision–recall balance. It does not solve classification: among
localized and matched objects, 42.60% classification-error headroom remains.
The evidence therefore supports both of the following statements:

- post-processing configuration was a real bottleneck in the default
  confidence-0.25 output;
- the classification scores remain the main unresolved fine-grained
  bottleneck after localization is made accessible.

The corrected raw report is stored outside Git at:

```text
Coffee_Bean_Detection/experiments/faruq-v3-yolo26n-baseline-v1/
val_reports/D0_seed42_operational_audit_corrected.json
```

