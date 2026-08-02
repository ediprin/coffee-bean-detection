# Faruq-v3 Predicted-ROI Multilevel Transfer Result

Date: 2026-08-02

Formal decision: **FAIL — PCA-128 absolute Macro-F1 below the frozen gate**

Mechanistic result: **the relative multilevel advantage survives predicted
boxes and capacity matching**.

The D0 detector remained frozen. Closed-form probes and train-only PCA were
fitted on grouped train data, validation was evaluated, and test remained
unavailable and locked.

## Result

| Representation | Dimensions | Train Macro-F1 | Validation Macro-F1 | Balanced accuracy | Bottom-3 F1 | Worst F1 | Top-3 accuracy | Generalization gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P5 raw | 512 | 92.86% | 73.37% | 73.53% | 56.96% | 54.05% | 89.52% | 19.49 pp |
| P3+P4+P5 raw | 896 | 98.58% | 78.85% | 78.90% | 67.47% | 63.64% | 92.38% | 19.73 pp |
| P5 PCA-128 | 128 | 77.91% | 67.50% | 67.94% | 47.99% | 45.00% | 86.48% | 10.41 pp |
| P3+P4+P5 PCA-128 | 128 | 81.96% | 70.98% | 71.20% | 52.90% | 50.00% | 91.43% | 10.98 pp |

Matched-ROI coverage was 99.83% on train and 99.81% on validation.

## Frozen-gate accounting

Passed:

- raw fusion Macro-F1 gain: **+5.48 points**;
- raw bottom-3 preserved: 67.47% versus 56.96%;
- capacity-matched fusion gain: **+3.48 points**;
- capacity-matched bottom-3 preserved: 52.90% versus 47.99%;
- capacity-matched bottom-3 above 50%: 52.90%;
- ground-truth-to-predicted fusion Macro-F1 retention: **99.71%**;
- train and validation coverage above 90%.

Failed:

- capacity-matched fusion absolute Macro-F1: **70.98%**, below the frozen 75%
  threshold.

The originally emitted action
`STOP_FUSION_ADVANTAGE_NOT_ROBUST` was semantically inaccurate: every explicit
relative-robustness criterion passed. The decision remains FAIL because the
gate was frozen, but the correct attribution is
`STOP_CAPACITY_MATCHED_ABSOLUTE_MACRO_BELOW_GATE`. The implementation was
corrected without changing a threshold or metric after observing the result.

## Bottom classes of capacity-matched fusion

| Class | Support | F1 |
|---|---:|---:|
| biji_muda | 25 | 50.00% |
| biji_berlubang_satu | 25 | 52.17% |
| biji_hitam | 25 | 56.52% |
| kulit_tanduk_ukuran_sedang | 24 | 57.14% |
| biji_hitam_sebagian | 24 | 65.31% |
| biji_coklat | 25 | 65.38% |
| biji_bertutul_tutul | 25 | 65.45% |
| kulit_kopi_ukuran_kecil | 26 | 67.86% |
| biji_berlubang_lebih_satu | 24 | 68.09% |
| kulit_kopi_ukuran_sedang | 25 | 68.09% |

## Interpretation

1. Box noise is not responsible for the ground-truth ROI fusion result. Raw
   fusion retains 99.71% of its ground-truth ROI Macro-F1.
2. Additional descriptor dimensionality is not the entire explanation. At the
   same 128 dimensions, multilevel fusion still improves Macro-F1 by 3.48
   points and bottom-3 F1 by 4.91 points.
3. The 128-dimensional bottleneck removes substantial signal from both
   descriptors. It reduces P5 from 73.37% to 67.50% and fusion from 78.85% to
   70.98%.
4. Therefore this result does not support an unrestricted 896-dimensional head,
   but it also does not falsify multilevel complementarity.
5. No detector training, extra seeds, or test access is authorized by the
   frozen gate.

## Methodological consequence

The clean next capacity control, if pursued, is not a validation sweep. It is a
single preregistered comparison at **512 dimensions**, because 512 is the native
dimension of the P5 control:

- P5 raw: 512 dimensions;
- P3+P4+P5 projected by train-only PCA: 512 dimensions.

This compares both representations at the exact capacity of the strongest
single-level descriptor without compressing P5 below its native size. It must
be frozen as one control, not searched over multiple ranks.

## Raw artifact

`experiments/faruq-v3-predicted-roi-transfer-v1/predicted_roi_transfer.json`
