# Faruq-v3 Label Identifiability Audit Protocol

Version: 1.0.0  
Status: frozen before execution  
Scope: development train and validation only

## Question

Before modifying YOLO26 again, determine whether the SNI size labels
`kecil/sedang/besar` contain an observable image-space geometry signal. A model
cannot reliably recover physical size when camera scale and reference geometry
are absent or inconsistent.

## Inputs

- grouped Faruq-v3 development archive;
- completed D0 seed-42 validation diagnostic;
- train and validation annotations only;
- no checkpoint execution and no GPU training.

The test split must not exist in the extracted development dataset.

## Measurements

For the `kulit_kopi`, `kulit_tanduk`, and `tanah_batu_ranting` size families,
measure per split:

- normalized box area;
- normalized long side;
- box area relative to the within-image median;
- class medians for small, medium, and large;
- pairwise order AUROC for small < medium, medium < large, and small < large;
- macro order AUROC and strict median ordering.

The D0 directional confusion pairs are also categorized as within-family size,
local-defect similarity, or cross-family/material errors.

## Frozen interpretation

- `strong`: macro order AUROC >= 0.80 and strictly ordered medians;
- `partial`: macro order AUROC >= 0.65 without satisfying the strong rule;
- `weak`: macro order AUROC < 0.65;
- missing size levels: insufficient evidence.

If every validation family is strong, a lightweight geometry-conditioned
classification experiment is justified. If any family is weak or missing, the
next action is dataset/scale calibration rather than another neural module. A
mixture of partial signals requires contact-sheet and source-scale inspection
before a model change.

Bounding-box geometry remains a proxy, not a physical millimetre measurement.
This audit cannot establish final generalization and does not authorize test
access.

## Recorded outcome — 2026-08-02

The audit returned `DATA_OR_SCALE_LIMITED`. `Kulit_kopi` and
`tanah_batu_ranting` showed strong and split-consistent order signals, but
`kulit_tanduk` was weak in train (0.577 macro order AUROC) and validation
(0.572). Its large median normalized area was smaller than its medium median in
both splits. The confusion taxonomy contained 45 local-defect, 25
within-family-size, and 16 cross-family/material errors among the reported top
pairs. No training or test access occurred. See
`docs/FARUQ_V3_LABEL_IDENTIFIABILITY_RESULT_2026-08-02.md`.
