# Faruq-v3 Label Identifiability Result

Date: 2026-08-02  
Protocol: `faruq-v3-label-identifiability-v1`  
Training executed: no  
Test images accessed: no

## Decision

**DATA_OR_SCALE_LIMITED.** Do not implement a global geometry-conditioned
classification head before the `kulit_tanduk` labels and source scale are
audited or repaired.

| Split | Family | Small / medium / large | Best feature | Macro order AUROC | Medians small / medium / large | Signal |
|---|---|---:|---|---:|---|---|
| Train | kulit_kopi | 150 / 141 / 144 | normalized area | 0.889 | 0.00333 / 0.00567 / 0.00912 | strong |
| Train | kulit_tanduk | 142 / 136 / 146 | normalized area | 0.577 | 0.00604 / 0.00803 / 0.00697 | weak |
| Train | tanah_batu_ranting | 144 / 138 / 140 | normalized area | 0.847 | 0.00404 / 0.01065 / 0.02196 | strong |
| Validation | kulit_kopi | 26 / 25 / 25 | normalized area | 0.853 | 0.00376 / 0.00534 / 0.00931 | strong |
| Validation | kulit_tanduk | 25 / 24 / 26 | normalized area | 0.572 | 0.00615 / 0.00769 / 0.00700 | weak |
| Validation | tanah_batu_ranting | 26 / 24 / 25 | normalized area | 0.881 | 0.00398 / 0.01069 / 0.02307 | strong |

The result is stable across development splits. `Kulit_kopi` and
`tanah_batu_ranting` preserve the expected small < medium < large order, while
`kulit_tanduk` does not: its large median is smaller than its medium median in
both train and validation. A network cannot reliably learn a global ordinal
size rule from this image-space signal.

## D0 confusion taxonomy

Among the top directional confusion counts:

- local-defect similarity: 45;
- within-family size: 25;
- cross-family or material: 16.

The largest single size error is `kulit_tanduk_ukuran_kecil` predicted as
`kulit_tanduk_ukuran_sedang` (16). The local-defect group is larger overall,
including `biji_muda` versus `biji_bertutul_tutul` and the one/multiple-hole
classes versus `biji_bertutul_tutul`.

## Consequence

The dataset is not globally unusable, and this result does not reduce every
failure to annotation quality. Two size families contain strong geometry, and
45 top-pair errors are local visual defects rather than size labels. It does
show that a single geometry module cannot be justified for all 21 labels.

Before another model experiment:

1. inspect paired contact sheets for all three `kulit_tanduk` size labels;
2. verify whether the SNI size definition is physical and whether the images
   contain a stable scale reference;
3. repair inconsistent labels or add calibration metadata when possible;
4. separately audit the local-defect pairs for label consistency and visible
   evidence.

Only after these checks may a classification refinement protocol be frozen.

