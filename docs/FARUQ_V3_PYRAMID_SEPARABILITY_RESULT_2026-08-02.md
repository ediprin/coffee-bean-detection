# Faruq-v3 Pyramid Feature Separability Result

Date: 2026-08-02

Status: **PASS — authorize a multilevel classification protocol only**

Detector training was not executed, validation was accessed, and test remained
unavailable and locked. The closed-form probes were fitted on grouped
Faruq-v3 train features and evaluated on validation features.

## Result

| Representation | Dimensions | Train Macro-F1 | Validation Macro-F1 | Balanced accuracy | Bottom-3 F1 | Worst F1 | Top-3 accuracy | Generalization gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 | 128 | 81.18% | 72.00% | 72.36% | 52.96% | 41.03% | 90.87% | 9.18 pp |
| P4 | 256 | 85.72% | 73.05% | 73.16% | 58.33% | 54.55% | 91.83% | 12.67 pp |
| P5 | 512 | 92.84% | 73.08% | 73.18% | 56.36% | 48.89% | 90.30% | 19.77 pp |
| P3+P4 | 384 | 91.29% | 74.79% | 74.92% | 58.20% | 51.16% | 92.21% | 16.50 pp |
| **P3+P4+P5** | **896** | **98.49%** | **79.08%** | **79.13%** | **67.91%** | **65.12%** | **92.21%** | **19.41 pp** |

The best single level was P5 at 73.08% validation Macro-F1. P3+P4+P5
improved it by **6.00 percentage points**, while also improving bottom-3 F1
from 56.36% to 67.91%. This exceeds the frozen two-point fusion gate and does
not sacrifice the lower tail.

P3 alone was not the answer. It trailed the best deeper level by 1.07 points,
so the result rejects a high-resolution-only explanation and supports
complementary information across the complete pyramid.

## Bottom classes of the best representation

| Class | Support | F1 |
|---|---:|---:|
| biji_muda | 25 | 65.12% |
| kulit_kopi_ukuran_kecil | 26 | 69.23% |
| biji_hitam | 25 | 69.39% |
| biji_berlubang_satu | 25 | 70.59% |
| kulit_tanduk_ukuran_besar | 26 | 71.70% |
| kulit_kopi_ukuran_sedang | 25 | 72.00% |
| kulit_tanduk_ukuran_sedang | 24 | 73.47% |
| biji_bertutul_tutul | 25 | 74.51% |
| biji_berlubang_lebih_satu | 24 | 76.00% |
| biji_normal | 26 | 76.36% |

Unlike the raw leaf-rank audit, the complete pyramid representation preserves
useful signal even in its lower tail. The weakest class remains `biji_muda`,
consistent with the earlier visual-identifiability audit.

## Mechanistic conclusion

1. D0 already contains useful class information distributed across P3, P4,
   and P5.
2. No single pyramid level provides the same separation. P5 supplies the best
   single-level signal, while P3 and P4 add complementary detail.
3. The weak raw candidate ranks therefore do not prove that the backbone lacks
   all fine-grained information. They show that the stock detection
   classification path does not exploit the distributed signal effectively.
4. A multilevel classification protocol is now justified. A P2-only,
   high-resolution-only, global reranking, or ontology route is not justified
   by this result.

## Limits

- This is a ground-truth ROI linear-probe audit, not detector mAP.
- The probe sees accurate boxes; predicted-box noise and assignment behavior
  are not represented.
- The 19.41-point train-to-validation gap signals meaningful overfitting risk.
- The 896-dimensional fused descriptor has greater capacity than every single
  descriptor. A model experiment therefore requires a capacity-matched control.
- PASS authorizes protocol design and static controls only. It does not yet
  authorize detector training, extra seeds, or test access.

## Required next control

Before training a detector, freeze a predicted-ROI transfer audit using the
same D0 checkpoint:

- fit identical P5 and P3+P4+P5 probes on class-agnostically matched predicted
  train ROIs;
- evaluate them on matched validation ROIs;
- compare against their ground-truth ROI results;
- proceed only if the fusion advantage remains at least two points and its
  bottom-3 F1 is not lower than the P5 control.

This control tests whether the six-point fusion gain survives the box noise
that an end-to-end detector must handle.

## Raw artifact

`experiments/faruq-v3-pyramid-separability-v1/pyramid_separability.json`
