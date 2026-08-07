# Faruq-v3 Multilevel Fusion CM512 Result

Date: 2026-08-02

Decision: **PASS — authorize a capacity-matched multilevel-head protocol**

The audit reused the frozen predicted-ROI caches. It performed no detector
inference, detector training, rank sweep, or test access.

## Result

| Representation | Dimensions | Train Macro-F1 | Validation Macro-F1 | Balanced accuracy | Bottom-3 F1 | Worst F1 | Top-3 accuracy | Generalization gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P5 PCA-512 | 512 | 93.22% | 73.69% | 73.70% | 60.23% | 57.14% | 89.33% | 19.53 pp |
| **P3+P4+P5 PCA-512** | **512** | **94.50%** | **80.14%** | **80.21%** | **67.93%** | **65.12%** | **91.62%** | **14.37 pp** |

Fusion deltas over the capacity-matched P5 control:

- Macro-F1: **+6.45 percentage points**;
- bottom-3 F1: **+7.70 points**;
- worst-class F1: **+7.97 points**;
- top-3 accuracy: **+2.29 points**;
- generalization gap: **5.16 points smaller**.

Every frozen decision criterion passed. The result is stronger than the raw
896-versus-512 comparison because both alternatives use exactly 512 projected
features and the same PCA/ridge pipeline.

## Bottom classes of fusion CM512

| Class | Support | F1 |
|---|---:|---:|
| biji_muda | 25 | 65.12% |
| kulit_tanduk_ukuran_sedang | 24 | 68.09% |
| kulit_kopi_ukuran_sedang | 25 | 70.59% |
| biji_hitam | 25 | 72.00% |
| kulit_kopi_ukuran_kecil | 26 | 73.08% |
| biji_berlubang_lebih_satu | 24 | 73.91% |
| biji_bertutul_tutul | 25 | 74.07% |
| kulit_tanduk_ukuran_besar | 26 | 74.51% |
| biji_berkulit_tanduk | 25 | 77.55% |
| biji_berlubang_satu | 25 | 78.43% |

## Conclusion

1. Fine-grained class information is distributed across P3, P4, and P5.
2. The fusion advantage survives predicted boxes, so it is not an artifact of
   perfect ground-truth regions.
3. The fusion advantage survives exact 512-dimensional capacity matching, so
   it is not explained by the earlier 896-dimensional descriptor alone.
4. The strongest remaining weak class is `biji_muda`, consistent with the
   prior visual-identifiability evidence.
5. A controlled multilevel classification head is now scientifically
   justified. P2-only, global reranking, ontology, and unrestricted
   high-dimensional alternatives remain rejected.

## Authorization boundary

PASS authorizes the next design step only:

- define a single end-to-end P3+P4+P5 classification branch;
- keep the D0 regression/localization path unchanged;
- use a P5-only 512-dimensional branch as the capacity-matched control;
- perform a static parameter, state-schema, output-contract, and latency audit;
- freeze loss, assignment, schedule, seed, and validation gate before training.

It does not yet authorize training, additional seeds, or test access.

## Raw artifact

`experiments/faruq-v3-fusion-cm512-v1/fusion_cm512.json`
