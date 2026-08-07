# Faruq-v3 — Final ACMC Optimization Result

**Date:** 2026-08-07  
**Dataset:** Faruq-v3 grouped development dataset  
**Selected model:** **ACMC1**  
**Development status:** **MODEL OPTIMIZATION STOPPED**  
**Test status:** **LOCKED — NOT OPENED**

## 1. Purpose of this record

This document freezes the experimental findings that led to the final model-selection decision for the Faruq-v3 coffee-bean detection study. It records the matched controls, ACMC1, ACMC2, the ACMC1 residual-error audit, and the final ACMC1-HCR screening.

The decision rule is methodological rather than post-hoc: failed candidates are not rescued by retuning after their frozen screening/confirmation gates fail.

---

## 2. Experimental protocol kept fixed

- Validation is used for model development and model selection.
- Test remains locked during development.
- Seeds used for paired confirmation: `42`, `123`, `2026`.
- Main validation metrics:
  - Macro mAP50–95
  - Bottom-3 class mAP50–95
  - Worst-class mAP50–95
- `D0FT` is the matched extra-training control.
- Candidate effect is interpreted relative to `D0FT`, not only relative to stock `D0`.
- No test result is used to select ACMC1, ACMC2, or ACMC1-HCR.

---

## 3. ACMC1 paired three-seed confirmation

ACMC1 uses ambiguity-conditioned multilevel classification correction while preserving the native YOLO26 localization branch.

### Aggregate three-seed result

| Metric | D0 mean | D0FT mean | ACMC1 mean | ACMC1 − D0FT |
|---|---:|---:|---:|---:|
| Macro mAP50–95 | 80.12% | 86.62% | **87.62%** | **+1.00 pp** |
| Bottom-3 class mAP50–95 | 66.58% | 76.58% | **79.13%** | **+2.56 pp** |
| Worst-class mAP50–95 | 60.18% | 73.05% | **76.30%** | **+3.24 pp** |

ACMC1 improved Macro mAP50–95 over D0FT in all three seeds. Tail gains were positive in mean, although Bottom-3 and Worst were not improved in every single seed.

**Decision:** ACMC1 passed the paired confirmation and became the selected model.

---

## 4. ACMC2: entropy + top-two margin gate

ACMC2 tested whether adding top-two class-margin uncertainty to entropy improved ACMC1.

### Aggregate three-seed result

| Metric | D0 mean | D0FT mean | ACMC1 mean | ACMC2 mean | ACMC2 − D0FT | ACMC2 − ACMC1 |
|---|---:|---:|---:|---:|---:|---:|
| Macro mAP50–95 | 80.12% | 86.62% | **87.62%** | 87.56% | +0.94 pp | **−0.06 pp** |
| Bottom-3 class mAP50–95 | 66.58% | 76.58% | 79.13% | **79.37%** | +2.80 pp | +0.24 pp |
| Worst-class mAP50–95 | 60.18% | 73.05% | 76.30% | **77.23%** | +4.17 pp | +0.93 pp |

### Frozen ACMC2 progression outcome

All frozen criteria passed except:

- `macro_mean_not_lower_than_acmc1 = False`

ACMC2 Macro mean was 87.56% versus ACMC1 87.62%.

**Decision:** `FAIL`  
**Next action:** `KEEP_ACMC1_AS_SELECTED_MODEL`

Interpretation: margin uncertainty improved some tail behavior but did not improve the overall Macro mean relative to entropy-only ACMC1.

---

## 5. ACMC1 residual-error attribution audit

A validation-only, three-seed residual-error audit was run on selected ACMC1 checkpoints. No training was executed and test remained unopened.

### Global aggregate

| Metric | Mean |
|---|---:|
| Precision | 84.30% |
| Recall | 83.82% |
| mAP50 | 89.99% |
| mAP50–95 | 87.62% |
| Detection accessibility @ IoU 0.50 | **94.55%** |
| Matched recall @ IoU 0.50 | **94.55%** |
| Class accuracy given IoU 0.50 match | **85.20%** |
| Classification headroom @ IoU 0.50 | **14.80%** |

### Frozen attribution counts across 21 classes

- `classification_or_ranking_limited`: **12**
- `detection_or_confidence_limited`: **1**
- `no_single_dominant_signal`: **8**

This supports the working conclusion that the dominant residual bottleneck for the hard classes is fine-grained classification/ranking rather than failure to localize an available object at IoU 0.50.

### Important hard-class examples

| Class | mAP50–95 | Detection accessibility @ IoU50 | Class accuracy given IoU50 match | Classification headroom |
|---|---:|---:|---:|---:|
| kulit_tanduk_ukuran_sedang | 77.37% | 90.28% | 74.17% | 25.83% |
| kulit_tanduk_ukuran_besar | 79.69% | 96.15% | 77.31% | 22.69% |
| biji_berkulit_tanduk | 84.17% | 96.00% | 82.16% | 17.84% |
| biji_normal | 84.28% | 91.03% | 83.19% | 16.81% |
| biji_hitam_sebagian | 84.67% | 93.06% | 77.43% | 22.57% |
| biji_bertutul_tutul | 86.10% | 88.00% | 75.51% | 24.49% |
| biji_berlubang_satu | 86.30% | 97.33% | 72.44% | 27.56% |
| kulit_kopi_ukuran_sedang | 86.67% | 93.33% | 69.93% | **30.07%** |

### Largest directional confusions across three seeds

| Expected | Predicted | Count |
|---|---|---:|
| kulit_kopi_ukuran_sedang | kulit_kopi_ukuran_kecil | 14 |
| kulit_tanduk_ukuran_besar | biji_berkulit_tanduk | 14 |
| kulit_tanduk_ukuran_sedang | kulit_tanduk_ukuran_kecil | 14 |
| kulit_kopi_ukuran_besar | kulit_kopi_ukuran_sedang | 13 |
| biji_berlubang_satu | biji_muda | 11 |
| biji_normal | biji_berlubang_satu | 11 |
| biji_berkulit_tanduk | kulit_tanduk_ukuran_besar | 10 |
| biji_bertutul_tutul | biji_muda | 9 |
| biji_coklat | biji_hitam | 8 |
| biji_muda | biji_hitam | 8 |
| biji_berlubang_lebih_satu | biji_berlubang_satu | 7 |

The residual errors therefore show structured fine-grained confusion rather than random class errors.

### High-IoU note

Several otherwise well-classified classes show a large AP75→AP95 drop. This indicates that very-strict box precision can still limit AP at IoU 0.95 for some classes, but this was not the dominant bottleneck targeted by ACMC and did not justify changing the native localization branch during this development stage.

---

## 6. ACMC1-HCR: hard-competitor ranking screening

The residual audit motivated one final targeted test: **ACMC1-HCR**.

ACMC1-HCR preserves the ACMC1 inference architecture and adds a training-only hard-competitor pairwise ranking term on the one-to-one classification branch. The strongest wrong class is selected dynamically from the current logits; no validation-derived confusion pair is hard-coded into the model.

The frozen screening used seed 42 only.

### Seed-42 result

| Model | Macro mAP50–95 | Bottom-3 class mAP50–95 | Worst-class mAP50–95 |
|---|---:|---:|---:|
| D0 | 79.97% | 68.72% | 65.09% |
| D0FT | 86.69% | 74.98% | 72.02% |
| **ACMC1** | **87.62%** | **80.40%** | **79.49%** |
| ACMC1H | 87.01% | 79.69% | 76.23% |

### ACMC1-HCR deltas

Against D0FT:

- Macro: **+0.319 pp**
- Bottom-3: **+4.711 pp**
- Worst: **+4.208 pp**

Against ACMC1:

- Macro: **−0.617 pp**
- Bottom-3: **−0.713 pp**
- Worst: **−3.259 pp**

### Frozen HCR screening criteria

| Criterion | Result |
|---|:---:|
| Macro gain over ACMC1 ≥ +0.20 pp | FAIL |
| At least one tail gain over ACMC1 ≥ +0.50 pp | FAIL |
| Neither tail drops >0.50 pp vs ACMC1 | FAIL |
| Macro gain over D0FT remains ≥ +0.50 pp | FAIL |

**Decision:** `FAIL`  
**Next action:** `STOP_OPTIMIZATION_KEEP_ACMC1`

Seed 123 and seed 2026 must **not** be run for ACMC1-HCR because the frozen seed-42 screening gate already failed.

---

## 7. Final development decision

The experimental sequence is now frozen as:

```text
D0
  → D0FT matched-control
  → ACMC1: PASS, selected
  → ACMC2 entropy+margin: FAIL progression gate
  → ACMC1 residual-error audit: classification/ranking remains dominant in hard classes
  → ACMC1-HCR hard-competitor ranking: FAIL seed-42 screening
  → STOP MODEL OPTIMIZATION
```

### Final selected development model

**ACMC1**

Rationale:

1. ACMC1 consistently improves Macro mAP50–95 over the matched D0FT control across three seeds.
2. ACMC1 produces substantial mean improvements on Bottom-3 and Worst-class metrics.
3. ACMC2 adds uncertainty complexity but does not improve Macro mean over ACMC1.
4. The residual audit confirms that remaining errors are predominantly structured fine-grained classification/ranking errors for hard classes.
5. A directly targeted hard-competitor ranking loss worsens Macro, Bottom-3, and Worst performance relative to ACMC1 in the frozen seed-42 screening.
6. Further tuning after these failed frozen gates would risk validation chasing rather than hypothesis-driven optimization.

Therefore no additional ACMC3/HCR retuning, generic attention module, backbone replacement, localization-branch modification, or new module stacking is authorized under the current development protocol.

---

## 8. Test-lock status

At the time of this record:

- ACMC1 test evaluation has **not** been opened.
- ACMC2 test evaluation has **not** been opened.
- ACMC1-HCR test evaluation has **not** been opened.
- ACMC1-HCR seed123/seed2026 confirmation has **not** been authorized.

The next legitimate stage, after freezing the final analysis/protocol artifacts, is a **single locked-test evaluation of the selected ACMC1 model**. No development decision may be changed using the test result.

---

## 9. Primary experiment artifacts

- ACMC1 paired confirmation:
  - `experiments/faruq-v3-acmc-paired-confirmation-v1/`
- ACMC2 paired confirmation summary:
  - `experiments/faruq-v3-acmc2-paired-confirmation-v1/val_reports/acmc2_paired_optimization_confirmation.json`
- ACMC1 residual-error audit:
  - `experiments/faruq-v3-acmc1-residual-error-audit-v1/acmc1_residual_error_attribution_v2.json`
  - `experiments/faruq-v3-acmc1-residual-error-audit-v1/acmc1_residual_error_attribution_v2.csv`
- ACMC1-HCR seed-42 screening:
  - `experiments/faruq-v3-acmc1h-hard-competitor-screening-v1/val_reports/acmc1h_seed42_screening.json`

This document records the model-development decision only; raw checkpoint and evaluator artifacts remain the primary numerical evidence.
