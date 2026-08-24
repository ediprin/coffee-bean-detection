# Bab II Concrete Reference Pool

Purpose: turn the reference-diversity policy into a concrete, section-by-section source pool for drafting. This is a **routing document**, not the final bibliography.

The latest master workbook reports 32 core proposal references, 60 Coffee Atlas references, 20 core references mapped to Project PDFs, zero pending core index audits, and an 18-paper full-text-verified related-work shortlist. Use that workbook as the bibliographic/index authority; this file controls **where and why** sources are used in Bab II.

## Status vocabulary

- `CORE-VERIFIED` — source is in the core master map and suitable for proposal use within its recorded claim boundary.
- `PROJECT-FULLTEXT` — full text has been read/mapped in the project/conversation, but final page-level extraction may still be needed during prose drafting.
- `FOUNDATIONAL` — canonical paper/book/software specification; final edition/page or official source should be cited appropriately.
- `OPTIONAL` — useful for breadth, but only include if it supports a sentence that is otherwise under-evidenced.

---

## 2.1 Biji Kopi Hijau dan Cacat Fisik Biji Kopi

Target: 5–8 distinct references. Primary function: standard, taxonomy, physical-defect vocabulary. Avoid turning this into a detector-performance section.

| Key | Source | Status | Intended sentence role |
|---|---|---|---|
| STD-01 | BSN — SNI 01-2907-2008 Biji Kopi | CORE-VERIFIED | Indonesian green-coffee defect/grading terminology and standard context |
| COF-07 | Kesiman et al. 2023 | CORE-VERIFIED | SNI-aligned 17-defect dataset/taxonomy context |
| COF-08 | Arwatchananukul et al. 2024 | CORE-VERIFIED | Independent 17-class green-Arabica defect taxonomy |
| COF-02 | Bahy & Rifai 2026 | CORE-VERIFIED | 20-class SNI-oriented detection taxonomy; use only taxonomy-related parts here |
| COF-04 | Hebert & Alamsyah 2026 | CORE-VERIFIED | SCA-style defect categories; do not substitute for official SCA authority |
| COF-SUP-01 | Kesiman et al. 2024, Coffection | PROJECT-FULLTEXT | SNI grading/application context and practical taxonomy implementation |
| COF-10 | de Oliveira et al. 2016 | CORE-VERIFIED / metadata-primary | Historical physical/color/morphology inspection context |

Recommended core set in final prose: `STD-01 + COF-07 + COF-08`, then add 1–3 domain papers only where needed.

---

## 2.2 Inspeksi Mutu Biji Kopi: Metode Konvensional dan Tantangannya

Target: 4–7 distinct references. Primary function: manual/computer-vision transition, practical inspection constraints, not fine-grained theory.

| Key | Source | Status | Intended sentence role |
|---|---|---|---|
| REV-01 | Motta et al. 2024 — coffee ML review | CORE-VERIFIED | Historical landscape and discovery support |
| COF-10 | de Oliveira et al. 2016 | CORE-VERIFIED / metadata-primary | Handcrafted/computational-intelligence coffee inspection |
| COF-11 | Chang & Huang 2021 | CORE-VERIFIED / primary | Transition to deep-learning defect inspection |
| COF-14 | Muchtar et al. 2025 | CORE-VERIFIED | Practical edge/deployment motivation |
| COF-15 | Hsia et al. 2022 | CORE-VERIFIED | Lightweight/explainable coffee quality inspection |
| COF-01 | Hong et al. 2026 | CORE-VERIFIED | One modern paper may support manual-inspection limitations, but do not let it dominate this section |

Reference rotation rule: if `COF-01` is used here, its use in §2.4 should be for a different claim (improved YOLOv10 / research pattern).

---

## 2.3 Object Detection

Target: 4–6 distinct references. Coffee papers are not needed for the basic definition.

| Key | Source | Status | Intended sentence role |
|---|---|---|---|
| DET-02 | Ren et al. 2015 — Faster R-CNN | FOUNDATIONAL | Two-stage detector taxonomy/background |
| DET-03 | Redmon et al. 2016 — original YOLO | FOUNDATIONAL | One-stage/unified object-detection framing |
| EVAL-01 | Lin et al. 2014 — COCO | FOUNDATIONAL | Object-detection/localization benchmark context |
| DIAG-01 | Feng et al. 2021 — TOOD | PROJECT-FULLTEXT | Classification/localization task alignment |
| DIAG-02 | Wu et al. 2020 | PROJECT-FULLTEXT | Different representations can favor classification vs localization |
| DIAG-03 | Jiang et al. 2018 — IoU-Net | PROJECT-FULLTEXT | Classification confidence is not localization quality |
| COF-09 | Lei et al. 2025 | CORE-VERIFIED | Optional coffee-specific bridge after the general theory, not the definition itself |

---

## 2.4 YOLO (You Only Look Once)

Target: 5–8 references. Combine canonical YOLO with a small set of coffee applications rather than repeating every coffee paper.

| Key | Source | Status | Intended sentence role |
|---|---|---|---|
| DET-03 | Redmon et al. 2016 | FOUNDATIONAL | Original YOLO concept |
| DET-01 | Jocher et al. 2026 — YOLO26 | CORE-VERIFIED | Transition to the detector family used in this thesis; detailed architecture belongs in §2.5 |
| COF-06 | Gope et al. 2024 | CORE-VERIFIED | YOLO-family viability in green coffee, small taxonomy |
| COF-16 | Gope et al. 2025 | CORE-VERIFIED | Broader cross-family/YOLO coffee benchmark, six classes |
| COF-01 | Hong et al. 2026 | CORE-VERIFIED | Improved YOLOv10 coffee pivot and targeted modification pattern |
| COF-02 | Bahy & Rifai 2026 | CORE-VERIFIED | Lightweight YOLO with larger SNI-oriented taxonomy |
| COF-05 | Jundullah et al. 2026 | CORE-VERIFIED | Optional modern 20-class YOLO coffee example; if used strongly in §2.6, keep this mention brief |

Recommended structure: canonical YOLO → YOLO in coffee → contrast small taxonomy vs larger taxonomy. Do not make §2.4 a chronology of every YOLO version.

---

## 2.5 YOLO26

Target: 2–4 sources. This section is intentionally narrow.

| Key | Source | Status | Intended sentence role |
|---|---|---|---|
| DET-01 | Jocher et al. 2026 — *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models* | CORE-VERIFIED | Primary architecture/design source; state preprint status |
| EVAL-01 | Lin et al. 2014 — COCO | FOUNDATIONAL | Benchmark metric context where needed |
| EVAL-02 | COCOeval official implementation | FOUNDATIONAL | Exact AP@[.50:.95] settings when discussed |
| REPO-PROTOCOL | Repository configuration/protocol | IMPLEMENTATION | Exact model checkpoint, Ultralytics version and experiment settings; not a literature citation |

Rule: repository evidence describes **our implementation**, while `DET-01` describes the published/preprint model.

---

## 2.6 Fine-Grained Object Detection

Target: 7–10 distinct sources. Deliberately mix general FG theory and multiple coffee studies.

| Key | Source | Status | Intended sentence role |
|---|---|---|---|
| FG-03 | Wang et al. 2020 | CORE-VERIFIED | General fine-grained recognition definition / subtle inter-class differences |
| FG-02 | Xie et al. 2025 | CORE-VERIFIED | Fine-grained object detection and discriminative representation |
| FG-01 | Xu et al. 2025 | CORE-VERIFIED | Frequency-aware FGOD bridge, not coffee proof |
| COF-07 | Kesiman et al. 2023 | CORE-VERIFIED | Direct coffee evidence: coarse-to-17-class difficulty jump |
| COF-08 | Arwatchananukul et al. 2024 | CORE-VERIFIED | 17-class coffee and controlled-to-unseen behavior |
| COF-03 | Samudra & Rachmawati 2025 | CORE-VERIFIED | Black vs partially-black visual similarity |
| COF-04 | Hebert & Alamsyah 2026 | CORE-VERIFIED | Difficult subtle tail classes in detection |
| COF-05 | Jundullah et al. 2026 | CORE-VERIFIED | 20-class detection and visually similar class disparity |
| COF-12 | Jiao et al. 2025 | CORE-VERIFIED | Internal multistage/attention solution for discriminative coffee representation |
| COF-13 | Hu et al. 2025 | CORE-VERIFIED | Explicit subtle visual-difference / metric-learning coffee evidence |
| COF-09 | Lei et al. 2025 | CORE-VERIFIED | Optional classification/localization separation in coffee |

Rule: Hong is not required here. If used, one sentence maximum as a secondary corroborating source because its primary home is §2.4.

---

## 2.7 Preprocessing Citra untuk Object Detection

Target: 7–10 distinct sources. Compare fixed, task-driven learned, transform-domain and agricultural analogues.

| Key | Source | Status | Intended sentence role |
|---|---|---|---|
| PRE-04 | Syauqi et al. 2025 | PROJECT-FULLTEXT | Fixed CLAHE-based composite preprocessing before YOLO; seed/spice analogue |
| PRE-05 | Chen et al. 2024 | PROJECT-FULLTEXT | Fixed preprocessing before YOLO in maize-seed crack inspection |
| PRE-01 | Liu et al. 2022 — IA-YOLO | PROJECT-FULLTEXT | Detection-driven adaptive image processing |
| PRE-02 | Qin et al. 2022 — DENet | PROJECT-FULLTEXT | Detection-driven LF/HF enhancement |
| PRE-06 | Tu et al. 2026 — WCTE | PROJECT-FULLTEXT | Transform-domain fixed preprocessing precedent |
| PRE-07 | Cai et al. 2023 — Retinexformer | PROJECT-FULLTEXT | Enhancement judged by downstream utility |
| PRE-08 | Yang & Soatto 2020 — FDA | PROJECT-FULLTEXT | Input-space Fourier manipulation precedent |
| PRE-03 | Li et al. 2025 — FE-YOLO | PROJECT-FULLTEXT | Use as transition into §2.8 rather than allowing it to dominate §2.7 |

Writing pattern: spatial/intensity preprocessing → task-driven preprocessing → transform/Fourier preprocessing → position of thesis.

---

## 2.8 Representasi Citra pada Domain Frekuensi

Target: 8–12 distinct references across the whole §2.8. Do not force all references into every sub-subsection.

### 2.8.1 DFT dan FFT

- `THEORY-01` Gonzalez & Woods — primary image-processing textbook.
- `THEORY-02` Bracewell — secondary Fourier foundation.
- `PRE-08` FDA — optional modern input-space example after the definition.

### 2.8.2 Magnitudo/Amplitudo dan Fase

- `THEORY-01` Gonzalez & Woods.
- `THEORY-02` Bracewell.
- `PRE-03` FE-YOLO — detector-oriented amplitude/phase processing example.
- `PRE-08` FDA — amplitude-manipulation example.

### 2.8.3 Representasi Radial dan Angular

- `SPEC-01` Cao et al. 2019 — primary radial/angular Fourier-energy research source.
- `SPEC-02` Zhang & Tan 2003 — orientation-spectrum texture signatures.
- `THEORY-01` for foundational spectral terminology where needed.

### 2.8.4 Pemrosesan Frekuensi untuk Object Detection

- `FREQ-01` Chi et al. 2020 — Fast Fourier Convolution.
- `FREQ-02` Li et al. 2024 — FDADNet surface-defect frequency processing.
- `FREQ-03` Chen et al. 2025 — FDConv.
- `WAVE-01` Finder et al. 2024 — WTConv.
- `FG-01` Xu et al. 2025 — fine-grained frequency parent bridge.
- `PRE-03` FE-YOLO — input Fourier enhancement comparator.
- `AGR-01` Zhao et al. 2026 — agricultural wavelet/frequency-aware detection.
- `AGR-02` PFENet 2026 — optional feature-space Fourier detector example.

Section rule: distinguish **input preprocessing** from **internal feature processing** explicitly.

---

## 2.9 Penelitian Terkait — main table shortlist

Target: 12–18 rows + final `Penelitian yang Diusulkan`. The latest master workbook has an 18-paper full-text-verified shortlist and marks 14 as `Main table = YES`. The proposal table should use the strongest 14 first and keep 4 optional for prose or replacement.

### Proposed 14-row balanced main table

#### A. Direct coffee / fine-grained coffee — 7 rows

1. `COF-01` Hong et al. 2026 — improved YOLOv10 pivot.
2. `COF-06` Gope et al. 2024 — YOLO-family green-coffee benchmark.
3. `COF-02` Bahy & Rifai 2026 — 20-class SNI-oriented lightweight YOLO.
4. `COF-05` Jundullah et al. 2026 — 20-class defect/contaminant detection.
5. `COF-04` Hebert & Alamsyah 2026 — 15-class subtle-defect detection.
6. `COF-07` Kesiman et al. 2023 — coarse-vs-fine SNI benchmark.
7. `COF-12` Jiao et al. 2025 **or** `COF-13` Hu et al. 2025 — recent discriminative coffee representation.

#### B. Fixed / task-driven preprocessing — 4 rows

8. `PRE-04` Syauqi et al. 2025 — white-pepper CLAHE-based preprocessing + YOLO.
9. `PRE-05` Chen et al. 2024 — maize-seed image enhancement + YOLOv8.
10. `PRE-01` IA-YOLO — adaptive task-driven preprocessing.
11. `PRE-03` FE-YOLO — learned Fourier enhancement before YOLO.

#### C. Frequency / fine-grained mechanism — 3 rows

12. `FG-01` Xu et al. 2025 — LFDet / AFAB frequency-aware fine-grained detection.
13. `FG-02` Xie et al. 2025 — discriminative FGOD representation.
14. `FREQ-02` FDADNet **or** `FREQ-03` FDConv — frequency-aware defect/dense-prediction contrast.

Then:

15. **Penelitian yang Diusulkan** — parameter-free frequency-angular input preprocessing + YOLO26.

### Optional replacement/support rows

- `COF-03` Samudra & Rachmawati 2025.
- `COF-08` Arwatchananukul et al. 2024.
- `COF-13` Hu et al. 2025 if Jiao occupies main row 7.
- `PRE-02` DENet or `PRE-06` WCTE.

Do not make table length a goal. Every row must reveal a distinct comparison dimension: domain, class granularity, method location, preprocessing type, frequency mechanism, or limitation.

---

## Unique-pool count

The routed pool above contains more than 40 distinct sources/authority entries before optional Coffee Atlas expansion. That is sufficient breadth for the campus-style Bab II without citation padding.

The drafting objective is therefore **not to search for arbitrary extra references**. The objective is to use this pool correctly, re-open primary PDFs for exact claims, and only expand when a subsection still has a genuine evidence gap.
