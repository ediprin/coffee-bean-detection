# Bab II Normalization Audit — 2026-08-25

Status: **citation-reuse hotspot closed in modular source-normalized assembly**.

## Problem found

The first source-grounded §2.2 reused:

- `COF-07` Kesiman et al. in §2.1, §2.2, and §2.6;
- `COF-08` Arwatchananukul et al. in §2.1, §2.2, and §2.6.

The citations were not wrong, but the routing was unnecessarily repetitive.

## New source introduced

Canonical key:

```text
COF-17 = García, Candelo-Becerra & Hoyos (2019)
         Quality and Defect Inspection of Green Coffee Beans Using a Computer Vision System
         Applied Sciences 9(19), 4195
         DOI 10.3390/app9194195
```

Primary full text has been opened in the project File Library.

Paper-scoped evidence used for §2.2:

- manual inspection is performed by personnel visually selecting beans;
- non-uniform selection can result from long work hours, lack of training, and operator factors;
- manual inspection is time-consuming/inefficient for large quantities;
- mechanical sorting based on size cannot evaluate all physical-appearance characteristics;
- the proposed classical machine-vision pipeline uses image acquisition, image processing, handcrafted physical features, KNN, and controlled illumination;
- black defect was easier than sour defect in their experiment, and the authors explicitly discuss feature similarity as a cause of prediction difficulty in their KNN feature space.

Boundary: this 2019 KNN system is historical/classical evidence and is not treated as an object-detection benchmark comparable to modern YOLO.

## Replacement module

`proposal/04_02_INSPECTION_QUALITY_NORMALIZED.md` is now the authoritative §2.2 module for final assembly.

Routing after normalization:

```text
§2.1 taxonomy
    STD-01 + COF-07 + COF-08 + COF-02

§2.2 manual -> classical CV -> modern automation
    COF-17 + COF-10 + REV-01 + COF-14

§2.6 fine-grained discrimination
    FG-03 + FG-02 + COF-07 + COF-08 + COF-03/04/05 + COF-12/13
```

This reduces substantive `COF-07/08` theory-section reuse from three sections to two.

## Index audit update

For Table 2.1, `Computers and Electronics in Agriculture` has been independently rechecked against 2024 Scopus/SJR-style journal metrics and is Q1. The modular related-work table may therefore replace the previous `quartile audit open` label for the Chen et al. maize-seed row with `Q1 — Computers and Electronics in Agriculture (2024 metrics)`.

`Digital Signal Processing` remains Q2 in 2024 JCR/SJR checks, consistent with the current FE-YOLO row.

## Remaining Bab II finalization tasks

- merge normalized §2.2 module into the generated final chapter;
- merge modular 18-study §2.9 table into the generated final chapter;
- pair DFT/FFT fundamentals with final textbook edition/page references;
- re-open every numerical table claim before final DOCX/PDF generation;
- keep cross-paper synthesis separate from paper facts.

No additional literature searching is required by default unless a concrete unsupported sentence appears during final drafting.