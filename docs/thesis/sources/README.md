# Thesis Source Registry

This directory is for **source metadata and evidence mapping**, not for committing publisher PDFs unless licensing explicitly permits it.

## Source groups

Maintain references under four conceptual groups:

### 1. DIRECT_COFFEE

Primary coffee-bean defect/grading/detection/classification papers.

Priority anchors currently include:

- Hong et al. (2026), improved YOLOv10 coffee defect detection;
- Bahy & Rifai (2026), lightweight YOLOv5s / SNI coffee-defect detection;
- Jundullah et al. (2026), multi-class YOLOv8 coffee defects and contaminants;
- Samudra & Rachmawati (2025), LSKNet-oriented coffee defect detection;
- Hebert & Alamsyah (2026), SCA coffee defect detection with YOLO;
- Kesiman et al. (2023), SNI fine-grained coffee benchmark;
- Arwatchananukul et al. (2024), 17-class coffee defects;
- Jiao et al., multistage/attention coffee grading;
- Hu et al., Siamese/few-shot coffee defect recognition;
- Gope et al., YOLO-family green-coffee comparison.

### 2. ADJACENT_SEED_GRAIN

Agricultural seed/grain/commodity papers used as nearby-domain precedent, e.g.:

- white pepper defect detection with CLAHE-based preprocessing + YOLOv8m;
- maize seed crack detection with image enhancement + YOLOv8;
- cocoa/wheat/rice papers only where their visual/task structure is genuinely relevant.

These sources support analogy, not direct coffee claims.

### 3. PREPROCESSING_FREQUENCY

Papers that establish the image-processing solution space, including:

- IA-YOLO — adaptive image processing before YOLO;
- DENet / DE-YOLO — low/high-frequency enhancement before YOLO;
- FE-YOLO — Fourier amplitude/phase enhancement before YOLO;
- WCTE and related transform-domain preprocessing work where relevant;
- FDA for fundamental input-space Fourier amplitude manipulation;
- Cao et al. for radial/angular Fourier-spectrum interpretation;
- Zhang & Tan for orientation-spectrum discrimination.

### 4. CLASSIFICATION_LOCALIZATION

Papers used to justify separate analysis of detector subtasks, including:

- TOOD;
- Rethinking Classification and Localization for Object Detection;
- IoU-Net.

These are conceptual/diagnostic references and should not be cited as coffee-specific evidence.

## Suggested source-registry schema

When the bibliography is formalized, create a machine-readable registry (CSV/YAML/JSON) with at least:

```text
key
authors
year
title
venue
doi_or_url
source_group
task
dataset
class_count
full_text_verified
pages_or_sections_used
claim_role
notes
```

## Evidence notes

For each core paper, record what it is allowed to support.

Example:

```text
HONG2026
  role:
    - direct coffee-YOLO precedent
    - subtle visually similar defect problem
    - baseline -> modification -> ablation research pattern
  must_not_be_used_for:
    - claiming all coffee literature is covered
    - proving frequency-domain processing solves coffee defects
```

This explicit positive/negative mapping should be maintained for the 15–20 papers that form the proposal's main argumentative chain.

## Full-text rule

Before a paper supplies a numerical or methodological claim in the final proposal, verify the original PDF/table/section. Metadata-only discoveries can remain in the broader Coffee Atlas but should not silently become core evidence.
