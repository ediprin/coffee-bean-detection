# Thesis Source Registry

This directory is for **source metadata, evidence mapping, citation routing and audit**, not for committing publisher PDFs unless licensing explicitly permits it.

## Source workflow

Use this chain when drafting proposal prose:

```text
master reference workbook / primary PDF
        ↓
CANONICAL_SOURCE_KEYS.md
        ↓
COFFEE_EVIDENCE_MATRIX.md or METHOD_BRIDGE_MATRIX.md
        ↓
BAB2_REFERENCE_POOL.md / REFERENCE_ALLOCATION_MATRIX.md
        ↓
proposal draft
        ↓
BAB2_CITATION_AUDIT.md
```

The purpose is to prevent three failure modes:

1. unsupported claims;
2. citation recycling (the same familiar paper used everywhere);
3. ambiguous source keys that silently point to different papers.

## Key files

- `CANONICAL_SOURCE_KEYS.md` — single stable key namespace. Latest master-reference IDs take precedence over older ad-hoc aliases.
- `COFFEE_EVIDENCE_MATRIX.md` — coffee-domain facts, allowed claims and overclaim boundaries.
- `METHOD_BRIDGE_MATRIX.md` — preprocessing, frequency, fine-grained and classification/localization mechanism literature.
- `REFERENCE_ALLOCATION_MATRIX.md` — policy for source diversity and subsection routing.
- `BAB2_REFERENCE_POOL.md` — concrete source pool for each Bab II subsection and the related-work shortlist.
- `BAB2_CITATION_AUDIT.md` — live audit of under-cited sections, overused empirical papers, unresolved keys and verification gates.
- `BACKGROUND_CLAIM_LEDGER.md` — claim-level traceability for Bab I.

## Source groups

### 1. DIRECT_COFFEE

Primary coffee-bean defect/grading/detection/classification papers. Core anchors include Hong, Bahy & Rifai, Jundullah, Samudra & Rachmawati, Hebert & Alamsyah, Kesiman, Arwatchananukul, Jiao, Hu, Lei and Gope.

These establish the **coffee-domain problem, taxonomy, classwise difficulty, current solution patterns and practical context**. They do not by themselves prove AF2 suitability.

### 2. ADJACENT_SEED_GRAIN

Agricultural seed/grain/commodity papers used as nearby-domain precedent, for example:

- white pepper defect detection with CLAHE-based preprocessing + YOLOv8m;
- maize seed crack detection with image enhancement + YOLOv8;
- cocoa/wheat/rice papers only where their visual/task structure is genuinely relevant.

These sources support analogy, not direct coffee claims.

### 3. PREPROCESSING_FREQUENCY

Papers that establish the image-processing solution space, including:

- IA-YOLO;
- DENet / DE-YOLO;
- FE-YOLO;
- WCTE;
- FDA;
- Retinexformer;
- Fast Fourier Convolution;
- FDADNet;
- FDConv;
- WTConv;
- Cao et al. for radial/angular Fourier energy;
- Zhang & Tan for orientation-spectrum discrimination.

### 4. CLASSIFICATION_LOCALIZATION

Papers used to justify separate analysis of detector subtasks, including:

- TOOD;
- Rethinking Classification and Localization for Object Detection;
- IoU-Net.

These are conceptual/diagnostic references and should not be cited as coffee-specific evidence.

### 5. FOUNDATIONAL / EVALUATION

Canonical sources for theory and benchmark definitions, including:

- Gonzalez & Woods for digital image processing / Fourier-domain image theory;
- Bracewell for Fourier foundations;
- Faster R-CNN and original YOLO for detector taxonomy;
- Microsoft COCO and official COCOeval for evaluation context/specification;
- YOLO26 primary preprint for the detector actually used in this thesis.

## Machine-readable registry schema

When the bibliography is formalized, maintain a CSV/YAML/JSON registry with at least:

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

For each core paper, record both positive and negative scope.

Example:

```text
COF-01 / HONG2026
  role:
    - direct coffee-YOLO precedent
    - subtle visually similar defect problem
    - baseline -> modification -> ablation research pattern
  must_not_be_used_for:
    - claiming all coffee literature is covered
    - proving frequency-domain processing solves coffee defects
```

This explicit mapping should be maintained for all papers that form the proposal's main argumentative chain.

## Full-text rule

Before a paper supplies a numerical or methodological claim in final proposal prose, verify the original PDF/table/section. Metadata-only discoveries can remain in the broader Coffee Atlas but must not silently become core evidence.

## Reference-diversity rule

A non-foundational empirical paper should normally have one primary argumentative home and at most one secondary theory use in Bab II. Reappearance in the `Penelitian Terkait` synthesis table is allowed and does not count as a theory-section reuse.

The objective is not maximum citation count. It is **coverage, source diversity, traceability, and correct assignment of each source to the claim it actually supports**.
