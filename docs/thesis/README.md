# Thesis Proposal Workspace

Branch: `proposal/thesis-foundation`

Base branch: `codex/af2-direct-from-pretrained`

## Purpose

This directory is the versioned reasoning and evidence base for the thesis proposal on frequency-angular image preprocessing for fine-grained coffee-bean defect detection with YOLO26.

The purpose is **not** to store a polished proposal immediately. It stores the conceptual chain, evidence rules, research-gap logic, pilot evidence, and document-generation constraints that future proposal chapters must follow.

## Frozen thesis direction

Working title:

> **Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

Core method:

```text
Input image
   -> parameter-free frequency-angular preprocessing (AF2)
   -> processed image
   -> YOLO26
   -> fine-grained coffee-defect detections
```

The thesis narrative must **not** be organized as `D0 -> D0FT -> AF2`. Those are experiment genealogy/control arms. The thesis narrative begins from the coffee-defect problem, then the fine-grained discrimination gap, then preprocessing/frequency-domain literature, then AF2 as the hypothesis to test.

## Directory structure

```text
docs/thesis/
├── README.md
├── foundation/
│   ├── 00_THESIS_CONCEPT.md
│   ├── 01_HONG_PIVOT_LITERATURE_METHOD.md
│   ├── 02_EVIDENCE_AND_CLAIM_RULES.md
│   ├── 03_RESEARCH_GAP_RQ_SCOPE.md
│   └── 04_PILOT_EVIDENCE.md
├── proposal/
│   └── README.md
└── sources/
    └── README.md
```

## Source hierarchy

Future writing must prioritize evidence in this order:

1. **Primary full-text coffee papers** for the coffee-domain problem, taxonomy, difficult classes, dataset protocol, results, limitations, and authors' own explanations.
2. **Hong et al. (2026)** as a pivot for understanding how a coffee-YOLO paper builds its problem statement and imports mechanisms from adjacent computer-vision literature.
3. **Primary preprocessing/frequency papers** for candidate mechanisms and methodological precedent.
4. **Repository code/config/results** for claims about AF2 implementation, training protocol, and pilot evidence.
5. Web metadata/snippets only for discovery or bibliographic confirmation, never as a substitute for full-text methodological evidence when the PDF is available.

## Evidence separation

Every future document must distinguish:

- **Paper fact**: explicitly supported by the cited source.
- **Cross-paper synthesis**: inference obtained by comparing multiple sources.
- **Research hypothesis**: proposition to be tested in this thesis.
- **Repository evidence**: result produced by the current code/experiment protocol.

Do not silently convert a hypothesis into a literature fact.

## Current project phase

Method exploration is temporarily frozen for proposal preparation. New variants are not required before the proposal is written. AF2-direct seed 42 is retained only as **pilot feasibility evidence**; it is not a final superiority claim.

Read the files in `foundation/` before generating or revising any proposal section.
