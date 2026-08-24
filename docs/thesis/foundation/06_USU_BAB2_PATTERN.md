# USU Bab II Pattern

Status: adopted structural convention for the thesis proposal.

Source basis: proposal example uploaded on 2026-08-24, titled *Analisis Pengaruh Variasi CNN Backbone terhadap Performa dan Interpretabilitas Integrasi CNN–Vision Transformer pada Klasifikasi Fertilitas Telur Burung Puyuh Menggunakan Grad-CAM*.

## Why this file exists

The proposal will follow the local campus convention rather than invent a literature-review structure that is academically valid but unfamiliar to the program.

We copy the **structural pattern**, not the content, claims, references, or wording of the example.

## Pattern observed in the campus proposal

Bab II follows a simple progression:

1. domain object / biological or application background;
2. conventional inspection method and its limitations;
3. primary deep-learning foundation;
4. secondary architecture / technical concept;
5. combined or method-specific architecture concept;
6. variants/components that are directly compared in the research;
7. an additional technique central to evaluation or interpretation;
8. `Penelitian Terkait` as a comparative table;
9. the final row of the table is `Penelitian yang Diusulkan` and explicitly states how the proposed work fills the gap.

The chapter therefore behaves as:

```text
application domain
    ↓
conventional process / problem
    ↓
core technical foundations
    ↓
method-specific concepts
    ↓
related-work comparison table
    ↓
proposed research position
```

## Adopted pattern for this thesis

The coffee thesis will use the same recognizable campus structure:

### 2.1 Biji Kopi Hijau dan Cacat Fisik Biji Kopi

Purpose:
- define green coffee beans as the inspection object;
- explain physical defects relevant to quality assessment;
- introduce defect taxonomy only to the level needed by the thesis;
- distinguish normal beans, defect categories, and contaminants where relevant.

Do not turn this section into a complete agronomy review.

### 2.2 Inspeksi Mutu Biji Kopi: Metode Konvensional dan Tantangannya

Purpose:
- explain manual visual inspection / grading;
- explain subjectivity, throughput, consistency, and fine-grained visual similarity;
- transition naturally to computer vision.

### 2.3 Object Detection

Purpose:
- explain what object detection predicts;
- distinguish classification and localization conceptually;
- introduce bounding box, class score, confidence, IoU, and one-stage detection at a level appropriate for the proposal.

This section is theory, not our method.

### 2.4 YOLO (You Only Look Once)

Purpose:
- explain the YOLO family as a one-stage detector;
- summarize why YOLO is relevant to real-time agricultural inspection;
- keep historical detail concise and only as needed to position YOLO26.

### 2.5 YOLO26

Purpose:
- describe the detector used in the thesis;
- explain only architecture/training properties that matter to the proposed comparison;
- cite the original YOLO26 source and repository/config evidence separately.

Do not describe AF2 here.

### 2.6 Fine-Grained Object Detection

Purpose:
- define the fine-grained setting as classes with small inter-class visual differences;
- connect to difficult coffee-defect taxonomies;
- explain why aggregate mAP may hide difficult classes;
- introduce discriminative representation as a research problem.

Important boundary:
coffee evidence establishes the discrimination problem, not a frequency-domain bottleneck.

### 2.7 Preprocessing Citra untuk Object Detection

Purpose:
- define preprocessing as transformation before the detector;
- distinguish fixed preprocessing from learned/task-driven enhancement;
- discuss contrast enhancement, denoising/sharpening only as relevant examples;
- include CLAHE as a conventional comparator / literature precedent;
- introduce detection-driven preprocessing literature such as IA-YOLO and DENet.

### 2.8 Representasi Citra pada Domain Frekuensi

This section contains the theoretical bridge to the thesis method.

#### 2.8.1 Discrete Fourier Transform dan Fast Fourier Transform

Explain spatial-to-frequency representation and notation.

#### 2.8.2 Magnitudo/Amplitudo dan Fase Spektrum

Explain what magnitude/amplitude and phase encode, without claiming that either is the proven coffee bottleneck.

#### 2.8.3 Representasi Radial dan Angular pada Spektrum Fourier

Explain polar frequency coordinates, radial energy, angular energy, and directional texture evidence.

#### 2.8.4 Pemrosesan Frekuensi untuk Object Detection

Position FE-YOLO, frequency-aware detection, wavelet/Fourier mechanisms, and the fine-grained frequency bridge.

Important boundary:
this subsection establishes a plausible solution space; it does not establish AF2 effectiveness on coffee.

### 2.9 Penelitian Terkait

Use the same table style as the campus proposal:

| No | Penulis & Tahun | Indeks | Fokus Penelitian | Metode / Model | Kontribusi dan Pengisian Gap Penelitian |
|---:|---|---|---|---|---|

The final row must be:

`Penelitian yang Diusulkan`

and should state the proposed contribution in campus-style language, but with cautious evidence boundaries.

## Related-work table composition

The table should not contain only coffee papers or only frequency papers. It should show the full reasoning chain.

Recommended composition:

- direct coffee detection anchors: Hong, Bahy & Rifai, Jundullah, Hebert & Alamsyah, Gope;
- fine-grained coffee diagnostic evidence: Kesiman / Arwatchananukul or another verified core paper;
- agricultural preprocessing analogue: Syauqi white pepper and/or Chen maize;
- task-driven input preprocessing: IA-YOLO / DENet;
- Fourier preprocessing: FE-YOLO;
- fine-grained frequency bridge: Xu et al. AFAB/LFDet;
- final row: proposed AF2 + YOLO26 research.

A table of roughly 9–12 prior studies plus the proposed research is acceptable; quality and function matter more than reaching an arbitrary count.

## Index-column rule

The campus example includes an `Indeks` column. We preserve it.

However:
- journal/conference name and quartile/index must be verified before final submission;
- never infer Q1/Q2 from memory;
- use `TBD - verify` in drafting if the bibliometric status has not been checked against the correct year/source.

## What NOT to copy from the example

The uploaded example contains some draft-like weaknesses (e.g. placeholder citation text, inconsistent bibliographic details, and some statements that require stronger source verification).

Therefore we copy only:
- chapter order;
- heading hierarchy;
- concise theory-first style;
- comparative `Penelitian Terkait` table;
- final proposed-research row.

We do **not** copy unsupported claims or citation quality.

## Frozen decision

For proposal drafting, this file supersedes the older free-form Bab II sequence in `01_PROPOSAL_SKELETON.md`.

Bab II should look familiar to the campus while preserving the stricter evidence rules in `02_EVIDENCE_AND_CLAIM_RULES.md`.