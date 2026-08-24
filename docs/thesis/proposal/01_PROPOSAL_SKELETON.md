# Proposal Skeleton — Synchronized

Working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

Status: working title; terminology may be refined after advisor feedback.

This skeleton is an index/consistency document. Detailed prose is maintained in the chapter files.

---

## Bab I — Pendahuluan

### 1.1 Latar Belakang

Authoritative draft: `02_BACKGROUND.md`.

Required argument chain:

```text
coffee quality / physical inspection
-> limitations of manual selection
-> computer vision and deep learning
-> YOLO viability in coffee
-> finer taxonomy exposes class-wise disparity / visual similarity
-> discriminative representation problem
-> coffee literature mostly modifies model internally
-> preprocessing is an alternative solution space
-> frequency/angular evidence makes AF2 technically testable
-> matched AF2 + YOLO26 experiment
-> pilot evidence = feasibility only
```

Causal guardrail:

```text
coffee fine-grained difficulty != proven frequency bottleneck
```

### 1.2–1.6 Identifikasi, Rumusan, Tujuan, Batasan, Kontribusi

Authoritative draft: `03_PROBLEM_FORMULATION.md`.

Current research questions:

**RQ1.** Apakah preprocessing citra berbasis frekuensi-angular dapat meningkatkan kinerja YOLO26 dalam mendeteksi cacat biji kopi secara fine-grained dibandingkan YOLO26 tanpa preprocessing tersebut?

**RQ2.** Bagaimana pengaruh preprocessing frekuensi-angular terhadap kelas-kelas cacat yang memiliki kinerja rendah atau sulit dibedakan?

**RQ3.** Apakah pola perubahan kinerja yang dihasilkan lebih konsisten dengan peningkatan diskriminasi kelas daripada peningkatan aksesibilitas proposal/lokalisasi mentah?

RQ3 guardrail: raw proposal accessibility adalah diagnostic/proxy dan tidak identik dengan full box-regression quality.

Main objectives:

1. evaluate native YOLO26 vs AF2-YOLO26 under matched conditions;
2. analyze aggregate and lower-tail class behavior;
3. compare discrimination-oriented and proposal-accessibility diagnostics without causal overclaim;
4. measure accuracy–efficiency trade-off.

---

## Bab II — Tinjauan Pustaka

Campus convention source: `../foundation/06_USU_BAB2_PATTERN.md`.

Main draft: `04_LITERATURE_REVIEW.md`.

Normalized assembly modules:

- `04_02_INSPECTION_QUALITY_NORMALIZED.md` — authoritative replacement for §2.2;
- `04_09_RELATED_WORK_TABLE.md` — authoritative §2.9 table.

### 2.1 Biji Kopi Hijau dan Cacat Fisik Biji Kopi

Standard + taxonomy context. Do not equate dataset labels with the complete SNI grading procedure.

### 2.2 Inspeksi Mutu Biji Kopi: Metode Konvensional dan Tantangannya

Use normalized module. Primary routing:

```text
COF-17 Garcia 2019 -> manual/mechanical inspection + classical machine vision
COF-10 de Oliveira -> traditional controlled feature engineering
REV-01 -> landscape only
COF-14 -> modern DL/edge transition
```

Do not recycle `COF-07/08` here; reserve them for taxonomy/fine-grained roles.

### 2.3 Object Detection

Use canonical detector sources plus classification/localization diagnostics.

### 2.4 YOLO

Original YOLO is the theory source; coffee studies only show domain adoption.

### 2.5 YOLO26

YOLO26 primary source is a 2026 preprint. AF2 is not part of YOLO26 backbone/neck/head.

### 2.6 Fine-Grained Object Detection

General FG/FGOD theory + independent coffee classification/detection evidence. Hong is not required as the dominant source here.

### 2.7 Preprocessing Citra untuk Object Detection

Separate fixed/composite, learned task-driven, transform-domain, and Fourier preprocessing. Downstream detector utility is the evaluation criterion.

### 2.8 Representasi Citra pada Domain Frekuensi

Subsections:

1. DFT/FFT;
2. amplitude/phase;
3. radial/angular representation;
4. frequency-aware processing in input space vs feature space.

Canonical angular theory keys are `SPEC-01/SPEC-02`; do not reuse deprecated `FREQ-*` meanings.

### 2.9 Penelitian Terkait

Use `04_09_RELATED_WORK_TABLE.md` with 18 prior studies + proposed research.

The table is a synthesis, not a ranking across incomparable datasets.

---

## Bab III — Metodologi Penelitian

Campus convention source: `../foundation/07_USU_BAB3_PATTERN.md`.

Authoritative proposal draft: `05_METHODOLOGY.md`.

Protocol audit: `../sources/BAB3_PROTOCOL_AUDIT.md`.

### 3.1 Arsitektur Umum Penelitian

```text
D0DIRECT:
RGB -> YOLO26n

AF2DIRECT:
RGB -> AF2 -> same YOLO26n
```

### 3.2 Dataset Penelitian

Frozen development contract:

```text
train      1,665 images / 2,986 annotations
validation   294 images /   526 annotations
classes       21
```

### 3.3 Persiapan dan Audit Dataset

Grouped split, parent/hash leakage gates, locked test, no test access during screening.

### 3.4 Preprocessing Frekuensi-Angular AF2

Active `mode=af2`:

```text
patch_size   = 32
overlap      = 0.50 -> stride 16
gamma        = 0.10
angular_bins = 360
chunk_size   = 128
eps          = 1e-8
```

Implementation choices:

- RGB channels independent;
- floor-to-angle-bin discretization;
- fold/overlap averaging;
- residual output `I' = I + I * minmax(recovered)`.

Important: shared config `radius_ratio=0.05` is inactive in pure `mode=af2`; it belongs to AF1/AF12 radial masking.

### 3.5 YOLO26 Detector

Same exact official pretrained source and matched 21-class target-head initialization for both arms.

### 3.6 Skenario Eksperimen

Primary paired delta:

\[
\Delta M=M_{AF2DIRECT}-M_{D0DIRECT}.
\]

Seed 42 = completed pilot screen.

Seeds 123 and 2026 = planned confirmation, not completed proposal results.

### 3.7 Konfigurasi Pelatihan

Direct protocol authority:

```text
max epochs   = 50
imgsz        = 640
batch        = 16
workers      = 2
patience     = 15
optimizer    = auto
pretrained   = true
cache        = false
close_mosaic = 10
max_det      = 500
deterministic= true
```

Do not import the conflicting old 100-epoch D0 schedule into this thesis design.

### 3.8 Evaluasi Performa

Primary / tail:

- mAP50 and mAP50–95;
- Macro mAP50–95;
- Bottom-3 class mAP50–95;
- Worst-class mAP50–95;
- per-class AP.

Diagnostic:

- raw top-500 proposal accessibility;
- localization-conditioned Top-1;
- correct-decision recall.

Efficiency:

- parameter count;
- latency;
- throughput;
- peak memory/VRAM.

Bottom-3 and Worst-class are study-defined summary metrics, not official COCO metrics.

### 3.9 Analisis Kesalahan dan Per-Class Behavior

Report per-seed and per-class behavior; use `consistent with`, not causal mechanism claims.

### 3.10 Perangkat dan Lingkungan Eksperimen

Record hardware, runtime, library versions, pretrained hash, repo commit, and seed.

### 3.11 Batas Pilot vs Tesis

Pilot seed 42 demonstrates feasibility only. Final direct-AF2 superiority, locked-test generalization, and final efficiency claims remain unestablished until confirmatory work is completed.

---

## Proposal-safe result wording

> Studi pendahuluan pada satu seed digunakan untuk menilai kelayakan awal rancangan. Hasil tersebut tidak diperlakukan sebagai kesimpulan final dan akan divalidasi menggunakan protokol eksperimen berulang pada penelitian tesis.

Never change this into a superiority claim before the repeated-seed confirmation is complete.
