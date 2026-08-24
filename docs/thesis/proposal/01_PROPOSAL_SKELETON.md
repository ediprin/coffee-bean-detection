# Proposal Skeleton — Synchronized

Working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

Status: working title; current methodology operationalizes both `Analisis` and `Optimasi`.

This file is an index/consistency document. Detailed prose is maintained in the chapter files.

---

## Bab I — Pendahuluan

Struktur Bab I mengikuti pola proposal kampus yang telah dijadikan acuan:

```text
1.1 Latar Belakang
1.2 Rumusan Masalah
1.3 Batasan Masalah
1.4 Tujuan Penelitian
1.5 Manfaat Penelitian
```

Bab I **tidak** menggunakan subbab terpisah `Identifikasi Masalah` maupun `Kontribusi yang Diharapkan`.

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
-> AF2 contains design choices that require controlled optimization
-> selected AF2 must be confirmed against matched native YOLO26
-> pilot evidence = feasibility only
```

Causal guardrail:

```text
coffee fine-grained difficulty != proven frequency bottleneck
```

### 1.2 Rumusan Masalah

Authoritative draft: `03_PROBLEM_FORMULATION.md`.

Current research questions:

1. Bagaimana pengaruh keputusan desain utama AF2 terhadap kinerja deteksi fine-grained cacat biji kopi, dan konfigurasi preprocessing frekuensi-angular seperti apa yang paling layak dipilih berdasarkan analisis terfaktor dan sensitivity analysis pada data pengembangan?
2. Apakah konfigurasi AF2 yang telah dipilih dapat meningkatkan kinerja YOLO26 dibandingkan native YOLO26 pada eksperimen konfirmatori yang dipasangkan?
3. Bagaimana pengaruh preprocessing frekuensi-angular terhadap kelas-kelas cacat yang memiliki kinerja rendah atau sulit dibedakan?
4. Apakah pola perubahan kinerja lebih konsisten dengan peningkatan diskriminasi kelas daripada peningkatan aksesibilitas proposal/lokalisasi mentah?

Diagnostic guardrail: raw proposal accessibility adalah diagnostic/proxy dan tidak identik dengan full box-regression quality.

### 1.3 Batasan Masalah

Batasan utama:

- 21 kelas pada dataset green-coffee object detection yang digunakan;
- YOLO26n sebagai detector utama;
- kontribusi berada pada input-space AF2, bukan modifikasi backbone/neck/head;
- structural AF2 factorization + limited sensitivity only;
- locked test tidak digunakan untuk selection/tuning;
- final confirmation menggunakan matched direct-from-pretrained arms dan paired seeds;
- mechanism/visual diagnostics tidak diperlakukan sebagai bukti kausal tunggal.

### 1.4 Tujuan Penelitian

1. menganalisis dan memilih konfigurasi AF2 melalui factorized analysis dan limited sensitivity;
2. mengonfirmasi selected AF2 terhadap matched native YOLO26;
3. menganalisis difficult/lower-tail classes;
4. mendiagnosis discrimination-vs-proposal-accessibility pattern;
5. mengevaluasi accuracy-efficiency trade-off.

### 1.5 Manfaat Penelitian

Manfaat diarahkan pada:

1. evidence penggunaan input-space frequency-angular preprocessing untuk fine-grained coffee-defect detection;
2. kerangka evaluasi aggregate + per-class + lower-tail + paired errors;
3. informasi kelas yang terbantu maupun mengalami regresi;
4. informasi accuracy-efficiency trade-off untuk penelitian lanjutan.

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

Primary routing:

```text
COF-17 Garcia 2019 -> manual/mechanical inspection + classical machine vision
COF-10 de Oliveira -> traditional controlled feature engineering
REV-01 -> landscape only
COF-14 -> modern DL/edge transition
```

### 2.3 Object Detection

Use canonical detector sources plus classification/localization diagnostics.

### 2.4 YOLO

Original YOLO is the theory source; coffee studies show domain adoption.

### 2.5 YOLO26

YOLO26 primary source is a 2026 preprint. AF2 is not part of the YOLO26 backbone/neck/head.

### 2.6 Fine-Grained Object Detection

General FG/FGOD theory + independent coffee classification/detection evidence.

### 2.7 Preprocessing Citra untuk Object Detection

Separate fixed/composite, learned task-driven, transform-domain, and Fourier preprocessing. Downstream detector utility is the evaluation criterion.

### 2.8 Representasi Citra pada Domain Frekuensi

Subsections:

1. DFT/FFT;
2. amplitude/phase;
3. radial/angular representation;
4. frequency-aware processing in input space vs feature space.

Canonical angular theory keys are `SPEC-01/SPEC-02`.

### 2.9 Penelitian Terkait

Use `04_09_RELATED_WORK_TABLE.md` with 18 prior studies + proposed research.

---

## Bab III — Metode Penelitian

Campus convention source: `../foundation/07_USU_BAB3_PATTERN.md`.

Method-design authority: `../foundation/08_BAB3_HONG_ADAPTED_OPTIMIZATION_DESIGN.md`.

Authoritative draft: `05_METHODOLOGY.md`.

Post-rewrite audit: `../sources/BAB3_HONG_REWRITE_AUDIT_2026-08-25.md`.

### 3.1 Kerangka Penelitian

```text
problem diagnosis
-> grouped dataset audit
-> AF2 reference
-> factorized AF2 optimization
-> optional limited parameter sensitivity
-> AF2* method freeze
-> matched native-vs-AF2 confirmatory experiment
-> aggregate + tail + mechanism + visualization/error + efficiency analysis
```

### 3.2 Dataset Penelitian

Frozen development contract:

```text
train      1,665 images / 2,986 annotations
validation   294 images /   526 annotations
classes       21
```

Subsections:

- source/characteristics;
- taxonomy;
- grouped split and leakage control;
- augmentation/input-transform distinction.

### 3.3 Baseline YOLO26

Use YOLO26n P3–P5 as fixed detector family.

Final confirmatory arms use the same exact official `yolo26n.pt` source and matched 21-class target-head initialization.

### 3.4 Arsitektur Metode yang Diusulkan

```text
Native:
RGB -> YOLO26n

Proposed:
RGB -> AF2* -> same YOLO26n
```

AF2 is an input frontend, not backbone/neck/head modification.

### 3.5 Preprocessing Frekuensi-Angular AF2

Reference settings:

```text
patch_size   = 32
overlap      = 0.50 -> stride 16
gamma        = 0.10
angular_bins = 360
chunk_size   = 128
eps          = 1e-8
```

Core chain:

```text
patch
-> FFT
-> angular density
-> entropy-adaptive threshold
-> directional weighting
-> IFFT
-> overlap reconstruction
-> residual enhancement
```

Reference residual:

\[
I'=I+I\odot\operatorname{MinMax}(R_{AF2}(I)).
\]

Important: `radius_ratio=0.05` is inactive in pure `mode=af2`.

### 3.6 Analisis dan Optimasi AF2

Main methodological adaptation from Hong:

```text
systematic factorized ablation
+ sensitivity analysis
+ later visualization/error analysis
```

Primary structural candidates:

```text
AF2C   reference
AF2WIN window/leakage factor
AF2ORI orientation-factor representation
AF2POL radial x angular factorization
AF2SOFT hard -> soft threshold
AF2LUM RGB-independent -> luminance/shared gate
```

These are one-factor alternatives, not modules to stack.

Optional mechanistic comparators `PCG1/WAV1` are not AF2 variants.

Parameter-sensitivity candidate set:

\[
\Theta_{AF2}=\{m,o,\gamma,K\}.
\]

Priority, if retained: `gamma` and `patch_size`. Exact additional candidate values must be frozen before observing their validation outcomes.

### 3.7 Rancangan Eksperimen Konfirmatori

Final paired comparison:

\[
\Delta M=M_{AF2^*}-M_{Native}.
\]

Planned paired seeds:

```text
42, 123, 2026
```

Seed 42 direct result = feasibility pilot only.

Important provenance split:

- historical factorization = development/selection evidence, using seed-matched D0 parent;
- final direct confirmation = official pretrained YOLO26n, matched target head.

Locked test is not used for model selection.

### 3.8 Konfigurasi Pelatihan

Final direct protocol authority:

```text
max epochs    = 50
imgsz         = 640
batch         = 16
workers       = 2
patience      = 15
optimizer     = auto
pretrained    = true
cache         = false
close_mosaic  = 10
max_det       = 500
deterministic = true
```

### 3.9 Metrik Evaluasi

Performance hierarchy:

- Macro mAP50–95 — primary;
- mAP50 / mAP50–95 — context;
- Bottom-3 — study-defined tail metric;
- Worst-class — study-defined safety indicator;
- per-class AP;
- Precision/Recall/F1 when useful for descriptive error analysis.

### 3.10 Analisis Mekanisme

Diagnostic:

- raw top-500 proposal accessibility;
- localization-conditioned Top-1;
- correct-decision recall.

Use `consistent with`, not causal mechanism claims.

### 3.11 Analisis Visualisasi

AF2-specific qualitative chain:

```text
Original RGB
-> patch
-> FFT magnitude
-> angular density D(theta)
-> threshold tau
-> retained response
-> reconstructed cue
-> AF2-enhanced RGB
-> prediction / activation visualization
```

CAM/EigenCAM is not named as final until YOLO26 compatibility is verified.

### 3.12 Analisis Kesalahan

- confusion/per-class errors;
- gain/regression by class;
- paired native-vs-AF2 outcome transitions;
- thesis-defined rescue/regression statistic;
- deterministic/fixed-seed qualitative sample selection to reduce cherry-picking.

### 3.13 Evaluasi Efisiensi

- parameter count;
- latency;
- throughput;
- peak memory/VRAM.

`parameter-free != compute-free`.

### 3.14 Lingkungan Implementasi dan Reproducibility

Record hardware, runtime, library versions, pretrained hash, repo commit, seed, dataset contract, and run contract.

### 3.15 Batas Pendahuluan / Optimasi / Bukti Final

- direct seed-42 pilot = feasibility only;
- old factorization results = development/selection evidence;
- selected AF2* direct paired confirmation = main thesis evidence.

---

## Proposal-safe result wording

> Studi pendahuluan pada satu seed digunakan untuk menilai kelayakan awal rancangan. Hasil tersebut tidak diperlakukan sebagai kesimpulan final dan akan divalidasi menggunakan protokol eksperimen berulang pada penelitian tesis.

Never change this into a superiority claim before repeated-seed confirmation is complete.