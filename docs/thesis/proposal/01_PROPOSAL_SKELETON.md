# Proposal Skeleton — Synchronized

Working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

Status: working title; current methodology operationalizes both `Analisis` and `Optimasi`.

This file is an index/consistency document. Detailed prose is maintained in the chapter files.

---

## Bab I — Pendahuluan

Struktur Bab I mengikuti **urutan dan gaya proposal kampus yang dijadikan acuan**, bukan urutan generik lain:

```text
1.1 Latar Belakang
1.2 Rumusan Masalah
1.3 Tujuan Penelitian
1.4 Batasan Penelitian
1.5 Manfaat Penelitian
1.6 Sistematika Penulisan
```

Bab I **tidak** menggunakan subbab terpisah `Identifikasi Masalah`, `Research Question`, maupun `Kontribusi yang Diharapkan` pada naskah proposal formal.

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

Campus-style rule:

- ditulis sebagai **narasi satu bagian**, bukan daftar RQ1–RQ4;
- paragraf merangkum masalah domain, gap metode, kebutuhan optimasi AF2, dan kebutuhan evaluasi konfirmatori;
- detail operasional eksperimen dipindahkan ke Bab III.

### 1.3 Tujuan Penelitian

Mengikuti gaya proposal acuan dengan pembuka:

> Tujuan dari penelitian ini adalah:

Tujuan formal disusun sebagai butir yang menjawab rumusan masalah:

1. merancang dan mengimplementasikan AF2 sebelum YOLO26;
2. menganalisis dan mengoptimasi keputusan desain AF2;
3. mengevaluasi dan membandingkan native YOLO26 vs AF2-YOLO26;
4. menganalisis difficult classes melalui diagnostic, visualisasi, dan error analysis;
5. mengevaluasi trade-off performa dan efisiensi.

### 1.4 Batasan Penelitian

Mengikuti gaya proposal acuan dengan pembuka:

> Batasan dari penelitian ini adalah:

Batasan utama:

- 21 kelas pada dataset green-coffee object detection yang digunakan;
- YOLO26n sebagai detector utama;
- kontribusi berada pada input-space AF2, bukan modifikasi backbone/neck/head;
- structural AF2 factorization + limited sensitivity only;
- locked test tidak digunakan untuk selection/tuning;
- final confirmation menggunakan matched direct-from-pretrained arms dan paired seeds;
- mechanism/visual diagnostics tidak diperlakukan sebagai bukti kausal tunggal.

### 1.5 Manfaat Penelitian

Mengikuti gaya proposal acuan dengan pembuka:

> Penelitian ini diharapkan dapat memberikan beberapa manfaat sebagai berikut:

Manfaat diarahkan pada bukti empiris AF2, dasar pemilihan konfigurasi, kontribusi literatur preprocessing input-space, analisis kelas yang terbantu/regresi, dan accuracy-efficiency trade-off.

### 1.6 Sistematika Penulisan

Mengikuti pola proposal acuan: Bab 1 sampai Bab 5 dijelaskan secara singkat dalam daftar bernomor.

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
COF-14 Muchtar -> modern deep-learning/edge transition
```

### 2.3 Object Detection

Detector foundation + classification/localization distinction.

### 2.4 YOLO

Original YOLO as conceptual anchor; coffee studies only as domain evidence.

### 2.5 YOLO26

Primary YOLO26 preprint + repository protocol; no unsupported architecture invention.

### 2.6 Fine-Grained Object Detection

General FGOD theory -> coffee-domain class difficulty -> representation problem.

### 2.7 Preprocessing Citra untuk Object Detection

Fixed, transform-domain, learned/task-driven, and Fourier input-space precedents.

### 2.8 Representasi Citra pada Domain Frekuensi

DFT/FFT -> amplitude/phase -> radial/angular spectrum -> input-space vs feature-space processing.

### 2.9 Penelitian Terkait

Authoritative table: `04_09_RELATED_WORK_TABLE.md`.

---

## Bab III — Metode Penelitian

Authoritative draft: `05_METHODOLOGY.md`.
Authoritative AF2 replacement module: `05_05_AF2_PRIMARY_SOURCE_HARDENED.md`.

Bab III follows the Hong-adapted but thesis-specific logic:

```text
overall method
-> dataset/split
-> baseline YOLO26
-> proposed AF2-YOLO26
-> AF2 operator
-> structural optimization
-> method freeze
-> matched confirmation
-> metrics
-> diagnostics
-> visualization/error analysis
-> efficiency
```

The Hong paper is a methodological template for systematic ablation/sensitivity and visualization/error analysis; its DSConv/SPPF-Attention/PConv modules and YOLOv10 hyperparameters are not copied.
