# Proposal Skeleton

Working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

Status: working title; terminology may be refined after advisor feedback.

## Bab I — Pendahuluan

### 1.1 Latar Belakang

Required evidence chain:

1. coffee quality / physical defect inspection matters;
2. manual inspection has consistency and throughput limitations;
3. computer vision and deep learning have been adopted;
4. YOLO-family methods are viable for coffee detection;
5. strong few/coarse-class results do not resolve detailed multi-class detection;
6. fine-grained taxonomies expose visually similar and difficult classes;
7. current coffee solutions mostly alter internal model representation;
8. image preprocessing is an alternative route to improve input utility;
9. task-driven and frequency-aware preprocessing has precedent outside coffee;
10. frequency-angular processing is therefore a testable hypothesis, not an assumed solution;
11. AF2 is positioned as parameter-free input-space preprocessing before YOLO26;
12. one-seed pilot evidence supports feasibility but is not a final conclusion.

### 1.2 Identifikasi Masalah

Draft problems:

- detailed coffee-defect taxonomies show substantial class-wise performance disparity;
- visually similar defect classes are difficult to discriminate;
- aggregate detector metrics can hide weak lower-tail classes;
- most coffee approaches improve discriminative representation internally, while parameter-free frequency-angular input preprocessing remains insufficiently studied in the audited coffee corpus;
- classification/discrimination and localization effects should be distinguished when interpreting gains.

### 1.3 Rumusan Masalah

RQ1. Apakah preprocessing citra berbasis frekuensi-angular dapat meningkatkan kinerja YOLO26 dalam mendeteksi cacat biji kopi secara fine-grained dibandingkan YOLO26 tanpa preprocessing tersebut?

RQ2. Bagaimana pengaruh preprocessing frekuensi-angular terhadap kelas-kelas cacat yang memiliki kinerja rendah atau sulit dibedakan?

RQ3. Apakah perubahan kinerja yang dihasilkan lebih berkaitan dengan kemampuan diskriminasi kelas daripada perubahan kemampuan lokalisasi objek?

### 1.4 Tujuan Penelitian

1. Mengimplementasikan dan mengevaluasi preprocessing frekuensi-angular sebagai front-end input YOLO26 untuk deteksi cacat biji kopi.
2. Menganalisis dampaknya terhadap kinerja agregat dan lower-tail class performance.
3. Menganalisis perubahan pada aspek diskriminasi kelas dan lokalisasi secara terpisah sejauh dapat diukur.
4. Mengevaluasi trade-off akurasi dan efisiensi karena AF2 tidak menambah learned parameters tetapi menambah komputasi input processing.

### 1.5 Batasan Penelitian

- task utama adalah object detection pada green coffee beans;
- detector utama YOLO26, dengan baseline native yang matched;
- AF2 bekerja pada input image, bukan sebagai neck/head/backbone module;
- proposal tidak mengklaim AF2 sebagai penyebab universal peningkatan seluruh kelas;
- novelty global / first-ever claim tidak digunakan tanpa systematic verification;
- pilot seed-42 hanya bukti kelayakan awal;
- final thesis validation should use repeated seeds / paired protocol as defined in the experiment protocol;
- test split remains locked according to repository protocol.

### 1.6 Manfaat / Kontribusi yang Diharapkan

Scientific contribution:

- controlled evaluation of parameter-free frequency-angular input preprocessing for fine-grained coffee-defect detection;
- class-wise and tail-oriented analysis rather than aggregate mAP only;
- diagnostic interpretation separating discrimination from localization where possible.

Engineering contribution:

- preprocessing can be attached to an existing detector without increasing detector learned-parameter count;
- explicit efficiency measurement prevents the misleading claim that parameter-free means compute-free.

## Bab II — Tinjauan Pustaka

**Structural convention:** follow the uploaded USU proposal pattern: application/domain foundation → conventional process/challenge → core technical foundations → method-specific theory → `Penelitian Terkait` comparison table → final `Penelitian yang Diusulkan` row.

Detailed rationale: `docs/thesis/foundation/06_USU_BAB2_PATTERN.md`.

### 2.1 Biji Kopi Hijau dan Cacat Fisik Biji Kopi

- green coffee bean as inspection object;
- physical defect concept and relevant taxonomy;
- normal / defect / contaminant distinction where required;
- concise standards context (SNI/SCA only to support the thesis scope).

### 2.2 Inspeksi Mutu Biji Kopi: Metode Konvensional dan Tantangannya

- manual visual grading / sorting;
- subjectivity, consistency, throughput;
- subtle visual similarity and detailed taxonomy;
- transition to automated computer vision.

### 2.3 Object Detection

- classification + localization;
- bounding box and class prediction;
- confidence / IoU basics;
- conceptual distinction between classification quality and localization quality.

### 2.4 YOLO (You Only Look Once)

- one-stage detection concept;
- brief YOLO-family development only as necessary;
- relevance to real-time agricultural inspection;
- coffee-domain viability evidence.

### 2.5 YOLO26

- detector used in the thesis;
- architecture/training properties relevant to the controlled comparison;
- original YOLO26 source + repository/config evidence;
- AF2 is not discussed as part of YOLO26.

### 2.6 Fine-Grained Object Detection

- small inter-class vs relatively larger intra-class variations;
- visually similar coffee defects;
- discriminative representation requirement;
- aggregate metric vs difficult-class / lower-tail behavior.

### 2.7 Preprocessing Citra untuk Object Detection

- preprocessing before detector;
- fixed vs learned/task-driven preprocessing;
- CLAHE / contrast enhancement as conventional precedent;
- detection-driven enhancement such as IA-YOLO and DENet;
- preprocessing quality must be judged by downstream detection, not visual aesthetics alone.

### 2.8 Representasi Citra pada Domain Frekuensi

#### 2.8.1 Discrete Fourier Transform dan Fast Fourier Transform

- spatial-to-frequency representation;
- core notation and reconstruction.

#### 2.8.2 Magnitudo/Amplitudo dan Fase Spektrum

- magnitude/amplitude and phase definitions;
- role in image representation;
- no claim that either is the proven coffee bottleneck.

#### 2.8.3 Representasi Radial dan Angular pada Spektrum Fourier

- polar frequency coordinates;
- radial energy / periodicity;
- angular energy / directional texture;
- theoretical basis for the term `frequency-angular`.

#### 2.8.4 Pemrosesan Frekuensi untuk Object Detection

- Fourier/wavelet preprocessing and frequency-aware feature methods;
- FE-YOLO as close Fourier-input comparator;
- frequency-aware fine-grained detection bridge such as Xu et al.;
- transfer to coffee remains an empirical research question.

### 2.9 Penelitian Terkait

Use the campus-style table:

| No | Penulis & Tahun | Indeks | Fokus Penelitian | Metode / Model | Kontribusi dan Pengisian Gap Penelitian |
|---:|---|---|---|---|---|

Recommended evidence mix:

1. Hong et al. — coffee YOLO pivot;
2. Bahy & Rifai — 20-class SNI coffee detection;
3. Jundullah et al. — 20-class coffee detection / class disparity;
4. Hebert & Alamsyah — subtle difficult coffee classes;
5. Kesiman et al. or Arwatchananukul et al. — fine-grained coffee taxonomy evidence;
6. Syauqi et al. — fixed CLAHE-based preprocessing + YOLO on white pepper;
7. Chen et al. — image enhancement + YOLO on maize seed cracks;
8. IA-YOLO / DENet — detection-driven input preprocessing;
9. FE-YOLO — learned Fourier enhancement before YOLO;
10. Xu et al. — frequency representation for fine-grained detection;
11. **Penelitian yang Diusulkan** — parameter-free frequency-angular preprocessing + YOLO26 for fine-grained coffee-defect detection.

Index/quartile values must be verified before final submission. `TBD - verify` is preferable to an inferred Q1/Q2 label.

## Bab III — Metodologi

### Core model comparison

Baseline:

```text
RGB image → native YOLO26
```

Proposed:

```text
RGB image → AF2 frequency-angular preprocessing → YOLO26
```

### AF2 conceptual pipeline

```text
image
  ↓
patch extraction
  ↓
FFT
  ↓
angular spectral analysis / weighting
  ↓
IFFT
  ↓
normalized residual enhancement
  ↓
enhanced image
  ↓
YOLO26
```

### Fair-comparison principles

- same starting pretrained checkpoint;
- same train/validation split;
- same seed pairing;
- same training budget;
- same augmentation and optimizer settings unless explicitly studied;
- no access to locked test set during method selection.

### Main evaluation

- mAP50;
- mAP50–95;
- Macro mAP50–95;
- Bottom-3 class mAP50–95;
- Worst-class mAP50–95;
- per-class AP / confusion where available;
- proposal/localization accessibility;
- localization-conditioned classification;
- correct-decision recall;
- parameters, latency, throughput, VRAM.

## Proposal-safe preliminary-result wording

> Studi pendahuluan pada satu seed digunakan untuk menilai kelayakan awal rancangan. Hasil tersebut tidak diperlakukan sebagai kesimpulan final dan akan divalidasi menggunakan protokol eksperimen berulang pada penelitian tesis.

Never change this to a superiority claim until the final repeated-seed protocol is complete.