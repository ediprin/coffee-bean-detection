# 03 — Research Gap, Questions, and Scope

## 1. Working title

> **Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

The wording may be refined for institutional style, but the conceptual scope should remain stable unless new evidence requires a change.

## 2. Three-layer research gap

### 2.1 Problem gap

The reviewed coffee literature provides repeated evidence that fine-grained coffee-defect recognition/detection is not adequately described by aggregate accuracy alone. As the taxonomy becomes more granular, difficult classes, visual similarity, and class-wise disparity become more visible.

The proposal should therefore focus on **fine-grained visual discrimination**, especially difficult/tail classes.

### 2.2 Methodological gap

In the reviewed coffee corpus, common responses to fine-grained difficulty are predominantly model-internal:

- backbone changes;
- specialized convolution;
- attention;
- multiscale feature fusion;
- transformer designs;
- metric/similarity learning.

By contrast, task-oriented input preprocessing and frequency-domain enhancement are well represented in adjacent detection literature, while parameter-free frequency-angular **input-space** preprocessing is not a mainstream approach in the reviewed coffee-defect corpus.

This wording is intentionally limited to the reviewed corpus. Do not claim universal absence until a systematic novelty search supports it.

### 2.3 Evaluation gap

Aggregate mAP/accuracy can hide difficult classes. The proposal therefore includes class-wise and tail-oriented evaluation, plus a diagnostic separation of classification and localization behavior.

## 3. Main research questions

### RQ1 — Overall effect

> Apakah preprocessing citra berbasis frekuensi-angular dapat meningkatkan kinerja YOLO26 dalam deteksi fine-grained cacat biji kopi dibandingkan YOLO26 tanpa preprocessing tersebut?

Primary evidence:

- Macro mAP50-95;
- mAP50 where needed;
- paired seed deltas.

### RQ2 — Difficult/tail classes

> Bagaimana pengaruh preprocessing frekuensi-angular terhadap kelas-kelas cacat yang memiliki kinerja rendah atau sulit dibedakan?

Primary evidence:

- Bottom-3 mAP50-95;
- Worst-class mAP50-95;
- per-class AP/confusion/error inspection.

### RQ3 — Classification versus localization

> Apakah perubahan kinerja yang dihasilkan preprocessing frekuensi-angular lebih berkaitan dengan kemampuan diskriminasi kelas daripada perubahan kemampuan lokalisasi objek?

Primary evidence:

- raw proposal accessibility;
- localization-conditioned Top-1 classification;
- correct-decision recall;
- supporting localization metrics where available.

### Optional RQ4 — Efficiency trade-off

> Bagaimana trade-off antara peningkatan kinerja dan biaya komputasi preprocessing frekuensi-angular pada pipeline YOLO26?

Evidence:

- detector parameter count;
- preprocessing learned parameter count (= 0 for AF2);
- latency;
- throughput;
- GPU memory.

This may be framed as evaluation rather than a standalone RQ depending on proposal format.

## 4. Research objectives

1. Formulate and integrate parameter-free frequency-angular image preprocessing into the YOLO26 input pipeline.
2. Evaluate its effect on overall fine-grained coffee-defect detection performance using matched controls.
3. Analyze its effect on difficult/tail defect classes using class-wise and lower-tail metrics.
4. Diagnose whether observed gains are more consistent with class discrimination or raw localization changes.
5. Quantify the computational trade-off introduced by the preprocessing stage.

## 5. Core experimental design

### Baseline

```text
Raw RGB -> YOLO26
```

### Proposed

```text
Raw RGB -> AF2 -> YOLO26
```

Matched conditions are mandatory.

A classical preprocessing control such as CLAHE may be added later if proposal reviewers require a stronger preprocessing baseline, but it is not part of the minimum viable core comparison.

## 6. Scope boundaries

### Included

- object detection;
- fine-grained multi-class coffee defects;
- YOLO26 baseline/proposed pipeline;
- input-space frequency-angular preprocessing;
- class-wise/tail analysis;
- classification-versus-localization diagnostic analysis;
- efficiency measurement.

### Not the main thesis

- segmentation;
- open-set recognition;
- counting as the principal objective;
- two-stage detection pipeline;
- wholesale redesign of YOLO26 architecture;
- stacking unrelated attention/neck/loss modules;
- claiming SOTA as a requirement.

## 7. Contribution wording

A defensible proposal-level contribution is:

> This research investigates a parameter-free frequency-angular image preprocessing strategy for YOLO26 and evaluates whether the strategy can improve fine-grained coffee-defect discrimination, particularly for difficult classes, while preserving localization behavior and without adding learned preprocessing parameters.

The contribution is an **analysis + controlled optimization/evaluation** contribution. It does not require claiming that the operator is globally novel or universally superior.

## 8. Title wording rule

Keep **"Analisis dan Optimasi"** only if the final thesis includes a defensible optimization dimension, for example controlled operator/configuration choices or direct-versus-alternative preprocessing analysis.

If the eventual study only evaluates one frozen AF2 operator against a baseline, a safer final title would use wording such as **"Analisis Pengaruh"** rather than "Optimasi".
