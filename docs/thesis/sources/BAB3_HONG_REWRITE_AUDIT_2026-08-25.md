# Bab III Hong-Adapted Rewrite Audit — 2026-08-25

Status: **Bab III rewritten; cross-chapter normalization still required before DOCX assembly**.

## 1. Source-of-truth order

1. `foundation/08_BAB3_HONG_ADAPTED_OPTIMIZATION_DESIGN.md`
2. `FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_PROTOCOL_2026-08-24.md`
3. `FARUQ_V3_AF2_SPECTRAL_FACTORIZATION_PROTOCOL.md`
4. AF2 implementation/config files
5. proposal prose

Hong et al. is a methodological template for systematic ablation, sensitivity analysis, visualization and error analysis. Its YOLOv10 modules and 5-fold protocol are not copied.

## 2. Bab III structure after rewrite

The active `proposal/05_METHODOLOGY.md` now contains:

- 3.1 Kerangka Penelitian
- 3.2 Dataset Penelitian
- 3.3 Baseline YOLO26
- 3.4 Arsitektur Metode yang Diusulkan
- 3.5 Preprocessing Frekuensi-Angular AF2
- 3.6 Analisis dan Optimasi AF2
- 3.7 Rancangan Eksperimen Konfirmatori
- 3.8 Konfigurasi Pelatihan
- 3.9 Metrik Evaluasi
- 3.10 Analisis Mekanisme
- 3.11 Analisis Visualisasi
- 3.12 Analisis Kesalahan
- 3.13 Evaluasi Efisiensi
- 3.14 Lingkungan Implementasi dan Reproducibility
- 3.15 Batas Studi Pendahuluan, Optimasi, dan Bukti Final

## 3. Optimization-title alignment

`Optimasi` is now operationalized as:

```text
factorized AF2 structural analysis
-> optional/limited parameter sensitivity
-> configuration selection
-> method freeze
```

Primary structural candidates:

- AF2C
- AF2WIN
- AF2ORI
- AF2POL
- AF2SOFT
- AF2LUM

PCG1 and WAV1 remain optional mechanistic comparators, not AF2 variants.

## 4. Important protocol distinction

The historical spectral-factorization screen used seed-matched D0 coffee checkpoints. The final confirmatory design uses the official YOLO26n pretrained artifact directly with matched target-head initialization.

Therefore:

- factorization genealogy = development/selection evidence;
- direct native-vs-AF2 confirmation = main thesis evidence.

The proposal must not silently merge these two provenance layers.

## 5. Analysis-title alignment

`Analisis` is now operationalized as:

- Macro mAP50-95;
- Bottom-3;
- Worst-class;
- per-class AP;
- proposal accessibility;
- localization-conditioned Top-1;
- correct-decision recall;
- AF2 input/spectral visualization;
- activation visualization only after YOLO26 compatibility audit;
- confusion/per-class error analysis;
- paired rescue-regression analysis;
- latency, throughput, parameters and VRAM.

## 6. Claim boundaries

PASS:

- AF2 is input preprocessing, not a YOLO module.
- Parameter-free is not compute-free.
- Tail metrics are study-defined.
- Proposal accessibility is a limited proxy, not complete localization quality.
- Visualization is interpretive support, not causal proof.
- Negative candidate results remain valid development evidence.
- Locked test is not used for model selection.

## 7. Open cross-chapter normalization

Before first DOCX release, Bab I must be updated so the explicit optimization design appears in the research questions/objectives. The current three-RQ draft was written before `foundation/08` and therefore underrepresents the word `Optimasi` in the title.

Recommended normalized RQ layout:

1. AF2 structural/parameter optimization question;
2. native-vs-selected-AF2 confirmatory effectiveness;
3. lower-tail/difficult-class effect;
4. discrimination-vs-proposal-accessibility diagnostic pattern.

The exact wording must be synchronized across:

- `proposal/03_PROBLEM_FORMULATION.md`;
- `proposal/01_PROPOSAL_SKELETON.md`;
- `sources/PROPOSAL_CROSS_CHAPTER_AUDIT.md`.

## 8. DOCX readiness

Bab III is not yet release-ready. Remaining gates:

- exact parent-paper equation/page citations for AFAB/LFDet-derived equations;
- final citation conversion from internal keys to APA author-year;
- exact parameter-sensitivity grid if that study is retained;
- YOLO26-compatible activation-visualization audit;
- figure generation for overall architecture, AF2 pipeline, and optimization flow;
- cross-chapter RQ/objective synchronization;
- USU template assembly and visual QA.