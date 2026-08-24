# Canonical Source Keys

Purpose: provide one stable citation-key namespace for all proposal drafts and prevent the same key from referring to different papers in different files.

## Authority order

When a key conflict exists, use this precedence:

1. latest `AF2_Proposal_Master_Reference_Map_BAB2_READY.xlsx` master reference ID;
2. this canonical registry;
3. section-specific working files;
4. older conversation/ad-hoc aliases.

No new proposal draft may invent a key that collides with an existing canonical key.

## Coffee / domain keys

| Key | Canonical source |
|---|---|
| STD-01 | BSN — SNI 01-2907-2008 Biji Kopi |
| REV-01 | Motta et al. — coffee ML/CV review |
| COF-01 | Hong et al. 2026 — improved YOLOv10 coffee defects |
| COF-02 | Bahy & Rifai 2026 — lightweight YOLOv5s, SNI 20-class |
| COF-03 | Samudra & Rachmawati 2025 — LSKNet + oriented detector |
| COF-04 | Hebert & Alamsyah 2026 — YOLOv12 SCA-style defect detection |
| COF-05 | Jundullah et al. 2026 — YOLOv8 multi-class defects/contaminants |
| COF-06 | Gope et al. 2024 — YOLO-family green-coffee benchmark |
| COF-07 | Kesiman et al. 2023 — SNI 3-vs-17-class benchmark |
| COF-08 | Arwatchananukul et al. 2024 — 17-class Thai Arabica defects |
| COF-09 | Lei et al. 2025 — decoupled classification/localization coffee workflow |
| COF-10 | de Oliveira et al. 2016 — traditional CV/computational intelligence coffee classification |
| COF-11 | Chang & Huang 2021 — deep-learning coffee defect inspection |
| COF-12 | Jiao et al. 2025 — Swin-HSSAM coffee grading |
| COF-13 | Hu et al. 2025 — Siamese few-shot coffee defect recognition |
| COF-14 | Muchtar et al. 2025 — edge AI defective coffee beans |
| COF-15 | Hsia et al. 2022 — explainable/lightweight green-coffee quality detection |
| COF-16 | Gope et al. 2025 — cross-family coffee benchmark |
| COF-17 | García, Candelo-Becerra & Hoyos 2019 — computer-vision quality/defect inspection of green coffee beans |
| COF-SUP-01 | Kesiman et al. 2024 — Coffection web-based SNI grading application |

## Detector / evaluation keys

| Key | Canonical source |
|---|---|
| DET-01 | Jocher et al. 2026 — Ultralytics YOLO26 preprint |
| DET-02 | Ren et al. 2015 — Faster R-CNN |
| DET-03 | Redmon et al. 2016 — original YOLO |
| EVAL-01 | Lin et al. 2014 — Microsoft COCO |
| EVAL-02 | COCOeval official implementation/specification |
| DIAG-01 | Feng et al. 2021 — TOOD |
| DIAG-02 | Wu et al. 2020 — Rethinking Classification and Localization |
| DIAG-03 | Jiang et al. 2018 — IoU-Net |

## Fine-grained keys

| Key | Canonical source |
|---|---|
| FG-01 | Xu et al. 2025 — LFDet / AFAB / AFAB-2 |
| FG-02 | Xie et al. 2025 — discriminative representation for FGOD |
| FG-03 | Wang et al. 2020 — cross-domain fine-grained recognition |

## Preprocessing keys

| Key | Canonical source |
|---|---|
| PRE-01 | Liu et al. 2022 — IA-YOLO |
| PRE-02 | Qin et al. 2022 — DENet / DE-YOLO |
| PRE-03 | Li et al. 2025 — FE-YOLO |
| PRE-04 | Syauqi et al. 2025 — CLAHE-based white-pepper YOLO |
| PRE-05 | Chen et al. 2024 — maize-seed X-ray preprocessing + YOLOv8 |
| PRE-06 | Tu et al. 2026 — WCTE transform-domain preprocessing + YOLO |
| PRE-07 | Cai et al. 2023 — Retinexformer |
| PRE-08 | Yang & Soatto 2020 — Fourier Domain Adaptation |

## Fourier / spectral / wavelet keys

The following namespace resolves a previous collision: older draft files used `FREQ-01/FREQ-02` for Cao and Zhang & Tan, while the master reference map already uses those keys for FFC and FDADNet. The master-map meaning is canonical.

| Key | Canonical source |
|---|---|
| THEORY-01 | Gonzalez & Woods — Digital Image Processing, 4th ed. |
| THEORY-02 | Bracewell — The Fourier Transform and Its Applications, 3rd ed. |
| SPEC-01 | Cao et al. 2019 — radial/angular Fourier-energy texture analysis |
| SPEC-02 | Zhang & Tan 2003 — orientation-spectrum texture signatures |
| WAVE-01 | Finder et al. 2024 — WTConv |
| FREQ-01 | Chi, Jiang & Mu 2020 — Fast Fourier Convolution |
| FREQ-02 | Li et al. 2024 — FDADNet |
| FREQ-03 | Chen et al. 2025 — Frequency Dynamic Convolution |
| AGR-01 | Zhao et al. 2026 — wavelet-based WGA-YOLO / WCR multi-crop disease detection |
| AGR-02 | PFENet 2026 — frequency-enhanced YOLO in hazy scenes |

## Deprecated aliases

Do not use these old meanings in new prose:

```text
FREQ-01 = Cao et al. 2019      # deprecated
FREQ-02 = Zhang & Tan 2003     # deprecated
FREQ-03 = WTConv               # deprecated
FREQ-04 = FDConv               # deprecated
```

Replace with:

```text
SPEC-01 = Cao et al. 2019
SPEC-02 = Zhang & Tan 2003
WAVE-01 = WTConv
FREQ-03 = FDConv
```

## Drafting rule

Before a proposal file is promoted from `structural draft` to `citation-ready draft`, all bracket keys must resolve uniquely through this registry or the master reference workbook. Numerical claims still require primary-PDF verification even when the key is canonical.
