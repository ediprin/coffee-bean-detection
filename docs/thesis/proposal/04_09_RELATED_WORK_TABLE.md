# 2.9 Penelitian Terkait — Working Source-Normalized Table

Status: **first-pass citation-ready synthesis**. This file is a modular companion to `04_LITERATURE_REVIEW.md` and will be merged into §2.9 when the full proposal document is generated.

Purpose: follow the campus-style related-work table while maintaining literature breadth. The table deliberately combines three evidence streams rather than showing only coffee-YOLO papers:

1. direct coffee / fine-grained coffee evidence;
2. preprocessing-before-detector evidence;
3. fine-grained / frequency methodological evidence.

Numerical results remain paper-scoped. A row's result may not be transferred to another dataset, detector, or domain.

## Tabel 2.1. Perbandingan Penelitian Terkait

| No | Penulis & Tahun | Indeks / Venue | Fokus Penelitian | Metode / Model | Kontribusi dan Pengisian Gap Penelitian |
|---:|---|---|---|---|---|
| 1 | Hong et al. (2026) [COF-01] | **Q1 — Current Research in Food Science** | Deteksi multi-class cacat green coffee | Improved YOLOv10: DSConv + SPPF-Attention + PConv | Menunjukkan YOLO modern dapat mencapai performa tinggi pada setup tujuh kelas, tetapi penulis tetap memperlakukan kemiripan visual defect sebagai tantangan teknis. Menjadi **pivot** pola penelitian: problem kopi → mekanisme dari literatur CV → validasi pada kopi. Gain modul Hong tidak ditransfer ke AF2. |
| 2 | Gope et al. (2024) [COF-06] | **Q1 — Scientific Reports** | Green-coffee defect detection, empat kelas | YOLOv3/v4/v5/v7/v8 + custom YOLOv8n | Menetapkan keluarga YOLO sebagai baseline yang praktis pada domain green coffee. Taxonomy empat kelas relatif coarse, sehingga hasil mendekati saturasi tidak menunjukkan bahwa deteksi 15–20 kelas sudah terselesaikan. |
| 3 | Bahy & Rifai (2026) [COF-02] | **SINTA 3 — International Journal on ICT** | Deteksi 20 kategori fisik berbasis SNI | Lightweight YOLOv5s + transfer learning/tuning | Memberi direct evidence pada taxonomy besar: P=0.817, R=0.816, mAP50=0.867, mAP50–95=0.601, dengan heterogenitas antarkelas. Mendukung evaluasi class-wise/lower-tail; tidak membuktikan penyebab kesulitan berada di frekuensi. |
| 4 | Jundullah et al. (2026) [COF-05] | **SINTA 3 — Brilliance** | Deteksi 20 kelas cacat/kontaminan | YOLOv8s | Mean mAP@0.5=0.75 tetapi performa antar kelas sangat berbeda; paper secara eksplisit menyatakan kelas dengan ciri visual khas lebih mudah dibanding defect yang visually similar. Bukti langsung untuk problem fine-grained discrimination. |
| 5 | Hebert & Alamsyah (2026) [COF-04] | **SINTA 3 — INOVTEK Polbeng** | Deteksi 15 kategori defect bergaya SCA | YOLOv12 | Melaporkan beberapa subtle classes jauh di bawah kelas khas; floater, fungus damage, dan slight insect damage menjadi contoh tail difficulty. Evidence tetap dataset-scoped dan tidak digunakan sebagai bukti frequency bottleneck. |
| 6 | Samudra & Rachmawati (2025) [COF-03] | **ICoDSA 2025 — conference** | Deteksi defect Arabica green coffee | Oriented R-CNN + LSKNet | LSKNet-S dilaporkan mAP0.5=0.879 vs YOLOv8s=0.856 pada setup tiga defect. Yang paling relevan bagi tesis adalah pembahasan misclassification black vs partially black karena kemiripan visual, bukan headline mAP-nya. |
| 7 | Arwatchananukul et al. (2024) [COF-08] | **Q1 — Smart Agricultural Technology** | Fine-grained classification 17 jenis cacat | Transfer-learning CNN; MobileNetV3 terbaik | 5-fold CV sangat tinggi tetapi unseen-data accuracy turun menjadi 88.63%. Digunakan sebagai diagnostic classification evidence bahwa aggregate controlled performance tidak selalu mewakili behavior pada data baru. Bukan object-detection evidence. |
| 8 | Jiao et al. (2025) [COF-12] | **Q1 — PLOS ONE** | Grading dan subdivisi defect green coffee | Swin Transformer + HS-FPN + selective attention + Fusion Loss | Menunjukkan salah satu respons coffee literature terhadap fine-grained difficulty adalah memperkaya **internal representation** melalui multistage fusion dan attention. Berbeda dari tesis yang mengubah input sebelum detector. |
| 9 | Hu et al. (2025) [COF-13] | **Q1 — LWT** | Few-shot recognition enam defect coffee | Siamese network | Paper secara eksplisit menargetkan *subtle visual differences between defect categories*; Siamese accuracy 94.95% vs conventional CNN 74.35% pada protokol mereka. Mendukung problem diskriminasi, bukan bounding-box performance. |
| 10 | Liu et al. (2022), IA-YOLO [PRE-01] | **AAAI 2022 — conference** | Task-driven preprocessing untuk adverse-weather detection | Differentiable image-processing filters + CNN-PP + YOLOv3 | Menunjukkan preprocessing dapat dipelajari untuk **downstream detection utility**, dan kualitas visual yang membaik tidak otomatis berarti deteksi membaik. Berbeda dari AF2 karena frontend IA-YOLO learned/adaptive. |
| 11 | Syauqi et al. (2025) [PRE-04] | **IEEE ICONS-IoT 2025 — conference** | White-pepper defect detection dengan preprocessing sebelum YOLO | Gamma correction + CLAHE + blending + NLM denoising + unsharp masking + YOLOv8m | Pada matched 50-epoch setup, mAP50–95 dilaporkan 79%→82%. Analog komoditas berbentuk biji yang kuat untuk desain raw-vs-preprocessed detector, tetapi hanya dua kelas dan treatment-nya **composite CLAHE-based**, bukan CLAHE tunggal. |
| 12 | Chen et al. (2024) [PRE-05] | **Q1 — Computers and Electronics in Agriculture (2024 metrics)** | Soft-X-ray maize-seed crack detection | Wavelet denoising + standardization + bilateral filtering + Laplacian sharpening + YOLOv8 | Paper memisahkan kontribusi preprocessing dan architecture optimization; image enhancement dilaporkan memberi tambahan 1.8 percentage points AP pada setup mereka. Berguna sebagai agricultural seed precedent, bukan estimasi gain kopi. |
| 13 | Li et al. (2025), FE-YOLO [PRE-03] | **Q2 — Digital Signal Processing (2024 JCR/SJR)** | Fourier enhancement sebelum object detector pada low-light | FFT → learned FENet/FPB amplitude/phase processing → IFFT → YOLO | Pembanding metodologis input-space Fourier paling dekat selain AFAB: Fourier enhancement dilakukan sebelum detector, tetapi learned dan berorientasi low-light. Mendukung plausibility, bukan coffee transfer. |
| 14 | Xu et al. (2025) [FG-01] | **Q1 — Neural Networks** | One-stage fine-grained aircraft detection | LFDet; AFAB-1/AFAB-2 + CGFI + FTIF | **Parent mechanism utama.** AFAB menggunakan patch-wise frequency processing; AFAB-2 memakai adaptive angular-amplitude suppression. Ablation AFAB-2 sendiri: MAR20 82.90→84.21 dan FAIRPlane 45.20→45.64 mAP50. Full LFDet gain tidak boleh diatribusikan ke AFAB-2 saja; aircraft→coffee transfer belum tervalidasi. |
| 15 | Xie et al. (2025) [FG-02] | **Q1 — IEEE TCSVT** | Fine-grained object detection pada remote sensing | DRNet + fine-grained branch + refinement + confusion-minimized loss | Independent high-quality evidence bahwa FGOD merupakan persoalan localization **dan** subordinate-category discrimination, serta dapat mengalami representation conflict/misalignment. Tidak mengharuskan DRNet pada kopi. |
| 16 | Chi, Jiang & Mu (2020) [FREQ-01] | **NeurIPS 2020** | Spatial–spectral representation pada general vision | Fast Fourier Convolution | Menunjukkan local/spatial dan global/spectral processing dapat dipadukan dalam feature representation. Digunakan sebagai general frequency precedent; bukan input preprocessing dan bukan bukti coffee frequency separability. |
| 17 | Li et al. (2024) [FREQ-02] | **Q2 — Processes** | Low-contrast industrial surface-defect detection | FDADNet: spatial/frequency representation + adaptive downsampling | Direct defect-domain evidence bahwa spatial detail dan frequency representation dapat dikombinasikan. Target-vs-background low contrast berbeda dari inter-class coffee similarity. |
| 18 | Chen et al. (2025) [FREQ-03] | **CVPR 2025** | Adaptive frequency processing untuk dense prediction | Frequency Dynamic Convolution | Memberi independent top-tier precedent bahwa modulasi frekuensi dapat dibuat content-adaptive pada dense vision. Mekanismenya internal convolution dan berbeda dari AFAB-2/AF2. |
| 19 | **Penelitian yang Diusulkan** | — | Deteksi fine-grained cacat biji kopi dengan preprocessing input | **AF2 parameter-free frequency-angular preprocessing + native YOLO26** | Menguji titik temu yang belum dapat disimpulkan dari dua literatur: coffee papers menetapkan fine-grained discrimination difficulty, sedangkan preprocessing/frequency papers menetapkan candidate mechanism. Kontribusi diuji melalui matched baseline, aggregate + lower-tail metrics, classification/localization diagnostics, serta accuracy–efficiency trade-off. |

## Sintesis tabel

Tabel di atas tidak dimaksudkan sebagai ranking lintas-paper karena dataset, taxonomy, task, dan evaluation protocol berbeda. Polanya dibaca dalam tiga lapis.

**Pertama, domain kopi.** Gope dan Hong menunjukkan bahwa keluarga YOLO merupakan pilihan yang layak pada setup coffee detection, tetapi Bahy, Jundullah, Hebert, Samudra, Arwatchananukul, Jiao, dan Hu menunjukkan bahwa ketika taxonomy menjadi lebih rinci atau kelas lebih mirip, kesulitan dapat berpindah ke diskriminasi antarkategori, class-wise tail, atau generalization. Karena itu, problem tesis tidak dirumuskan sebagai “YOLO gagal”, melainkan sebagai kebutuhan mempertahankan cue yang cukup diskriminatif pada taxonomy cacat yang fine-grained.

**Kedua, preprocessing.** IA-YOLO, Syauqi, Chen-maize, dan FE-YOLO menunjukkan beberapa cara berbeda untuk mengubah citra sebelum detector: learned task-driven filters, composite fixed enhancement, transform/spatial enhancement, dan learned Fourier enhancement. Kesamaan yang relevan adalah preprocessing harus dinilai melalui downstream task; perbedaannya adalah tidak satu pun treatment tersebut identik dengan AF2.

**Ketiga, frequency/fine-grained mechanisms.** Xie memberi independent FGOD theory; FFC, FDADNet, dan FDConv menunjukkan bahwa spectral/frequency representation telah digunakan di general vision, defect detection, dan dense prediction; Xu memberi parent method yang secara langsung menggabungkan frequency-angular processing dengan fine-grained object detection. Karena evidence utama Xu berasal dari aircraft imagery, efektivitas transfer ke coffee-YOLO tetap menjadi pertanyaan empiris.

Rantai positioning penelitian karena itu adalah:

```text
coffee literature
    -> fine-grained discrimination problem

preprocessing literature
    -> input transformation can affect downstream detection

frequency / FGOD literature
    -> spectral + angular processing is a plausible representation mechanism

unresolved transfer
    -> parameter-free frequency-angular preprocessing + YOLO26 on coffee defects
```

## Anti-overclaim note

Table 2.1 must never be summarized as “frequency processing has been proven to solve coffee defect detection.” The strongest defensible synthesis is:

> Dalam literatur yang ditinjau, penelitian kopi menunjukkan adanya masalah diskriminasi pada kategori cacat yang visualnya berdekatan, sementara penelitian di luar domain kopi menunjukkan bahwa preprocessing dan representasi frekuensi dapat memengaruhi utility fitur untuk berbagai tugas vision. Efektivitas preprocessing frekuensi-angular parameter-free pada fine-grained coffee defect detection belum dapat disimpulkan dari kedua kelompok literatur tersebut dan karena itu diuji secara empiris pada penelitian ini.
