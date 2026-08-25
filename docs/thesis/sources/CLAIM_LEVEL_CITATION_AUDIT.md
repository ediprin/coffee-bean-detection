# Claim-Level Citation Audit — Proposal

Status: **ACTIVE — primary/full-text evidence gate**

Dokumen ini mengaudit apakah kalimat-kalimat penting pada BAB I–III benar-benar didukung oleh isi sumber primer/full text. Dokumen ini berbeda dari `OFFICIAL_CITATION_AUDIT.md`: metadata resmi yang benar **tidak otomatis** membuktikan klaim metodologis atau empiris yang ditulis di proposal.

## Aturan

1. Klaim empiris/metodologis harus ditautkan ke bagian, halaman, tabel, gambar, atau persamaan pada sumber primer bila tersedia.
2. Workbook/master map hanya boleh dipakai sebagai locator; ia bukan bukti akhir sebuah klaim.
3. Abstract boleh mendukung klaim yang memang eksplisit di abstract, tetapi klaim mekanisme rinci harus diperiksa pada method/full text.
4. Jangan memindahkan hasil satu paper/domain menjadi bukti kausal untuk metode tesis.
5. Jika direct primary evidence belum diekstrak pada audit ini, status tetap `PENDING DIRECT PRIMARY CHECK`, meskipun source sudah terpetakan di workbook.

---

## A. Klaim masalah fine-grained pada biji kopi

### COF-02 — Bahy & Rifai (2026)

**Status: VERIFIED — PRIMARY PDF**

Primary PDF: *Real-Time Coffee Bean Defect Detection Based on SNI 01-2907-2008 Standards Using Lightweight YOLOv5s Architecture*.

Locator yang sudah diverifikasi:
- Abstract, p. 29.
- Table VI, p. 36.
- Table VIII, p. 38.
- Conclusion/Future Work, pp. 40–41.

Klaim yang aman:
- studi mendeteksi **20 kategori cacat fisik** yang didefinisikan berdasarkan SNI 01-2907-2008;
- dataset dilaporkan terdiri dari **107 citra dan 13.863 anotasi**;
- analisis per kelas menunjukkan kelas morfologis yang jelas dapat jauh lebih mudah dideteksi daripada kelas yang ambigu secara visual;
- *slight insect damage* dilaporkan lebih lemah dan pembahasan paper mengaitkan kesulitan tersebut dengan bias tekstur dan ambiguitas kontras.

Batas klaim:
- paper ini **tidak** membuktikan bahwa domain frekuensi adalah bottleneck cacat kopi;
- paper ini **tidak** membuktikan bahwa preprocessing frekuensi-angular akan meningkatkan YOLO26.

Implikasi untuk BAB I saat ini: kalimat tentang 20 kategori dan variasi kinerja antarkelas dapat dipertahankan.

### COF-04 — Hebert & Alamsyah (2026)

**Status: VERIFIED — PRIMARY PDF**

Primary PDF: *Detection of Coffee Bean Defects in Speciality Coffee Association Standards using YOLOv12*.

Locator yang sudah diverifikasi:
- p. 94, analisis AP per kelas dan Conclusion.

Klaim yang aman:
- studi menggunakan **15 jenis cacat**;
- *Cherry Pods* dilaporkan AP 0,89, sedangkan *Floater* 0, *Fungus Damage* 0,18, dan *Slight Insect Damage* 0,15;
- penulis menjelaskan bahwa *Slight Insect Damage* dapat berupa titik hitam atau bekas gigitan kecil yang tertutup tekstur alami, *Fungus Damage* dapat memiliki warna yang mirip permukaan biji, dan *Floater* memiliki warna/bentuk yang mirip biji normal;
- penulis menyimpulkan bahwa variasi visual halus dan ukuran cacat kecil berkontribusi pada rendahnya kinerja beberapa kategori, bersama faktor dataset seperti ketidakseimbangan kelas dan keterbatasan sampel.

Batas klaim:
- temuan ini dataset-scoped dan tidak boleh ditulis sebagai hukum umum semua model/dataset kopi;
- tidak membuktikan frequency-domain causality.

Implikasi untuk BAB I saat ini: kalimat bahwa beberapa kategori memiliki AP jauh lebih rendah dan bahwa tanda kecil/tekstur/kemiripan visual berkontribusi pada kesulitan **didukung langsung** oleh primary PDF.

### COF-05 — Jundullah et al. (2026)

**Status: VERIFIED — PRIMARY PDF**

Primary PDF: *YOLOv8-Based Multi-Class Detection of Coffee Bean Defects and Contaminants for Automated Quality Grading*.

Locator yang sudah diverifikasi langsung:
- Table 3, p. 319: metrik per kelas dan nilai rata-rata;
- Fig. 6 dan Discussion, p. 320: confusion matrix serta pembahasan kategori yang mirip secara visual.

Klaim yang aman:
- paper mengevaluasi YOLOv8s pada kategori cacat/kontaminan multi-kelas dan melaporkan rata-rata precision 0,76, recall 0,75, dan mAP@0.5 0,75;
- kelas dengan karakteristik visual yang khas dilaporkan lebih mudah dikenali daripada sejumlah kelas yang mirip secara visual;
- paper secara eksplisit membahas kebingungan antara varian biji hitam serta kelas lain yang memiliki degradasi warna atau kerusakan struktural yang serupa;
- Discussion mengaitkan kesulitan dengan perbedaan visual halus, kemiripan struktural/warna, ukuran objek kecil, dan kondisi pengambilan citra top-down;
- pada konteks eksperimen mereka, penulis menyatakan tantangan utama lebih terkait dengan diskriminasi fine-grained dibanding lokalisasi objek.

Batas klaim:
- pernyataan tersebut berlaku pada dataset dan konfigurasi mereka; jangan ditulis sebagai hukum umum bahwa semua coffee-YOLO mempunyai bottleneck klasifikasi;
- paper tidak membuktikan bahwa pemrosesan frekuensi akan menyelesaikan kebingungan kelas;
- paper tidak membuktikan efektivitas preprocessing tesis.

Implikasi untuk BAB I/BAB II: kalimat bahwa kategori dengan ciri visual khas cenderung lebih mudah dikenali daripada kelas yang saling mirip **didukung langsung** oleh primary PDF.

### COF-07 — Kesiman et al. (2023)

**Status: VERIFIED — PRIMARY PDF**

Primary PDF: *Benchmarking A New Dataset for Coffee Bean Defects Classification Based on SNI 01-2907-2008*.

Locator yang sudah diverifikasi:
- Table III–IV dan discussion, p. 79;
- Conclusion, pp. 79–80.

Klaim yang aman:
- benchmark 17 kelas memberikan test accuracy **39,82% MobileNet** dan **53,35% InceptionResNetV2**;
- paper menyatakan kedua arsitektur masih kesulitan mengidentifikasi setiap jenis cacat;
- conclusion membandingkan 3-class benchmark yang relatif mudah dengan 17-class benchmark yang jauh lebih sulit.

Batas klaim:
- ini adalah **classification evidence**, bukan object-detection result;
- aman digunakan untuk mendukung diagnosis granularitas/fine-grained, bukan untuk menyimpulkan angka kinerja YOLO.

Implikasi untuk BAB I saat ini: kalimat bahwa 17 kelas jauh lebih menantang dibanding 3 kelas didukung langsung.

### COF-13 — Hu et al. (2025)

**Status: VERIFIED — PRIMARY PUBLISHER PDF**

Primary PDF: *Siamese networks for few-shot coffee bean defect detection*, LWT 235 (2025) 118631.

Locator yang sudah diverifikasi:
- p. 2: problem framing dan metode;
- p. 11: Discussion/Table 2.

Klaim yang aman:
- paper secara eksplisit menyatakan adanya **subtle visual differences between defect types** yang membutuhkan feature discrimination;
- metode menggunakan Siamese neural network dengan dual branches/shared weights dan pairwise similarity;
- discussion menyatakan studi menargetkan keterbatasan sampel dan perbedaan visual halus antar kategori.

Batas klaim:
- istilah "detection" pada judul paper ini tidak boleh otomatis diperlakukan sebagai bounding-box object detection; mekanismenya adalah pairwise/few-shot recognition/classification-oriented.

Implikasi untuk BAB I saat ini: kalimat tentang perbedaan visual halus dan penggunaan Siamese network untuk diskriminasi dapat dipertahankan, tetapi jangan menyebutnya sebagai bukti object detection berbasis bounding box.

---

## B. Analisis visual dan interpretabilitas

### COF-01 — Hong et al. (2026)

**Status: VERIFIED — PRIMARY PUBLISHER PDF**

Primary PDF: *Automated detection of defective coffee beans based on improved YOLOv10 framework*.

Locator yang sudah diverifikasi:
- §5.7 **Performance visualization and interpretability analysis**, p. 12;
- Fig. 5: comparison of activation distributions;
- Fig. 6: normalized confusion matrix.

Klaim yang aman:
- Hong et al. secara eksplisit menggunakan **EigenCAM** untuk visualisasi aktivasi saat inference;
- Fig. 5 membandingkan citra asli, activation map baseline YOLOv10, dan activation map improved model;
- paper juga menggunakan confusion-matrix diagnostics sebagai bagian analisis interpretabilitas.

Batas klaim:
- interpretasi heatmap Hong adalah interpretasi penulis pada model mereka; jangan dipindahkan sebagai bukti bahwa preprocessing tesis pasti membuat model lebih fokus;
- pada proposal, Eigen-CAM dipakai sebagai **rencana analisis kualitatif pendukung**, bukan bukti kausal tunggal.

Implikasi untuk BAB III: keberadaan subbab analisis visual dan penggunaan Eigen-CAM sebagai kandidat utama memiliki precedent langsung dari paper kopi Hong et al.

---

## C. Landasan DFT/FFT

### THEORY-01 — Gonzalez & Woods, *Digital Image Processing*, 4th ed.

**Status: OFFICIAL PUBLISHER METADATA VERIFIED / FORMULA-LEVEL FULLTEXT LOCATOR PENDING**

Sumber resmi yang telah diverifikasi: Pearson official catalog.

Metadata convention yang dikunci untuk proposal:
- Rafael C. Gonzalez dan Richard E. Woods;
- *Digital Image Processing*, Global Edition, 4th edition;
- Pearson;
- tahun sitasi: **2018**;
- ISBN-13: **9781292223049**.

Dasar pemilihan convention:
- Pearson official Global Edition page mengidentifikasi edisi ke-4 dan ©2018;
- record bibliografis ISBN yang sama mengidentifikasi Pearson, 2018, 1024 halaman;
- jangan mencampur metadata ini dengan digital-update/US listing yang memiliki ISBN atau tanggal publikasi berbeda.

Cakupan sumber yang aman:
- Chapter 4 adalah *Filtering in the Frequency Domain*;
- buku tersebut merupakan sumber fundamental untuk transformasi Fourier/DFT pada citra dan pemrosesan domain frekuensi.

Gate yang belum selesai:
- formula DFT/iDFT dan definisi amplitude/phase yang saat ini tertulis di BAB II belum mempunyai page-level locator dari salinan full text resmi yang tersimpan di project source;
- karena itu, THEORY-01 dapat digunakan sebagai **bibliographic/theoretical anchor**, tetapi audit halaman formula tetap ditandai terbuka sampai halaman buku yang dipakai tersedia secara langsung.

Larangan:
- jangan mengklaim buku ini membuktikan AFAB-2 efektif;
- jangan mengklaim buku ini membuktikan cacat kopi memiliki signature frekuensi tertentu.

---

## D. Parent mechanism frekuensi-angular — Xu et al. (2025)

### FG-01 — LFDet / AFAB / AFAB-2

**Status: VERIFIED — PRIMARY PUBLISHER PDF, FORMULA-LEVEL**

Primary paper: X. Xu, Z. Chen, Y. Hu, & G. Wang, *More Signals Matter to Detection: Integrating Language Knowledge and Frequency Representations for Boosting Fine-Grained Aircraft Recognition*, *Neural Networks*, 187 (2025), 107402, DOI `10.1016/j.neunet.2025.107402`.

Locator primer yang sudah dibuka langsung:
- §3.3 **Adaptive Frequency Augmentation Branch (AFAB)**, pp. 4–6;
- §3.3.1 **Patch-wise DFT**, p. 5;
- §3.3.3, pp. 5–6, Eq. (9)–(13);
- §4.4.1 dan Table 6, p. 13, untuk pemisahan AFAB-1 dan AFAB-2;
- Table 8 dan pembahasannya, p. 15, untuk sensitivitas `γ`.

#### D.1 Patch-wise DFT

Klaim sumber yang terverifikasi:
- Xu et al. membagi citra menggunakan sliding window menjadi patch berukuran `m × m`;
- paper menetapkan `m = 32` karena maximum downsampling rate feature space pada rancangan mereka adalah 32;
- DFT dilakukan pada setiap patch dan patch-wise iDFT digunakan untuk kembali ke domain spasial;
- paper menyatakan menggunakan **large overlap** untuk mengurangi diskontinuitas akibat pseudo high-frequency components pada tepi patch.

Batas klaim:
- paper **tidak memberikan angka 50% overlap** pada teks primer yang telah diverifikasi;
- karena itu, overlap 50% / stride 16 pada proposal adalah **keputusan implementasi penelitian**, bukan parameter yang boleh diatributkan kepada Xu et al.;
- `m=32` boleh disebut mengikuti reference setting Xu, tetapi tidak boleh dinyatakan optimal untuk citra kopi.

#### D.2 Distribusi angular dan Eq. (9)

Xu et al. mendefinisikan angular density distribution:

\[
D_i^P(\theta)=\sum_r A_i^P(r\cos\theta,r\sin\theta),
\qquad \theta\in[0,360^\circ).
\]

Paper menginterpretasikan density yang lebih tinggi sebagai respons arah yang memiliki struktur edge/texture lebih nyata pada data mereka, sedangkan density rendah berkaitan dengan arah yang strukturnya lebih lemah dan lebih mungkin memuat gangguan.

Batas klaim:
- domain kontinu `θ ∈ [0,360°)` pada paper **bukan sama dengan** keputusan implementasi memakai 360 discrete bins;
- 360 bin adalah diskretisasi penelitian/repository dan harus ditulis sebagai keputusan implementasi;
- interpretasi edge/texture tersebut berasal dari data fine-grained aircraft remote sensing dan tidak boleh dipindahkan sebagai karakteristik fisik pasti dari cacat kopi.

#### D.3 Entropi, adaptive threshold, dan Eq. (10)–(13)

Xu et al. secara eksplisit menuliskan:

\[
E^P_{i,A}=-\sum_\theta D_i^{*P}(\theta)\log D_i^{*P}(\theta),
\]

\[
t_i^P=\frac{\gamma}{1+\exp(-E^P_{i,A})},
\]

kemudian menekan arah dengan normalized angular density di bawah/equal threshold dan mempertahankan normalized density untuk arah lainnya. Adjusted amplitude dibentuk melalui perkalian amplitude dengan angular density yang telah diproses, lalu direkonstruksi menggunakan **adjusted amplitude + original phase + iDFT**.

Dengan demikian, komponen berikut memang memiliki parent mechanism langsung dari Xu:
- normalisasi angular density menjadi distribusi;
- information entropy;
- logistic adaptive threshold dengan `γ`;
- hard suppression untuk normalized density yang tidak melewati threshold;
- pembobotan amplitude berdasarkan respons angular;
- mempertahankan fase asli saat rekonstruksi.

Batas klaim implementasi:
- penambahan `ε` untuk numerical stability adalah detail implementasi penelitian;
- indeks kanal `c` dan keputusan memproses RGB per-channel secara independen adalah detail implementasi penelitian kecuali ditemukan pernyataan eksplisit lain pada sumber primer;
- operasi implementasi yang mengalikan complex spectrum dengan bobot real dapat dijelaskan sebagai instansiasi yang mempertahankan fase, tetapi teks proposal sebaiknya mengikuti formulasi sumber: **adjusted amplitude dipasangkan dengan original phase**.

#### D.4 Nilai gamma

Xu et al. menguji `γ = 0, 0.05, 0.1, 0.15, 0.2` pada Table 8 dan melaporkan `γ = 0.1` sebagai nilai terbaik pada tiga benchmark mereka. Paper kemudian menyatakan menggunakan `γ = 0.1` pada seluruh eksperimen.

Implikasi proposal:
- `γ = 0.10` boleh digunakan sebagai **reference initialization** berdasarkan Xu et al.;
- nilai tersebut **tidak boleh diasumsikan optimal untuk kopi**;
- proposal sudah tepat bila memasukkan `γ` ke planned sensitivity analysis.

#### D.5 AFAB-1 tidak sama dengan AFAB-2

Xu et al. memisahkan:
- AFAB-1 = *patch-specific adaptive high-pass filter*;
- AFAB-2 = *patch-specific chaotic amplitude suppressor* berbasis distribusi angular.

Table 6/7 memperlakukan keduanya sebagai subkomponen terpisah. Karena proposal saat ini berfokus pada mekanisme angular AFAB-2, reference configuration **tidak boleh ditulis seolah menggunakan radial/high-pass AFAB-1**.

#### D.6 Rekonstruksi dan gating: bagian sumber vs adaptasi tesis

Xu et al. menyatakan raw spatial domain dan recovered spatial domain difusikan melalui **gating mechanism**, dengan recovered space berfungsi mengatur information flow pada raw spatial domain.

Namun, dari passage primer yang sudah diekstrak, exact thesis implementation berikut belum boleh diklaim sebagai persamaan Xu:

\[
G(I)=\operatorname{MinMax}(R_{FA}(I)),
\]

\[
I'=I+I\odot G(I).
\]

Statusnya:
- overlap fold/average reconstruction = implementasi penelitian;
- min-max normalization = implementasi penelitian;
- exact residual gate `I + I⊙G(I)` = instansiasi/adaptasi penelitian;
- parameter-free frontend ≠ compute-free; biaya FFT/iFFT tetap harus dievaluasi.

### Ringkasan provenance yang harus konsisten dengan BAB III

| Elemen | Status |
|---|---|
| Patch-wise DFT/iDFT | Sumber Xu et al. |
| `m=32` | Reference setting Xu; dipakai sebagai nilai awal, bukan optimum kopi |
| 50% overlap / stride 16 | Adaptasi implementasi penelitian |
| Angular density `θ∈[0,360°)` | Sumber Xu, Eq. (9) |
| 360 discrete bins | Diskretisasi implementasi penelitian |
| Entropy + logistic threshold | Sumber Xu, Eq. (10)–(11) |
| Hard angular suppression | Sumber Xu, Eq. (12) |
| Amplitude weighting | Sumber Xu, Eq. (13) |
| Original phase + iDFT | Sumber Xu |
| `γ=0.1` | Reference setting Xu; harus diuji sensitivitas pada kopi |
| `ε` numerical stabilization | Implementasi penelitian |
| Per-channel RGB processing | Implementasi penelitian |
| Fold/average overlap reconstruction | Implementasi penelitian |
| Min-max normalization | Implementasi penelitian |
| `I'=I+I⊙G(I)` | Adaptasi gating penelitian |
| AFAB-1 radial/high-pass | **Tidak digunakan** dalam reference angular-only preprocessing tesis |

Batas klaim terbesar:
- Xu et al. adalah **parent method pada aircraft remote sensing**, bukan validasi transfer ke biji kopi;
- proposal harus tetap menyatakan efektivitas pada coffee-YOLO sebagai pertanyaan empiris.

---

## E. Evaluasi COCO / mAP50–95

### EVAL-02 — Official COCOeval implementation

**Status: VERIFIED — OFFICIAL COCO API SOURCE CODE**

Sumber resmi: repository organisasi `cocodataset/cocoapi`, file `PythonAPI/pycocotools/cocoeval.py`.

Locator yang diverifikasi langsung pada source code resmi:
- komentar parameter evaluasi menyatakan `iouThrs - [.5:.05:.95] T=10 IoU thresholds`;
- implementasi `setDetParams()` membentuk threshold IoU dari 0,50 sampai 0,95 dengan langkah 0,05;
- evaluator mendukung `bbox` sebagai salah satu `iouType`.

Klaim yang aman:
- jika proposal menyebut evaluasi gaya COCO dengan AP yang dirata-ratakan pada IoU 0,50:0,05:0,95, definisi rentang threshold tersebut memiliki sumber implementasi resmi;
- konfigurasi proposal tetap harus mengikuti evaluator yang benar-benar digunakan pada eksperimen final.

Batas klaim:
- jangan menyatakan semua output Ultralytics identik dengan seluruh konfigurasi COCOeval tanpa memeriksa implementasi evaluator yang dipakai;
- `max_det=500` pada eksperimen tesis adalah konfigurasi penelitian dan tidak boleh disamakan dengan default `maxDets` COCOeval.

---

## F. Status snapshot claim-level gate

```text
Bahy 2026        = VERIFIED primary PDF
Hebert 2026      = VERIFIED primary PDF
Jundullah 2026   = VERIFIED primary PDF, Table 3 p.319 + Discussion p.320
Kesiman 2023     = VERIFIED primary PDF
Hu 2025          = VERIFIED primary publisher PDF
Hong 2026 XAI    = VERIFIED primary publisher PDF, §5.7 p.12
Xu 2025 AFAB-2   = VERIFIED primary PDF sampai Eq. (9)–(13) + gamma sensitivity
COCOeval         = VERIFIED official cocodataset/cocoapi implementation
DFT/FFT textbook = official Pearson metadata locked; page-level formula locator masih pending
```

Proposal **belum disebut citation-ready penuh** karena halaman formula DFT/iDFT dan amplitude/phase dari textbook fundamental belum tersedia sebagai primary full-text project source. Semua claim-level gate lain yang tercantum pada snapshot di atas sudah ditutup.