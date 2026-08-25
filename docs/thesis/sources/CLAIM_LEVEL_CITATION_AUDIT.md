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
- Full-text audit sebelumnya juga memetakan Table VI p. 36, Table VIII p. 38, Conclusion/Future Work pp. 40–41; gunakan halaman ini hanya setelah direct page check bila klaim tabel spesifik akan ditulis.

Klaim yang aman:
- studi mendeteksi **20 kategori cacat fisik** yang didefinisikan berdasarkan SNI 01-2907-2008;
- dataset dilaporkan terdiri dari **107 citra dan 13.863 anotasi**;
- analisis per kelas menunjukkan kelas morfologis yang jelas dapat jauh lebih mudah dideteksi daripada kelas yang ambigu secara visual;
- abstract secara eksplisit menyebut *slight insect damage* lebih lemah dan mengaitkannya dengan **texture bias** dan **contrast ambiguity**.

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

**Status: PENDING DIRECT PRIMARY CHECK**

Primary PDF sudah memiliki locator repository/File Library dan workbook full-text audit sebelumnya mencatat 2.000 citra, 3.983 object labels, 20 kelas, serta perbedaan antara kategori visual yang khas dan varian black/sour yang mirip. Namun pada pass audit ini direct excerpt dari PDF primer belum berhasil diekstrak.

Tindakan:
- jangan memperkuat atau memperinci klaim Jundullah lebih jauh sampai direct PDF excerpt/page locator dibuka kembali;
- kalimat formal yang bergantung pada interpretasi "kategori khas lebih mudah daripada kategori mirip" tetap harus dianggap **belum ditutup claim-level gate**.

### COF-07 — Kesiman et al. (2023)

**Status: VERIFIED — PRIMARY PDF**

Primary PDF: *Benchmarking A New Dataset for Coffee Bean Defects Classification Based on SNI 01-2907-2008*.

Locator yang sudah diverifikasi:
- Table IV dan Conclusion, p. 79.

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

Metadata yang aman:
- Rafael C. Gonzalez dan Richard E. Woods;
- *Digital Image Processing*, 4th edition;
- Pearson;
- official Pearson catalog menempatkan **Filtering in the Frequency Domain** sebagai Chapter 4 dan secara eksplisit mencantumkan *The Discrete Fourier Transform (DFT)* pada daftar isi.

Catatan edisi:
- halaman Pearson untuk cetakan/global edition menunjukkan metadata publikasi 2017/©2018, sementara digital-update listing yang lebih baru memiliki tanggal publikasi berbeda. Proposal harus memilih **satu edition/ISBN convention** dan tidak mencampur tahun dari listing berbeda.

Gate yang belum selesai:
- formula DFT/iDFT dan definisi amplitude/phase yang saat ini tertulis di BAB II belum diberi page/section locator full text dari textbook;
- sampai locator full text ditutup, jangan menganggap penambahan nama buku saja sudah menyelesaikan formula-level audit.

Tindakan berikut:
1. tetapkan edition/ISBN yang benar-benar akan dipakai;
2. cek halaman/subbagian DFT/iDFT dan amplitude/phase pada full text;
3. baru tambahkan sitasi ke BAB II dan entri bibliography;
4. setelah itu rerun citation crosswalk dan bidirectional audit.

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

## E. Status snapshot claim-level gate

```text
Bahy 2026       = VERIFIED primary PDF
Hebert 2026     = VERIFIED primary PDF
Kesiman 2023    = VERIFIED primary PDF
Hu 2025         = VERIFIED primary publisher PDF
Hong 2026 XAI   = VERIFIED primary publisher PDF
Xu 2025 AFAB-2  = VERIFIED primary PDF sampai Eq. (9)–(13) + gamma sensitivity
Jundullah 2026  = PENDING direct primary excerpt in current pass
DFT/FFT textbook= official metadata verified; formula-level fulltext locator pending
```

Proposal **belum boleh disebut citation-ready penuh** sampai klaim sensitif yang masih pending ditutup atau kalimat yang bergantung padanya direvisi/dihapus.