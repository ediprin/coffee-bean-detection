# Claim-Level Source Audit — Proposal

Status: **PRIMARY-FULL-TEXT HARD GATE**

Dokumen ini mengaudit **klaim**, bukan hanya metadata bibliografis. Sebuah paper boleh memiliki metadata resmi yang benar tetapi tetap tidak boleh dipakai untuk klaim tertentu apabila full text primer tidak mendukung klaim tersebut.

## Aturan

1. Klaim metodologis dan hasil penelitian terdahulu harus ditopang oleh full text primer.
2. Metadata publisher, DOI, indeks, dan quartile tidak membuktikan mekanisme metode.
3. Jika paper memiliki inkonsistensi internal, inkonsistensi tersebut dicatat dan tidak diselaraskan melalui asumsi.
4. Klaim dari domain lain tidak dipindahkan menjadi fakta domain kopi. Transfer metode selalu ditulis sebagai hipotesis/rancangan yang akan diuji.
5. Hasil full model tidak boleh dipindahkan menjadi hasil satu submodul tanpa ablation yang mendukung.
6. Analisis visual/CAM bersifat interpretatif; visualisasi tidak dianggap bukti kausal tunggal.

---

## Klaim inti yang telah diverifikasi dari full text primer

| Key | Klaim yang aman dipakai dalam proposal | Locator primer | Batas klaim / jangan overclaim |
|---|---|---|---|
| COF-01 — Hong et al. (2026) | Studi mendeteksi tujuh kategori cacat kopi dengan improved YOLOv10. Paper juga menggunakan **EigenCAM** untuk membandingkan activation map baseline dan model usulan, lalu confusion matrix untuk menelaah inter-class confusion. Immature bean dibahas memiliki misclassification ke beberapa kelas lain dan dikaitkan penulis dengan visual similarity/intra-class heterogeneity. | Primary PDF, p. 12, Fig. 5 (EigenCAM), Fig. 6 (confusion matrix), bagian interpretability; Conclusion. | Jangan menyatakan EigenCAM membuktikan kausalitas. Jangan memindahkan gain DSConv/SPPF-Attention/PConv ke preprocessing frekuensi-angular. Hasil hanya berlaku pada dataset tujuh kelas dan arsitektur Hong. |
| COF-02 — Bahy & Rifai (2026) | Paper menyatakan 20 kategori fisik SNI, dataset 107 citra dengan 13.863 anotasi. Per-class performance heterogen; Table VII menunjukkan Pod Bean mAP50 0.986, Immature Bean 0.976, sedangkan Slight Insect Damage 0.626. Penulis menyebut morphological variability dapat menghasilkan gap performa yang besar. | Primary PDF, Abstract p. 29; Table VII dan pembahasan p. 37. | Aman untuk mendukung **per-class heterogeneity** pada taksonomi rinci. Jangan menyimpulkan bahwa penyebabnya adalah bottleneck frekuensi atau bahwa AFAB-2 akan memperbaikinya. |
| COF-03 — Samudra & Rachmawati (2025) | Studi menggunakan tiga kelas: black, partially black, small coffee husk. Confusion analysis menunjukkan model kesulitan membedakan black dan partially black dan penulis mengaitkannya dengan visual similarity. | Primary IEEE PDF, p. 696, Section “Detection per Defect Class”, Fig. 6–9; abstrak juga menyatakan visual similarity. | Hanya tiga kelas. Jangan digeneralisasi sebagai bukti bahwa seluruh taksonomi cacat kopi memiliki pola yang sama. Jangan gunakan sebagai bukti efektivitas frequency processing. |
| COF-04 — Hebert & Alamsyah (2026) | Table 5 menunjukkan AP Cherry Pods 0.89, Floater 0, Fungus Damage 0.18, Slight Insect Damage 0.15. Penulis menjelaskan slight insect damage berupa titik hitam/bekas gigitan halus yang dapat tersembunyi oleh tekstur alami, fungus memiliki bercak tipis tidak beraturan dengan warna mirip permukaan biji, dan floater memiliki warna/bentuk mirip biji normal. | Primary PDF, p. 94, Table 5 dan pembahasan; Conclusion. | Sangat kuat untuk mendukung masalah **subtle visual variation/small defect**. Tidak membuktikan frequency-domain bottleneck dan tidak membuktikan AFAB-2 cocok untuk kopi. |
| COF-05 — Jundullah et al. (2026) | Paper **melaporkan** evaluasi multi-class dengan 20 kelas dan mean P=0.76, R=0.75, mAP50=0.75. Pembahasan menyatakan kelas dengan karakteristik visual khas lebih mudah, sedangkan black variants/brown/sour yang mirip secara visual lebih sulit; penulis mengaitkan confusion dengan subtle color degradation, structural similarity, small object size, dan top-down imaging. | Primary PDF, p. 319, Table 3 dan pembahasan; bagian Discussion/Conclusion setelah Table 3. | **INKONSISTENSI INTERNAL WAJIB DICATAT:** Conclusion menyebut “across 20 classes”, tetapi Table 3 yang tercetak tampak memuat 23 baris label kelas. Proposal boleh menulis “paper melaporkan evaluasi pada 20 kelas”, tetapi jangan menyatakan bahwa Table 3 secara independen membuktikan tepat 20 label tanpa catatan. Jangan menggeneralisasi kalimat “fine-grained classification rather than localization” sebagai fakta universal semua dataset kopi. |
| COF-07 — Kesiman et al. (2023) | Dataset menyediakan benchmark 3 kelas dan ground-truth 17 jenis cacat untuk defective subset. MobileNet mencapai 92.52% pada 3 kelas dan 39.82% pada 17 kelas; InceptionResNetV2 91.29% dan 53.35%. Penulis secara eksplisit menyatakan klasifikasi 17 jenis cacat masih sulit. | Primary IEEE PDF, Abstract p. 75; Table IV / hasil dan Conclusion sekitar p. 79. | Ini **classification evidence**, bukan object-detection result. Aman untuk diagnosis bahwa granularitas kategori memperbesar difficulty, tetapi tidak boleh mengklaim detector akan mengalami penurunan numerik yang sama. |
| FG-01 — Xu et al. (2025) | LFDet menggunakan AFAB pada data space. AFAB melakukan Fourier transform pada patch lokal. **AFAB-2** menghitung distribusi density angular dari amplitude, menormalisasi distribusi, menghitung information entropy untuk adaptive threshold, menekan arah dengan density rendah, kemudian merekonstruksi dengan amplitude yang disesuaikan dan phase asli. | Primary PDF, pp. 5–6, §3.3.3, Eq. (9)–(13). | AFAB-1 radial/high-pass dan AFAB-2 angular adalah submekanisme berbeda. Metode tesis mengadaptasi prinsip AFAB-2 sebagai standalone input preprocessing; jangan menyebutnya full AFAB/LFDet. Jangan memindahkan full LFDet gain menjadi gain AFAB-2. Domain sumber adalah fine-grained aircraft remote sensing, bukan kopi. |
| PRE-04 — Syauqi et al. (2025) | Preprocessing white-pepper bukan CLAHE tunggal. Pipeline dilakukan berurutan: gamma correction → CLAHE → bilinear blending/interpolation → denoising (NLM) → unsharp masking/sharpening, sebelum YOLOv8m. | Primary IEEE PDF, p. 19, Section B “Image Processing”. | Gunakan istilah **CLAHE-based/composite preprocessing pipeline**, bukan “CLAHE saja”. Hasil pada white pepper tidak membuktikan manfaat metode frekuensi-angular pada kopi. |

---

## Discrepancy register

### D-01 — Jundullah et al. (2026): jumlah kelas

Full text paper secara eksplisit menyatakan pada Conclusion bahwa model mencapai mAP@0.5 pada **20 classes**. Namun, Table 3 yang tercetak tampak mencantumkan **23 baris label kelas**. Karena dokumen primer sendiri tidak sepenuhnya konsisten, audit ini **tidak mengoreksi paper melalui asumsi**.

Aturan penulisan formal:

> Aman: “Jundullah et al. (2026) melaporkan evaluasi sistem pada 20 kelas cacat dan kontaminan.”

> Jangan: “Table 3 membuktikan dataset terdiri tepat dari 20 kelas.”

Jika jumlah kelas menjadi argumen sentral, taxonomy harus direkonsiliasi dari bagian dataset/label source paper sebelum angka tersebut diperlakukan sebagai fakta independen.

### D-02 — Kesiman vs kutipan sekunder Bahy

Primary paper Kesiman et al. (2023) secara eksplisit menyatakan **17 classes of coffee bean with defects** dan melaporkan hasil 17-class benchmark. Bahy & Rifai (2026) pada abstract mereka menyebut prior study dengan “18 simplified classes”. Untuk proposal, fakta tentang studi Kesiman harus mengikuti **paper Kesiman sendiri sebagai sumber primer**, bukan deskripsi sekunder Bahy.

---

## Claim wording yang direkomendasikan untuk proposal

### Problem statement coffee-domain

Aman:

> Sejumlah penelitian menunjukkan adanya perbedaan kinerja antarkelas pada deteksi cacat biji kopi, terutama ketika model harus membedakan kategori dengan karakteristik visual yang berdekatan atau tanda cacat yang halus.

Dukungan primer: COF-02, COF-03, COF-04, COF-05, COF-07.

Tidak aman:

> Domain frekuensi adalah bottleneck utama deteksi cacat biji kopi.

Tidak ada paper kopi yang diaudit di atas yang membuktikan klaim tersebut.

### Transfer frequency-angular

Aman:

> Xu et al. (2025) menunjukkan bahwa pemrosesan frekuensi lokal dan distribusi angular dapat digunakan pada fine-grained aircraft detection. Efektivitas mekanisme serupa pada cacat biji kopi masih perlu diuji secara empiris.

Tidak aman:

> Karena cacat kopi sulit dibedakan, AFAB-2 pasti akan meningkatkan klasifikasi cacat kopi.

### Visual analysis

Aman:

> Hong et al. (2026) menggunakan EigenCAM sebagai analisis kualitatif pendamping hasil kuantitatif. Penelitian ini mengadaptasi pola evaluasi visual tersebut untuk membandingkan respons model, dengan interpretasi yang tetap dibatasi sebagai bukti pendukung.

Tidak aman:

> EigenCAM membuktikan model menggunakan fitur tekstur tertentu sebagai penyebab keputusan.

---

## Status berikutnya

Audit berikutnya harus mencakup:

1. PRE-05 Chen et al. (2024) — exact preprocessing sequence dari primary Elsevier paper;
2. PRE-02 Qin et al. (2022) — exact DENet/DE-YOLO mechanism dari primary ACCV/Springer/CVF full text;
3. PRE-03 Li et al. (2025) — exact Fourier amplitude/phase mechanism dari primary DSP full text;
4. SPEC-01 Cao et al. (2019) dan SPEC-02 Zhang & Tan (2003) — exact radial/angular texture claims;
5. THEORY-01 / THEORY-02 — authoritative textbook support untuk definisi DFT/FFT jika persamaan fundamental tetap digunakan pada BAB II;
6. EVAL-01/EVAL-02 — exact COCO metric wording jika threshold schedule dan implementasi evaluasi dijelaskan secara rinci.

Proposal belum disebut **claim-audited complete** sampai sumber-sumber yang masih dipakai pada BAB I–III telah melewati tahap ini.
