# PPT Seminar Proposal — Source Teks

> Sumber isi ilmiah: `docs/thesis/proposal/` pada branch `proposal/thesis-foundation`.
> Dokumen ini adalah sumber teks ringkas untuk presentasi. Naskah formal BAB I–III tetap menjadi sumber utama apabila terdapat perbedaan.
> Rumusan masalah dan tujuan penelitian ditulis naratif, bukan per poin.
> Deck visual menggunakan font **Poppins** untuk seluruh teks.

---

## SLIDE 01 — SAMPUL

### ANALISIS DAN OPTIMASI PRAPEMROSESAN CITRA BERBASIS FREKUENSI-ANGULAR PADA YOLO26 UNTUK DETEKSI FINE-GRAINED CACAT BIJI KOPI

**[Nama Mahasiswa]**  
**[NIM]**

**Dosen Pembimbing**  
[Dosen Pembimbing 1]  
[Dosen Pembimbing 2]

**Seminar Proposal Tesis**

---

## SLIDE 02 — LATAR BELAKANG

### Latar Belakang

Pemeriksaan fisik biji kopi masih dapat bergantung pada pengamatan manusia sehingga konsistensinya dipengaruhi pengalaman, kondisi pengamatan, pelatihan, dan beban kerja pemeriksa.

*Object detection* memungkinkan model mengenali kategori cacat sekaligus lokasi objek pada citra. Namun, pada susunan kategori yang lebih rinci, perbedaan kinerja antarkelas menjadi lebih terlihat karena beberapa cacat memiliki kemiripan warna, tekstur, bentuk, atau tanda lokal kecil.

**Inti masalah:** detektor memerlukan representasi citra yang cukup diskriminatif untuk membedakan cacat yang berdekatan secara visual.

---

## SLIDE 03 — RUMUSAN MASALAH

### Rumusan Masalah

Deteksi cacat biji kopi dengan jumlah kategori yang rinci memiliki tantangan karena beberapa jenis cacat mempunyai karakteristik visual yang relatif serupa sehingga kemampuan model dalam mengenali setiap kelas dapat berbeda. Dalam literatur biji kopi yang ditinjau, peningkatan kinerja umumnya dilakukan melalui modifikasi komponen di dalam model, sedangkan pengolahan citra berdasarkan informasi frekuensi dan arah sebelum proses deteksi masih perlu dikaji lebih lanjut pada kasus cacat biji kopi.

**Fokus penelitian:** penerapan prapemrosesan citra berbasis frekuensi-angular pada YOLO26n, variasi desainnya, dan pengaruhnya terhadap deteksi *fine-grained* cacat biji kopi.

---

## SLIDE 04 — TUJUAN, BATASAN, DAN MANFAAT PENELITIAN

### Tujuan, Batasan, dan Manfaat Penelitian

**Tujuan Penelitian**  
Menerapkan dan menganalisis prapemrosesan citra berbasis frekuensi-angular pada YOLO26n, menentukan konfigurasi dari variasi desain yang diuji, serta mengevaluasi pengaruhnya terhadap kinerja deteksi dan biaya komputasi.

**Batasan Masalah**
- *Object detection* biji kopi hijau.
- YOLO26n sebagai model utama.
- Prapemrosesan frekuensi-angular tanpa modifikasi *backbone*, *neck*, atau *detection head*.
- Dataset utama dan konfirmasi digunakan secara terpisah.

**Manfaat Penelitian**  
Kajian empiris mengenai prapemrosesan frekuensi-angular untuk deteksi *fine-grained* cacat biji kopi, termasuk pengaruh pada kinerja keseluruhan, kelas sulit, dan biaya komputasi.

---

## SLIDE 05 — PENELITIAN TERKAIT: BIJI KOPI

### Penelitian Terkait — Biji Kopi

| Penelitian | Metode/Model | Relevansi |
|---|---|---|
| Hong et al. (2026) | Improved YOLOv10 | Deteksi 7 kategori cacat kopi; analisis kebingungan antarkelas. |
| Bahy & Rifai (2026) | Lightweight YOLOv5s | Deteksi 20 kategori fisik berbasis SNI; kinerja antarkelas heterogen. |
| Tarekegn & Debelee (2025) | KN-YOLOv8 | Deteksi 13 kelas cacat dan satu kelas normal; dataset multiobjek berskala anotasi besar. |
| Samudra & Rachmawati (2025) | LSKNet + Oriented R-CNN | Deteksi 3 kelas Arabika berbasis SNI; kebingungan *black* dan *partially black*. |

**Pesan utama:** literatur kopi menunjukkan kelayakan deteksi berbasis YOLO, tetapi performa dapat berbeda antarkelas ketika kategori semakin rinci.

---

## SLIDE 06 — PENELITIAN TERKAIT: BIJI KOPI DAN RESEARCH GAP

### Penelitian Terkait — Biji Kopi dan Research Gap

| Penelitian | Metode/Model | Relevansi |
|---|---|---|
| Jundullah et al. (2026) | YOLOv8s | Deteksi multikelas cacat dan kontaminan; ketimpangan kinerja antarkelas. |
| Hebert & Alamsyah (2026) | YOLOv12 | Deteksi 15 kategori cacat; beberapa kelas memiliki AP lebih rendah. |
| Kesiman et al. (2023) | MobileNet / InceptionResNetV2 | Klasifikasi berbasis SNI; peningkatan dari 3 ke 17 kelas menaikkan kesulitan diskriminasi. |
| Gope et al. (2024) | Perbandingan varian YOLO | Menunjukkan kelayakan keluarga YOLO pada deteksi biji kopi hijau. |

**Research gap:** penelitian deteksi cacat kopi yang ditinjau terutama meningkatkan kinerja melalui pemilihan atau modifikasi model. Mekanisme frekuensi-angular belum dievaluasi sebagai prapemrosesan citra masukan untuk deteksi *fine-grained* cacat biji kopi dalam literatur yang ditinjau.

---

## SLIDE 07 — LANDASAN PRAPEMROSESAN DAN REPRESENTASI FREKUENSI

### Landasan Prapemrosesan dan Representasi Frekuensi

**Fine-Grained Object Detection**  
Kategori berdekatan secara visual membutuhkan representasi yang lebih diskriminatif.

**Input-Space Preprocessing**  
Transformasi citra sebelum detektor dapat dinilai dari efeknya pada tugas deteksi.

**Frequency–Angular Representation**  
Amplitudo Fourier dapat dianalisis berdasarkan radius dan arah relatif terhadap pusat spektrum.

**Catatan:** frekuensi-angular merujuk pada representasi Fourier lokal dan analisis amplitudo berdasarkan arah, bukan *oriented bounding box*.

---

## SLIDE 08 — RANCANGAN UMUM PENELITIAN

### Rancangan Umum Penelitian

Penelitian menggunakan eksperimen komparatif untuk menganalisis pengaruh prapemrosesan citra berbasis frekuensi-angular terhadap kinerja YOLO26n pada deteksi *fine-grained* cacat biji kopi. Arsitektur YOLO26n dipertahankan pada perbandingan utama, sedangkan perlakuan eksperimen diberikan pada citra masukan.

Alur umum:

```text
Persiapan robusta_SNI_Dataset
→ Model acuan B0 dev
→ Uji variasi C0–C5
→ Pilih dan bekukan C*
→ Konfirmasi lintas dataset
→ Evaluasi akhir
```

**Kontrak utama:** *backbone*, *neck*, dan *detection head* YOLO26n tidak dimodifikasi.  
**Anti-leakage:** C* dipilih hanya pada dataset utama, lalu dibekukan sebelum konfirmasi.

---

## SLIDE 09 — DATASET PENELITIAN

### Dataset Penelitian

**Dataset Utama**  
**robusta_SNI_Dataset** — 21 kelas; tersedia sebagai *instance segmentation* dan anotasi digunakan sebagai *bounding box*. Dataset ini menjadi satu-satunya dataset untuk memilih C*.

**Dataset Konfirmasi**
- Coffee Bean Defect (Capstone) — 14 kelas.
- Green Coffee Bean Defects (Lulus) — 6 kelas.
- Coffee Bean Defects (Niacubilla) — 9 kelas.

**Split:** 70% train, 15% validation, 15% test. Setiap dataset digunakan secara terpisah.

---

## SLIDE 10 — PRAPEMROSESAN CITRA BERBASIS FREKUENSI-ANGULAR

### Prapemrosesan Citra Berbasis Frekuensi-Angular

Prapemrosesan mengadaptasi mekanisme angular AFAB-2 sebagai *frontend* pada ruang masukan. Tahapan utama:

```text
Citra RGB → Patch lokal → FFT 2D → Amplitudo dan fase
→ Distribusi angular → Ambang adaptif → Pembobotan spektral
→ Inverse FFT → Rekonstruksi → Residual fusion
```

Persamaan residual utama:

$$
I'^c = I^c + I^c \odot G^c
$$

Fase asli dipertahankan untuk rekonstruksi, sedangkan respons spektral membentuk gate G. Prapemrosesan tidak menambahkan parameter trainable.

---

## SLIDE 11 — ANALISIS VARIASI DESAIN PRAPEMROSESAN

### Analisis Variasi Desain Prapemrosesan

| Kode | Perubahan utama | Tujuan pengujian |
|---|---|---|
| C0 | Konfigurasi frekuensi-angular referensi | Menjadi acuan prapemrosesan. |
| C1 | Fungsi jendela Hann | Menguji pengaruh batas patch. |
| C2 | Orientasi tak bertanda | Menguji arah dan orientasi. |
| C3 | Tiga pita radial | Menguji seleksi angular pada wilayah radial berbeda. |
| C4 | Ambang lunak | Menguji pembobotan bertahap di sekitar ambang. |
| C5 | Panduan luminansi | Menguji kebutuhan pembobotan RGB terpisah. |

Seluruh konfigurasi menggunakan kontrak residual C0 yang sama dan ditambahkan secara kumulatif.

---

## SLIDE 12 — RANCANGAN EKSPERIMEN

### Rancangan Eksperimen

Empat kondisi utama:

| Kode | Kondisi | Peran |
|---|---|---|
| B0 | YOLO26n | Kondisi acuan. |
| B1 | CLAHE + YOLO26n | Kontrol peningkatan kontras lokal. |
| B2 | C0 + YOLO26n | Frontend frekuensi-angular referensi. |
| B3 | C* + YOLO26n | Konfigurasi frekuensi-angular terpilih. |

Perbandingan B2−B0 mengukur efek *frontend* referensi, B3−B2 mengukur efek optimasi desain, dan B3−B1 membandingkan konfigurasi terpilih terhadap CLAHE.

C* dipilih pada validation set robusta_SNI_Dataset, lalu dibekukan sebelum konfirmasi dan test akhir. Seluruh run dimulai dari bobot awal `yolo26n.pt`.

---

## SLIDE 13 — EVALUASI KINERJA DETEKSI DAN EFISIENSI KOMPUTASI

### Evaluasi Kinerja Deteksi dan Efisiensi Komputasi

**Kinerja Deteksi**
- mAP50–95 sebagai metrik utama.
- mAP50, precision, dan recall sebagai metrik tambahan.

**Fine-Grained / Per Kelas**
- AP per kelas.
- AP_H untuk tiga kelas sulit.
- AP_worst, confusion matrix, FP, dan FN.

**Efisiensi Komputasi**
- waktu prapemrosesan t_pra;
- waktu inferensi model t_model;
- latency end-to-end t_total;
- FPS dan peak GPU memory.

**Prinsip evaluasi:** hasil test tidak digunakan untuk memilih ulang metode; biaya FFT dan seluruh operasi frontend masuk dalam latency end-to-end.

---

## SLIDE 14 — PENUTUP

### TERIMA KASIH

**Pertanyaan & Diskusi**

---

# Catatan Penyaji / Backup

## A. Detail Pemilihan C*

Konfigurasi C0–C5 dibandingkan pada validation set dataset utama. Kandidat dengan selisih kurang dari 0,001 terhadap mAP50–95 tertinggi masuk kelompok *tie*. Jika lebih dari satu kandidat, digunakan AP kelompok tiga kelas sulit, kemudian median *end-to-end latency* sebagai *tie-break* berikutnya.

## B. Sensitivitas Terbatas

- ukuran patch: `m ∈ {16, 32, 64}`;
- gamma: `γ ∈ {0.05, 0.10, 0.15}`;
- jika menggunakan soft threshold: `T ∈ {0.01, 0.02, 0.05}`;
- overlap tetap 50%.

## C. Kontrak Eksperimen

Seluruh run YOLO26n dimulai dari bobot resmi `yolo26n.pt`. Tidak ada pewarisan checkpoint antarkondisi atau antardataset. C* hanya dipilih pada dataset utama dan dibekukan sebelum konfirmasi.

## D. Backup Metode

- FFT, amplitudo, dan fase.
- Entropy-based adaptive threshold.
- Hard threshold vs soft threshold.
- Tiga radial bands pada C3.
- Luminance guidance pada C5.
- Alasan pemilihan CLAHE sebagai baseline konvensional.
- Protokol pengukuran latency end-to-end.
