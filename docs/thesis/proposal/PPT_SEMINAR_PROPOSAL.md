# PPT Seminar Proposal — Source Teks

> Sumber isi ilmiah: `docs/thesis/proposal/` pada branch `proposal/thesis-foundation`.
> Dokumen ini adalah sumber teks ringkas untuk presentasi. Naskah formal BAB I–III tetap menjadi sumber utama apabila terdapat perbedaan.
> Rumusan masalah dan tujuan penelitian ditulis naratif, bukan per poin.

---

## SLIDE 01 — SAMPUL

### ANALISIS DAN OPTIMASI PRAPEMROSESAN CITRA BERBASIS FREKUENSI-ANGULAR PADA YOLO26 UNTUK DETEKSI FINE-GRAINED CACAT BIJI KOPI

**[Nama Mahasiswa]**  
**[NIM]**

**Dosen Pembimbing**  
[Dosen Pembimbing 1]  
[Dosen Pembimbing 2]

**Seminar Proposal**

---

## SLIDE 02 — PEMBAHASAN MATERI

### Pembahasan Materi

**01** Pendahuluan  
**02** Penelitian Terdahulu  
**03** Metodologi Penelitian

---

## SLIDE 03 — PENDAHULUAN

### Pendahuluan

**01**  
Inspeksi mutu biji kopi hijau masih banyak bergantung pada pengamatan visual, sehingga konsistensinya dapat dipengaruhi pengalaman dan kondisi pemeriksa.

**02**  
Deteksi otomatis menjadi lebih menantang ketika kategori cacat semakin rinci karena beberapa kelas memiliki kemiripan pada warna, tekstur, bentuk, dan detail lokal.

**03**  
Kondisi tersebut mendorong kebutuhan representasi citra yang lebih diskriminatif untuk deteksi *fine-grained* cacat biji kopi.

---

## SLIDE 04 — RUMUSAN MASALAH

### Rumusan Masalah

Deteksi *fine-grained* cacat biji kopi menghadapi kemiripan visual antarkelas, sedangkan pemanfaatan informasi frekuensi-angular sebelum proses deteksi masih terbatas. Penelitian ini mengkaji penerapan dan optimasinya pada YOLO26n terhadap kinerja deteksi dan biaya komputasi.

---

## SLIDE 05 — BATASAN MASALAH

### Batasan Masalah

- Deteksi *fine-grained* cacat biji kopi hijau.
- Dataset utama **robusta_SNI_Dataset (21 kelas)**.
- Dataset Capstone, Lulus, dan Niacubilla sebagai konfirmasi.
- Model utama **YOLO26n** tanpa modifikasi *backbone*, *neck*, dan *head*.
- Optimasi difokuskan pada **prapemrosesan frekuensi-angular**.
- Evaluasi utama menggunakan **mAP50–95** dan biaya komputasi *end-to-end*.

---

## SLIDE 06 — PENELITIAN TERDAHULU

### Penelitian Terdahulu

| Penelitian | Pendekatan | Relevansi |
|---|---|---|
| Hong et al. (2026) | Pengembangan YOLOv10 untuk 7 kategori cacat kopi | Menunjukkan modifikasi internal model untuk deteksi cacat kopi |
| Jiao et al. (2025) | Swin Transformer, *multistage feature fusion*, dan *selective attention* | Meningkatkan diskriminasi melalui representasi internal |
| Li et al. (2025) | Pemrosesan amplitudo dan fase Fourier sebelum YOLO | Menunjukkan pemrosesan spektral pada ruang masukan |
| Xu et al. (2025) | AFAB pada *fine-grained object detection* | Menjadi rujukan utama mekanisme frekuensi-angular |

**Posisi penelitian:** menguji dan mengoptimasi prapemrosesan frekuensi-angular pada ruang masukan tanpa mengubah arsitektur utama YOLO26n.

---

## SLIDE 07 — LANDASAN FREKUENSI-ANGULAR

### Mengapa Frekuensi-Angular?

Transformasi Fourier merepresentasikan citra berdasarkan komponen frekuensi. Selain besar frekuensi, distribusi spektrum dapat dianalisis berdasarkan arah atau orientasi.

Pada cacat *fine-grained*, perbedaan visual dapat muncul sebagai tekstur, detail lokal, dan pola permukaan yang halus. Penelitian ini menguji apakah informasi frekuensi-angular tersebut dapat membentuk representasi masukan yang lebih membantu detektor.

**Frekuensi** → skala perubahan visual  
**Angular** → arah/orientasi pola spektral

---

## SLIDE 08 — DATASET PENELITIAN

### Dataset Penelitian

**Dataset Utama**  
**robusta_SNI_Dataset** — 21 kelas — pengembangan dan pemilihan konfigurasi.

**Dataset Konfirmasi**  
Capstone — 14 kelas  
Lulus — 6 kelas  
Niacubilla — 9 kelas

Setiap dataset digunakan secara terpisah dengan taksonomi masing-masing. Pembagian data dirancang **70% train, 15% validation, 15% test**, dengan pengelompokan berdasarkan sumber citra ketika informasi tersebut tersedia.

---

## SLIDE 09 — KARAKTERISTIK METODE

### Karakteristik Prapemrosesan

**Input**  
Citra RGB hasil augmentasi

**Proses utama**  
Patch lokal → FFT 2D → amplitudo dan fase → distribusi angular/radial-angular → pembobotan spektral → inverse FFT → rekonstruksi

**Output**  
Citra hasil residual dengan ukuran spasial tetap, kemudian diberikan ke YOLO26n.

**Prinsip utama**  
Arsitektur YOLO26n tidak dimodifikasi dan prapemrosesan tidak menambahkan parameter *trainable*.

---

## SLIDE 10 — METODOLOGI PENELITIAN

### Metodologi Penelitian

Penelitian menggunakan rancangan eksperimen komparatif dengan empat kondisi utama:

**B0** — YOLO26n tanpa prapemrosesan  
**B1** — CLAHE + YOLO26n  
**B2** — C0 + YOLO26n  
**B3** — C* + YOLO26n

**B2 − B0** mengukur efek *frontend* frekuensi-angular referensi.  
**B3 − B2** mengukur efek optimasi desain.  
**B3 − B1** membandingkan metode terpilih dengan peningkatan kontras lokal konvensional.

---

## SLIDE 11 — ALUR PRAPEMROSESAN

### Alur Prapemrosesan Frekuensi-Angular

```text
Citra RGB
   ↓
Pembentukan Patch Lokal
   ↓
FFT 2D
   ↓
Amplitudo + Fase
   ↓
Distribusi Angular / Radial-Angular
   ↓
Adaptive Threshold + Spectral Weighting
   ↓
Inverse FFT + Rekonstruksi
   ↓
Residual Fusion
   ↓
YOLO26n
```

Persamaan residual utama:

**I′ = I + I ⊙ G**

---

## SLIDE 12 — VARIASI DESAIN

### Analisis Variasi Desain

**C0** — konfigurasi frekuensi-angular referensi  
↓  
**C1** — fungsi jendela Hann  
↓  
**C2** — orientasi tak bertanda  
↓  
**C3** — tiga pita radial  
↓  
**C4** — ambang lunak  
↓  
**C5** — panduan luminansi  
↓  
**C\*** — konfigurasi terpilih

Setiap tahap menambahkan satu perubahan utama secara kumulatif untuk menganalisis pengaruh desain prapemrosesan.

---

## SLIDE 13 — ALUR PENELITIAN

### Alur Penelitian

```text
robusta_SNI_Dataset
        ↓
Grouped Split 70/15/15
        ↓
Baseline Pengembangan B0 (seed 42)
        ↓
Tetapkan 3 Kelas Sulit
        ↓
Uji C0 → C5
        ↓
Pilih Struktur Kandidat
        ↓
Analisis Sensitivitas Terbatas
        ↓
Pilih dan Bekukan C*
        ↓
Konfirmasi Multi-seed
        ↓
Evaluasi Akhir pada Test Set
```

---

## SLIDE 14 — KONFIRMASI LINTAS DATASET

### Konfirmasi Metode

Setelah **C\*** dipilih pada dataset utama, konfigurasi tersebut **dibekukan** dan tidak dituning ulang pada dataset konfirmasi.

```text
                 C*
                  ↓
        ┌─────────┼─────────┐
        ↓         ↓         ↓
    Capstone    Lulus   Niacubilla
        ↓         ↓         ↓
      B0 vs B3  B0 vs B3  B0 vs B3
```

Seed konfirmasi: **123, 2026, 31415**.

Konsistensi dinilai dari arah dan besarnya perubahan kinerja pada masing-masing dataset, bukan dari perbandingan mAP absolut antardataset.

---

## SLIDE 15 — TUJUAN PENELITIAN

### Tujuan Penelitian

Penelitian ini bertujuan untuk menerapkan dan menganalisis prapemrosesan citra berbasis frekuensi-angular pada YOLO26n untuk deteksi *fine-grained* cacat biji kopi, menentukan konfigurasi dari variasi desain yang diuji, serta mengevaluasi pengaruhnya terhadap kinerja deteksi dan biaya komputasi.

---

## SLIDE 16 — EVALUASI

### Evaluasi Penelitian

**Kinerja Deteksi**  
Metrik utama: **mAP50–95**  
Tambahan: mAP50, precision, recall, AP per kelas

**Kelas Sulit**  
Rerata AP tiga kelas tersulit dan AP kelas terendah digunakan untuk melihat apakah peningkatan juga terjadi pada kelas yang sulit dikenali.

**Efisiensi**  
Waktu prapemrosesan, waktu inferensi, *end-to-end latency*, throughput/FPS, dan *peak allocated GPU memory*.

---

## SLIDE 17 — PENUTUP

### TERIMA KASIH

**Analisis dan Optimasi Prapemrosesan Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

---

# Catatan Penyaji / Backup

## A. Pemilihan C*

Konfigurasi C0–C5 dibandingkan pada validation set dataset utama. Kandidat dengan selisih kurang dari 0,001 terhadap mAP50–95 tertinggi masuk kelompok *tie*. Jika lebih dari satu kandidat, digunakan AP kelompok tiga kelas sulit, kemudian median *end-to-end latency* sebagai *tie-break* berikutnya.

## B. Sensitivitas Terbatas

- ukuran patch: `m ∈ {16, 32, 64}`;
- gamma: `γ ∈ {0.05, 0.10, 0.15}`;
- jika menggunakan soft threshold: `T ∈ {0.01, 0.02, 0.05}`;
- overlap tetap 50%.

## C. Radial-Angular

Pada C3, spektrum dibagi menjadi tiga pita radial. Distribusi angular, normalisasi, entropi, dan ambang dihitung terpisah pada setiap pita sehingga seleksi orientasi dilakukan relatif terhadap wilayah frekuensi yang berbeda.

## D. Kontrak Eksperimen

Seluruh run YOLO26n dimulai dari bobot resmi `yolo26n.pt`. Tidak ada pewarisan checkpoint antarkondisi atau antardataset. C* hanya dipilih pada dataset utama dan dibekukan sebelum konfirmasi.