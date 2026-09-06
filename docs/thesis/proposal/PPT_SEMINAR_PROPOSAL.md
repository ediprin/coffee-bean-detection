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

| Penelitian | Pendekatan | Fokus |
|---|---|---|
| Hong et al. (2026) | Improved YOLOv10 | Deteksi cacat kopi |
| Jiao et al. (2025) | Multistage fusion + attention | Diskriminasi fitur cacat kopi |
| Li et al. (2025) | Fourier preprocessing + YOLO | Pemrosesan spektral sebelum deteksi |
| Xu et al. (2025) | AFAB | Frekuensi-angular untuk *fine-grained detection* |

**Gap:** penelitian pada kopi lebih banyak mengembangkan representasi internal model, sedangkan pemanfaatan prapemrosesan frekuensi-angular sebelum detektor belum dikaji secara khusus.

---

## SLIDE 07 — MENGAPA FREKUENSI-ANGULAR?

### Mengapa Frekuensi-Angular?

**Fine-grained defect**  
→ perbedaan kecil pada tekstur dan pola permukaan

**Frequency**  
→ menangkap karakteristik perubahan dan detail visual

**Angular**  
→ menangkap distribusi respons berdasarkan arah

**Hipotesis penelitian:** representasi tersebut dapat membantu menghasilkan masukan yang lebih diskriminatif bagi detektor.

---

## SLIDE 08 — DATASET PENELITIAN

### Dataset Penelitian

**Dataset Utama**  
**robusta_SNI_Dataset** — 21 kelas  
Pengembangan dan pemilihan **C\***

**Dataset Konfirmasi**  
Capstone — 14 kelas  
Lulus — 6 kelas  
Niacubilla — 9 kelas

**Split 70% train · 15% validation · 15% test**  
Setiap dataset digunakan secara terpisah.

---

## SLIDE 09 — METODOLOGI PENELITIAN

### Metodologi Penelitian

Empat kondisi eksperimen utama:

**B0** — YOLO26n  
**B1** — CLAHE → YOLO26n  
**B2** — C0 → YOLO26n  
**B3** — C* → YOLO26n

**B2 − B0** → efek *frequency-angular reference frontend*  
**B3 − B2** → efek optimasi desain  
**B3 − B1** → perbandingan terhadap CLAHE

---

## SLIDE 10 — ALUR PRAPEMROSESAN

### Alur Prapemrosesan Frekuensi-Angular

```text
Citra RGB
   ↓
Patch Lokal
   ↓
FFT 2D
   ↓
Analisis Amplitudo & Arah
   ↓
Adaptive Spectral Weighting
   ↓
Inverse FFT
   ↓
Rekonstruksi
   ↓
Residual Fusion
   ↓
YOLO26n
```

**I′ = I + I ⊙ G**

**Parameter-free frontend · YOLO26n tidak dimodifikasi**

---

## SLIDE 11 — OPTIMASI DESAIN

### Optimasi Desain

**C0** — Reference frequency-angular  
↓  
**C1** — + Hann window  
↓  
**C2** — + Unsigned orientation  
↓  
**C3** — + Radial bands  
↓  
**C4** — + Soft threshold  
↓  
**C5** — + Luminance guidance  
↓  
**C\*** — konfigurasi terpilih

Setiap konfigurasi menambahkan satu perubahan utama secara kumulatif.

---

## SLIDE 12 — ALUR PENELITIAN

### Alur Penelitian

```text
robusta_SNI_Dataset
        ↓
Split 70 / 15 / 15
        ↓
Baseline B0
        ↓
Tetapkan Hard Classes
        ↓
Evaluasi C0–C5
        ↓
Sensitivity Analysis
        ↓
Pilih & Bekukan C*
        ↓
Multi-seed Confirmation
        ↓
Final Test
```

**Test set tidak digunakan untuk memilih C\*.**

---

## SLIDE 13 — KONFIRMASI LINTAS DATASET

### Konfirmasi Lintas Dataset

```text
             C* dibekukan
                  ↓
       ┌──────────┼──────────┐
       ↓          ↓          ↓
   Capstone     Lulus    Niacubilla
   14 kelas     6 kelas     9 kelas
       ↓          ↓          ↓
    B0 vs B3   B0 vs B3   B0 vs B3
```

**Seeds: 123 · 2026 · 31415**

**Tidak ada retuning C\*** pada dataset konfirmasi.

---

## SLIDE 14 — TUJUAN PENELITIAN

### Tujuan Penelitian

Menganalisis dan mengoptimasi prapemrosesan citra berbasis frekuensi-angular pada YOLO26n untuk deteksi *fine-grained* cacat biji kopi serta mengevaluasi pengaruhnya terhadap kinerja deteksi dan biaya komputasi.

---

## SLIDE 15 — EVALUASI PENELITIAN

### Evaluasi Penelitian

**Deteksi**  
- mAP50–95
- mAP50
- Precision & Recall

**Fine-grained**  
- AP per kelas
- AP_H
- AP_worst

**Efisiensi**  
- Preprocessing time
- End-to-end latency
- FPS
- Peak GPU memory

---

## SLIDE 16 — PENUTUP

### TERIMA KASIH

**Pertanyaan & Diskusi**

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

## E. Detail Metode untuk Backup Slide

- FFT, amplitudo, dan fase.
- Entropy-based adaptive threshold.
- Hard threshold vs soft threshold.
- Tiga radial bands pada C3.
- Luminance guidance pada C5.
- Alasan pemilihan CLAHE sebagai baseline konvensional.
- Aturan pemilihan C* dan sensitivity analysis.
- Protokol pengukuran latency end-to-end.
- Alasan penggunaan YOLO26n sebagai detektor utama.