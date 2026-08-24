# 07 — Pola Bab III Proposal Kampus

## Tujuan

File ini membekukan pola penyajian Bab III yang mengikuti gaya proposal kampus yang telah diberikan pengguna. Yang ditiru adalah **struktur, urutan penjelasan, dan tingkat kedetailan penyajian**, bukan isi metode dari proposal contoh.

Bab III harus terbaca sebagai alur penelitian yang dapat direalisasikan, bukan sebagai paper metode yang langsung melompat ke persamaan operator.

## Pola umum yang diadopsi

Urutan narasi metodologi:

```text
arsitektur/alur umum penelitian
    -> dataset
    -> persiapan dan pemisahan data
    -> treatment utama
    -> model detector
    -> skenario eksperimen
    -> konfigurasi pelatihan
    -> evaluasi
    -> analisis kesalahan/mekanisme
    -> lingkungan eksperimen
```

Pola ini meniru konvensi proposal kampus yang menempatkan gambar arsitektur umum di awal Bab III, kemudian menjelaskan tiap blok secara berurutan dan menggunakan tabel ringkas untuk dataset, model, skenario, konfigurasi, metrik, dan perangkat.

## Struktur Bab III untuk tesis AF2

### 3.1 Arsitektur Umum Penelitian

Jelaskan dua alur yang dibandingkan:

```text
Baseline:
RGB -> YOLO26n -> prediksi

Treatment:
RGB -> AF2 -> YOLO26n -> prediksi
```

Tekankan bahwa YOLO26 sama pada kedua arm dan perbedaan yang diisolasi berada pada input frontend.

Sertakan diagram alur penelitian pada dokumen final.

### 3.2 Dataset Penelitian

Isi minimum:

- nama dataset / versi immutable;
- jumlah kelas;
- jumlah citra dan anotasi per split;
- format bounding box;
- relasi taxonomy dengan konteks SNI;
- batas bahwa label eksperimen mengikuti dataset, bukan otomatis seluruh prosedur grading SNI.

Gunakan tabel ringkasan dataset.

### 3.3 Persiapan dan Pembagian Dataset

Jelaskan:

- grouped split;
- audit duplicate/hash/parent leakage;
- train/validation separation;
- locked-test policy;
- data augmentation yang berasal dari training pipeline detector dan bukan AF2.

Jangan mencampur split engineering dengan AF2.

### 3.4 Preprocessing Frekuensi-Angular (AF2)

Subbagian teknis yang disarankan:

1. pembentukan overlapping patches;
2. FFT dan dekomposisi amplitude/phase;
3. angular-density distribution;
4. entropy-conditioned threshold;
5. directional amplitude weighting;
6. inverse FFT;
7. overlap averaging dan residual image reconstruction.

Wajib bedakan:

- prinsip dari parent AFAB-2;
- keputusan transfer/implementasi repository;
- parameter yang aktif pada `mode=af2`;
- parameter konfigurasi bersama yang tidak aktif pada AF2.

### 3.5 YOLO26 sebagai Detector

Jelaskan varian yang benar-benar digunakan dan integration boundary.

AF2 **bukan** backbone, neck, atau detection head.

### 3.6 Skenario Eksperimen

Minimum paired comparison:

| Arm | Initialization | Input | Detector |
|---|---|---|---|
| D0DIRECT | official `yolo26n.pt` | native RGB | YOLO26n |
| AF2DIRECT | exact same source/state | AF2(RGB) | same YOLO26n |

RNG/head initialization dan training schedule harus matched.

Proposal boleh menyatakan rencana multi-seed setelah pilot seed-42, tetapi pilot tidak boleh ditulis sebagai final confirmation.

### 3.7 Konfigurasi Pelatihan

Gunakan direct-from-pretrained protocol sebagai authority, bukan `configs/D0_yolo26n.yaml` lama jika terjadi konflik.

Konfigurasi proposal yang saat ini dibekukan:

```text
seeds         = 42, 123, 2026   # planned confirmation
max epochs    = 50
imgsz         = 640
batch         = 16
workers       = 2
patience      = 15
optimizer     = auto
pretrained    = true
cache         = false
close_mosaic  = 10
max_det       = 500
deterministic = true
```

Seed 123 dan 2026 adalah **rencana konfirmasi**, bukan hasil yang sudah tersedia.

### 3.8 Evaluasi Performa

Pisahkan empat lapis evaluasi:

1. aggregate detection;
2. class-wise/lower-tail detection;
3. mechanism diagnostics classification-vs-localization;
4. efficiency.

Metric study-specific harus diberi definisi matematis, bukan diperlakukan sebagai metric standar COCO.

### 3.9 Analisis Kesalahan dan Mekanisme

Analisis minimal:

- per-class AP;
- difficult / lower-tail classes;
- confusion/error pattern bila tersedia;
- raw proposal accessibility;
- localization-conditioned Top-1;
- correct-decision recall.

Gunakan kata **konsisten dengan** ketika menafsirkan mekanisme; diagnostic bukan causal proof.

### 3.10 Perangkat dan Lingkungan Eksperimen

Catat pada setiap run:

- GPU/CPU/RAM;
- OS/runtime;
- Python/PyTorch/CUDA;
- Ultralytics version;
- pretrained artifact hash;
- repository commit;
- seed.

Perbandingan latency/VRAM harus dilakukan pada hardware/runtime yang sama.

## Tabel yang disarankan

Untuk mengikuti gaya kampus, Bab III final minimal memiliki:

- Tabel 3.1 Ringkasan Dataset;
- Tabel 3.2 Parameter AF2;
- Tabel 3.3 Skenario Eksperimen;
- Tabel 3.4 Konfigurasi Pelatihan;
- Tabel 3.5 Metrik Evaluasi;
- Tabel 3.6 Diagnostic Mechanism;
- Tabel 3.7 Perangkat dan Lingkungan Eksperimen.

## Aturan evidence

Authority order untuk Bab III:

```text
1. frozen direct-from-pretrained protocol
2. actual AF2 implementation code/config
3. immutable dataset audit
4. evaluation specifications
5. pilot result record
```

Jika config lama bertentangan dengan direct protocol, **direct protocol menang untuk proposal direct-AF2**.

Pilot evidence tidak boleh mengubah metode secara retroaktif.

## Aturan temporal proposal

Bab III harus membedakan:

- **sudah dilakukan**: seed-42 pilot / preflight evidence;
- **akan dilakukan**: multi-seed confirmation dan final efficiency/test evaluation;
- **dibekukan sebelum eksperimen**: matched initialization, schedule, metric definitions, decision rules.

Hindari menulis rencana masa depan seolah-olah sudah menghasilkan hasil final.