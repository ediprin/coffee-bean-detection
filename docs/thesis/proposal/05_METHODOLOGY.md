# BAB III — METODOLOGI PENELITIAN

Status: **proposal draft, source-grounded from the frozen direct-AF2 protocol and current repository implementation**.

Authority for this chapter:

- `docs/FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_PROTOCOL_2026-08-24.md`;
- `configs/afab/AF2_yolo26n_chaotic_amplitude.yaml`;
- `src/coffee_detector/afab/operator.py`;
- immutable Faruq-v3 dataset contract recorded in the direct protocol.

Important temporal boundary: seed-42 direct-AF2 is preliminary feasibility evidence. Seeds 123 and 2026 are planned confirmation runs and are **not** treated as completed results in this proposal.

---

## 3.1 Arsitektur Umum Penelitian

Penelitian ini mengevaluasi pengaruh preprocessing citra berbasis frekuensi-angular terhadap kinerja YOLO26 pada deteksi fine-grained cacat biji kopi. Untuk mengisolasi kontribusi preprocessing, struktur detector dipertahankan sama dan dua alur utama dibandingkan secara berpasangan.

Baseline menerima citra RGB secara langsung:

\[
I \xrightarrow{\mathrm{YOLO26n}} \hat{Y}_{\mathrm{native}}.
\]

Treatment menerapkan AF2 sebelum detector:

\[
I \xrightarrow{\mathrm{AF2}} I'
\xrightarrow{\mathrm{YOLO26n}} \hat{Y}_{\mathrm{AF2}}.
\]

Dengan demikian, AF2 ditempatkan sebagai **input frontend**, bukan sebagai modifikasi backbone, neck, maupun detection head. Kedua arm menggunakan detector YOLO26n P3–P5 yang sama, sumber pretrained yang sama, inisialisasi target head yang dipadankan, dataset yang sama, dan training schedule yang sama. Satu-satunya treatment yang dimaksudkan adalah transformasi citra AF2 yang aktif pada setiap forward pass arm AF2.

**Gambar 3.1 yang akan dibuat pada dokumen final:** diagram dua jalur paralel `Native RGB -> YOLO26n` dan `RGB -> AF2 -> YOLO26n`, kemudian bertemu pada blok evaluasi yang sama.

---

## 3.2 Dataset Penelitian

Penelitian menggunakan dataset **Faruq-v3 grouped development archive** untuk deteksi 21 kelas yang digunakan pada protokol eksperimen. Split pengembangan yang dibekukan terdiri atas data train dan validation. Seluruh kelas target terdapat pada kedua split.

### Tabel 3.1. Ringkasan dataset pengembangan

| Split | Jumlah citra | Jumlah anotasi | Jumlah kelas | Penggunaan |
|---|---:|---:|---:|---|
| Train | 1.665 | 2.986 | 21 | Optimisasi parameter YOLO26 |
| Validation | 294 | 526 | 21 | Model selection, evaluasi primary metrics, dan mechanism diagnostics |
| Locked test | Tidak dibuka pada screening | — | — | Tidak digunakan selama proposal/pilot |

Label kelas mengikuti taxonomy dataset yang digunakan oleh eksperimen. Konteks SNI digunakan untuk menjelaskan relevansi cacat fisik, tetapi sistem tidak diasumsikan merekonstruksi keseluruhan prosedur grading dan perhitungan nilai cacat SNI.

---

## 3.3 Persiapan, Pembagian, dan Audit Dataset

Dataset menggunakan grouped split yang telah diaudit terhadap kebocoran antar-split. Kontrak data mensyaratkan tidak adanya parent overlap dan exact-hash overlap antara train dan validation. Pada screening direct-from-pretrained, development root juga tidak boleh memuat direktori `test` sehingga test tidak dapat terakses secara tidak sengaja selama training, model selection, maupun diagnostic analysis.

Pemisahan ini penting karena citra turunan dari sumber yang sama dapat menghasilkan estimasi performa yang terlalu optimistis apabila masuk ke split yang berbeda. Karena itu, split diperlakukan sebagai bagian dari experimental contract dan tidak diubah sebagai fungsi dari hasil AF2.

Transformasi augmentasi yang berasal dari training pipeline YOLO diperlakukan terpisah dari AF2. AF2 bukan teknik penambahan sampel dan tidak mengubah label bounding box melalui crop, translasi, resize non-uniform, atau warp koordinat. Operator mempertahankan ukuran tensor input; perubahan yang dihasilkan berada pada nilai piksel/representasi spektral.

---

## 3.4 Preprocessing Frekuensi-Angular AF2

AF2 bekerja secara lokal pada overlapping patches, melakukan transformasi Fourier, menghitung distribusi amplitude berdasarkan arah, menekan arah dengan respons relatif rendah berdasarkan threshold adaptif berbasis entropy, lalu merekonstruksi hasil ke ruang spasial. Operator tidak memiliki parameter trainable.

### 3.4.1 Pembentukan overlapping patches

Untuk citra input

\[
I \in \mathbb{R}^{B\times 3\times H\times W},
\]

setiap channel RGB diproses secara independen. Citra dibagi menjadi patch berukuran

\[
m\times m,\qquad m=32.
\]

Dengan overlap 0,50, stride efektif adalah

\[
s=m(1-0.50)=16.
\]

Jika dimensi citra tidak tepat terhadap grid patch, implementasi melakukan padding dengan mode replicate. Setelah seluruh patch diproses, area overlap direkonstruksi menggunakan fold dan dibagi dengan jumlah kontribusi patch pada setiap posisi (*overlap averaging*).

### 3.4.2 FFT, amplitude, dan phase

Untuk patch \(P_i\), transformasi Fourier dua dimensi ditulis secara konseptual sebagai

\[
F_i=\operatorname{fftshift}\left(\mathcal{F}(P_i)\right).
\]

Amplitude/magnitude dan phase adalah

\[
A_i(u,v)=|F_i(u,v)|,
\]

\[
\phi_i(u,v)=\arg F_i(u,v).
\]

Implementasi repository menjalankan bagian FFT dalam `float32` untuk keamanan CUDA/AMP, kemudian mengembalikan output ke dtype input.

### 3.4.3 Diskretisasi sudut dan angular density

Untuk setiap koordinat frekuensi relatif terhadap pusat patch, sudut dihitung melalui

\[
\theta(u,v)=\operatorname{atan2}(v-v_c,u-u_c),
\]

kemudian dipetakan ke domain \([0,360^\circ)\). Konfigurasi menggunakan 360 angular bins sehingga implementasi melakukan floor-to-bin sebesar satu derajat. Pemilihan diskretisasi ini adalah keputusan transfer implementasi repository; parent paper memberikan domain sudut tetapi tidak menetapkan detail binning kode tersebut.

Untuk channel \(c\), angular density pada bin \(k\) dihitung dengan menjumlahkan magnitude seluruh koefisien yang jatuh pada arah tersebut:

\[
D_i^c(k)=
\sum_{(u,v):\,b(u,v)=k} A_i^c(u,v).
\]

Density kemudian dinormalisasi menjadi probabilitas:

\[
p_i^c(k)=\frac{D_i^c(k)}{\sum_jD_i^c(j)+\varepsilon}.
\]

### 3.4.4 Entropy-conditioned threshold

Entropy angular dihitung sebagai

\[
H_i^c=-\sum_k p_i^c(k)\log\left(p_i^c(k)+\varepsilon\right).
\]

Threshold adaptif AF2 kemudian

\[
\tau_i^c=
\frac{\gamma}{1+\exp(-H_i^c)},
\qquad \gamma=0.10.
\]

Karena \(H_i^c\) dihitung dari patch dan channel yang sedang diproses, threshold bersifat **content-adaptive** walaupun operator tidak mempunyai parameter yang dipelajari melalui backpropagation.

### 3.4.5 Directional amplitude weighting

Density dinormalisasi terhadap nilai maksimum pada patch/channel:

\[
q_i^c(k)=
\frac{D_i^c(k)}{\max_jD_i^c(j)+\varepsilon}.
\]

Bobot arah ditentukan oleh

\[
w_i^c(k)=
\begin{cases}
0, & q_i^c(k)\le\tau_i^c,\\
q_i^c(k), & q_i^c(k)>\tau_i^c.
\end{cases}
\]

Setiap koefisien Fourier menerima bobot sesuai angular bin-nya:

\[
\widetilde{F}_i^c(u,v)
=F_i^c(u,v)\;w_i^c(b(u,v)).
\]

AF2 mempertahankan phase asli secara implisit karena filtering dilakukan dengan mengalikan koefisien kompleks oleh bobot real non-negatif.

### 3.4.6 Inverse FFT dan residual reconstruction

Patch yang telah difilter dikembalikan ke ruang spasial:

\[
\widetilde{P}_i
=\Re\left\{
\mathcal{F}^{-1}
\left(\operatorname{ifftshift}(\widetilde F_i)\right)
\right\}.
\]

Seluruh patch kemudian digabungkan dengan overlap averaging untuk menghasilkan respons spasial

\[
R_{AF2}(I).
\]

Respons tersebut dinormalisasi per citra dan per channel menggunakan min–max normalization:

\[
G(I)=
\operatorname{MinMax}(R_{AF2}(I)).
\]

Output preprocessing adalah residual gate

\[
I' = I + I\odot G(I).
\]

Dari bentuk tersebut, dimensi spasial tetap sama:

\[
\operatorname{shape}(I')=\operatorname{shape}(I).
\]

Namun preservation shape/coordinate geometry tidak boleh ditafsirkan sebagai jaminan bahwa box prediction detector setelah training akan identik.

### 3.4.7 Parameter AF2 yang dibekukan

### Tabel 3.2. Konfigurasi AF2

| Parameter | Nilai | Peran |
|---|---:|---|
| `mode` | `af2` | Mengaktifkan angular-density filtering |
| `patch_size` | 32 | Ukuran patch lokal |
| `overlap` | 0,50 | Overlap patch; stride 16 |
| `gamma` | 0,10 | Skala entropy-conditioned threshold |
| `angular_bins` | 360 | Diskretisasi arah |
| `chunk_size` | 128 | Batas pemrosesan patch per chunk untuk memori/komputasi |
| `eps` | \(10^{-8}\) | Stabilitas numerik |
| RGB processing | independen | Tiap channel ditransformasi terpisah |
| reconstruction | fold + overlap average | Menggabungkan patch |
| trainable parameters | 0 | Operator parameter-free |

Konfigurasi bersama juga memuat `radius_ratio=0.05`, tetapi parameter tersebut digunakan oleh jalur AFAB-1/`af12` untuk radial high-pass mask dan **tidak aktif ketika `mode=af2`**. Karena itu `radius_ratio` tidak diperlakukan sebagai hyperparameter metode AF2 pada tesis ini.

---

## 3.5 YOLO26 sebagai Detector Baseline dan Treatment

Detector yang digunakan adalah YOLO26n P3–P5. Kedua arm dibangun dari official pretrained artifact yang sama. Frozen protocol mencatat SHA-256 pretrained:

```text
9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
```

Sumber pretrained memiliki 80-class head, sedangkan target Faruq-v3 memiliki 21 kelas. Oleh karena itu, tensor target head yang shape-incompatible memerlukan inisialisasi baru. Untuk mencegah random initialization menjadi confounder, kedua arm membangun 21-class detector dalam isolated RNG fork dengan seed yang sama sebelum common pretrained weights dimuat. Preflight mensyaratkan seluruh persistent detector state identik antara `D0DIRECT` dan `AF2DIRECT` sebelum training.

AF2 tidak menambahkan trainable parameter ke detector. Dengan demikian, setelah common initialization dan source transfer, satu-satunya treatment yang dimaksudkan adalah operator input AF2.

---

## 3.6 Skenario Eksperimen

Eksperimen utama menggunakan matched-control design.

### Tabel 3.3. Skenario eksperimen utama

| Arm | Initialization | Preprocessing | Detector | Fungsi |
|---|---|---|---|---|
| `D0DIRECT` | official `yolo26n.pt` + matched 21-class head | Tidak ada | YOLO26n P3–P5 | Native control |
| `AF2DIRECT` | exact same artifact/state | AF2 aktif sejak forward pertama | YOLO26n P3–P5 | Treatment |

Perbandingan utama didefinisikan sebagai paired delta:

\[
\Delta M=M_{AF2DIRECT}-M_{D0DIRECT}.
\]

Seed 42 telah digunakan sebagai screening pendahuluan. Karena screen tersebut memenuhi kriteria promosi yang dibekukan sebelumnya, proposal merencanakan konfirmasi pada tiga seed:

\[
\mathcal{S}=\{42,123,2026\}.
\]

Hasil seed 42 diperlakukan sebagai pilot; hasil seed 123 dan 2026 belum tersedia pada tahap proposal.

### Preflight fairness gates

Sebelum training, setiap paired run harus memastikan:

1. pretrained artifact hash sama;
2. model YAML sama;
3. training schedule sama;
4. AF2 mapping sama dengan konfigurasi frozen;
5. source head sesuai 80-class checkpoint;
6. detector states native dan AF2 identik sebelum training;
7. jumlah parameter detector sama;
8. AF2 mempunyai nol learned parameter;
9. probe AF2 finite, shape-preserving, dan nonzero;
10. dataset audit lulus dan test tidak terbuka.

---

## 3.7 Konfigurasi Pelatihan

Konfigurasi paired training mengikuti frozen direct-from-pretrained protocol.

### Tabel 3.4. Konfigurasi pelatihan

| Parameter | Nilai |
|---|---:|
| Maximum epoch | 50 |
| Image size | 640 |
| Batch size | 16 |
| Workers | 2 |
| Patience | 15 |
| Optimizer | `auto` |
| Pretrained | `true` |
| Cache | `false` |
| Close mosaic | 10 |
| Maximum detections | 500 |
| Deterministic | `true` |
| Planned seeds | 42, 123, 2026 |

Early stopping tetap mengikuti `patience=15`; 50 epoch merupakan batas maksimum, bukan kewajiban bahwa kedua arm harus berhenti pada epoch yang sama.

Catatan penting untuk reproducibility: repository juga memiliki konfigurasi baseline lama dengan schedule yang berbeda. Untuk penelitian direct-AF2 ini, **frozen direct-from-pretrained protocol adalah authority** dan konfigurasi lama tidak digunakan untuk mendefinisikan Bab III.

---

## 3.8 Evaluasi Performa

Evaluasi dipisahkan menjadi aggregate performance, lower-tail performance, mechanism diagnostics, dan efficiency. Pemisahan tersebut diperlukan agar peningkatan rata-rata tidak menutupi kelas yang tetap sulit dan agar perubahan detection score tidak langsung ditafsirkan sebagai localization gain.

### 3.8.1 Per-class dan Macro mAP50–95

Misalkan \(AP_c\) adalah AP kelas \(c\) yang dirata-ratakan pada rentang IoU 0,50–0,95 sesuai evaluation implementation. Dengan \(C=21\), Macro mAP50–95 didefinisikan sebagai

\[
\mathrm{Macro}=
\frac{1}{C}\sum_{c=1}^{C}AP_c.
\]

Metric ini menjadi ukuran aggregate utama pada protokol internal penelitian.

### 3.8.2 Bottom-3 dan Worst-class

Untuk menilai lower tail, urutkan per-class AP:

\[
AP_{(1)}\le AP_{(2)}\le\dots\le AP_{(C)}.
\]

Bottom-3 adalah

\[
\mathrm{Bottom3}=
\frac{AP_{(1)}+AP_{(2)}+AP_{(3)}}{3},
\]

sedangkan Worst-class adalah

\[
\mathrm{Worst}=AP_{(1)}.
\]

Bottom-3 dan Worst-class merupakan **study-defined summary metrics**, bukan klaim bahwa kedua metric tersebut merupakan standar resmi COCO.

### Tabel 3.5. Metrik utama

| Metrik | Tujuan |
|---|---|
| mAP50 / mAP50–95 | Ringkasan standar performa detector |
| Macro mAP50–95 | Rata-rata performa antarkelas |
| Bottom-3 mAP50–95 | Stabilitas lower-tail tiga kelas tersulit |
| Worst-class mAP50–95 | Safety indicator kelas dengan performa terendah |
| Per-class AP | Identifikasi kelas yang terbantu/dirugikan |

### 3.8.3 Mechanism diagnostics

Setelah checkpoint terbaik tersedia, protokol menjalankan diagnosis pada validation dengan:

```text
imgsz        = 640
match IoU    = 0.50
confidence   = 0.25
NMS IoU      = 0.70
max_det      = 500
raw counts   = 50, 100, 300, 500
```

Tiga headline diagnostic adalah:

### Tabel 3.6. Diagnostic classification–localization

| Diagnostic | Interpretasi terbatas |
|---|---|
| Raw top-500 proposal accessibility | Apakah ground-truth sudah dapat diakses oleh proposal mentah/top candidates; proxy proposal/localization availability |
| Localization-conditioned Top-1 | Ketepatan kelas setelah kondisi lokalisasi dipenuhi; proxy diskriminasi kelas |
| Correct-decision recall | Proporsi keputusan akhir benar di antara target yang dievaluasi |

Diagnostic digunakan untuk membantu atribusi pola, bukan menggantikan mAP. Jika raw proposal accessibility tetap relatif sama tetapi localization-conditioned Top-1 meningkat, wording yang diizinkan adalah hasil **lebih konsisten dengan** peningkatan diskriminasi kelas daripada peningkatan proposal accessibility. Wording causal seperti “AF2 hanya memperbaiki klasifikasi” tidak dibenarkan hanya dari diagnostic ini.

### 3.8.4 Evaluasi efisiensi

Karena AF2 tidak mempunyai parameter trainable tetapi tetap melakukan patch extraction, FFT, angular aggregation, inverse FFT, dan reconstruction, parameter-free tidak berarti compute-free. Evaluasi akhir perlu mencatat setidaknya:

- jumlah parameter detector;
- latency per image;
- throughput;
- peak GPU memory/VRAM;
- kondisi hardware/runtime yang sama pada kedua arm.

---

## 3.9 Analisis Kesalahan dan Per-Class Behavior

Analisis tidak berhenti pada aggregate mAP. Untuk setiap seed, penelitian akan memeriksa:

- perubahan AP per kelas;
- kelas yang konsisten berada pada Bottom-3;
- kelas dengan gain/loss terbesar;
- hubungan perubahan lower-tail dengan aggregate score;
- pola error classification pada proposal yang sudah terlokalisasi;
- apakah perubahan diagnostic stabil lintas seed.

Apabila hasil berbeda antar-seed, laporan harus menampilkan delta per seed dan tidak hanya mean. Kesimpulan mekanisme harus mengikuti stabilitas evidence yang benar-benar diamati.

---

## 3.10 Perangkat dan Lingkungan Eksperimen

Untuk reproducibility, setiap run harus menyimpan identitas lingkungan yang memengaruhi hasil dan pengukuran efisiensi.

### Tabel 3.7. Rekaman lingkungan eksperimen

| Komponen | Informasi yang wajib direkam |
|---|---|
| Repository | branch + commit SHA |
| Python | versi runtime |
| PyTorch | versi |
| CUDA | versi |
| Ultralytics | versi; frozen direct screen menggunakan 8.4.96 |
| GPU | model dan VRAM |
| Pretrained source | file + SHA-256 |
| Random seed | nilai per run |
| Dataset contract | versi/split/hash audit |

Perbandingan latency dan memory hanya dianggap comparable jika dilakukan menggunakan runtime, hardware, image size, precision mode, warm-up, dan measurement procedure yang sama.

---

## 3.11 Batas antara Studi Pendahuluan dan Eksperimen Tesis

Studi pendahuluan seed 42 telah menunjukkan bahwa direct-AF2 layak dipromosikan ke konfirmasi multi-seed menurut decision rule yang dibekukan sebelum training. Hasil tersebut hanya digunakan sebagai **feasibility evidence** dalam proposal.

Eksperimen tesis yang direncanakan adalah paired confirmation pada tiga seed, analisis per kelas/lower tail, mechanism diagnostics, dan accuracy–efficiency comparison. Locked test tidak digunakan untuk memilih metode atau mengubah parameter preprocessing selama screening.

Dengan pemisahan ini, hasil pilot membantu menunjukkan kelayakan metodologi tetapi tidak dipakai sebagai kesimpulan final penelitian.