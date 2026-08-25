# Proposal Skeleton — Formal Artifact Contract

Judul kerja:

**Analisis dan Optimasi Prapemrosesan Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

Direktori `docs/thesis/proposal/` merupakan **satu-satunya sumber utama naskah proposal tesis**. Naskah formal harus dapat dibaca oleh pembaca akademik yang tidak mengetahui struktur kode, nama konfigurasi internal, riwayat eksperimen, atau hasil pilot penelitian.

## 1. Source of Truth

Artefak formal utama adalah:

```text
docs/thesis/proposal/
├── 01_PROPOSAL_SKELETON.md
├── BAB_I_PENDAHULUAN.md
├── BAB_II_TINJAUAN_PUSTAKA.md
├── BAB_III_METODOLOGI_PENELITIAN.md
└── DAFTAR_PUSTAKA.md
```

BAB I–III duplikat yang sebelumnya berada di root repository telah dihapus. Semua agent yang mengerjakan proposal harus membaca dan mengubah file di `docs/thesis/proposal/`. Generator DOCX juga hanya membaca naskah formal dari direktori tersebut.

## 2. BAB I — Pendahuluan

Authority: `docs/thesis/proposal/BAB_I_PENDAHULUAN.md`.

```text
1.1 Latar Belakang
1.2 Rumusan Masalah
1.3 Batasan Masalah
1.4 Tujuan Penelitian
1.5 Manfaat Penelitian
```

### LOCKED SECTIONS

**Subbab 1.2 Rumusan Masalah dan Subbab 1.4 Tujuan Penelitian berstatus LOCKED.**

- Isi kedua subbab tersebut tidak boleh ditulis ulang, dipecah menjadi butir baru, diperluas, dipersempit, atau diubah redaksinya tanpa perintah eksplisit dari pengguna.
- Audit konsistensi harus menyesuaikan BAB II dan BAB III terhadap Rumusan Masalah dan Tujuan Penelitian yang sudah ada, bukan sebaliknya.
- Perubahan pada Latar Belakang, Batasan Masalah, atau Metodologi tidak otomatis memberi izin untuk mengubah 1.2 atau 1.4.

Kondisi metodologis yang harus konsisten dengan BAB III:

- dataset utama adalah dataset primer yang akan dikumpulkan;
- target awal adalah 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008 ditambah satu kelas biji normal;
- jumlah kelas akhir ditentukan setelah kecukupan data tiap kelas diperiksa;
- YOLO26n merupakan model utama;
- YOLO26n tanpa prapemrosesan merupakan model acuan;
- CLAHE digunakan sebagai pembanding peningkatan kontras lokal;
- optimasi berarti pengujian variasi desain prapemrosesan yang telah ditetapkan, bukan pencarian global optimum;
- RT-DETRv3-R18 hanya merupakan evaluasi tambahan jika sumber daya memungkinkan;
- mAP50–95 merupakan metrik utama.

BAB I tidak memuat hasil eksperimen penelitian sendiri, nama branch eksperimen, nama konfigurasi internal, atau nilai performa internal.

## 3. BAB II — Tinjauan Pustaka

Authority: `docs/thesis/proposal/BAB_II_TINJAUAN_PUSTAKA.md`.

```text
2.1 Biji Kopi Hijau, Cacat Fisik, dan Benda Asing
2.2 Inspeksi Mutu Biji Kopi
2.3 Deteksi Objek
2.4 You Only Look Once (YOLO)
2.5 YOLO26 dan Pembanding Arsitektur
2.6 Fine-Grained Object Detection
2.7 Prapemrosesan Citra untuk Deteksi Objek
2.8 Representasi Citra pada Domain Frekuensi
    2.8.1 Discrete Fourier Transform dan Fast Fourier Transform
    2.8.2 Amplitudo dan Fase
    2.8.3 Representasi Radial dan Angular
    2.8.4 Pemrosesan Frekuensi pada Computer Vision
2.9 Visualisasi Aktivasi Model
2.10 Penelitian Terkait
```

BAB II boleh memuat hasil penelitian terdahulu yang telah diverifikasi dari sumber primer/resmi. Hasil penelitian terdahulu tidak boleh ditulis sebagai bukti bahwa metode yang diusulkan pasti efektif pada dataset tesis.

Guardrail penting:

- AFAB/AFAB-2 disebut sebagai metode sumber Xu et al. (2025), bukan sebagai metode yang diciptakan penelitian ini;
- penelitian hanya mengadaptasi prinsip AFAB-2 sebagai konfigurasi referensi pada ruang masukan, bukan keseluruhan LFDet, AFAB-1, CGFI, atau FTIF;
- Syauqi et al. (2025) diperlakukan sebagai pipeline prapemrosesan komposit, sehingga hasilnya tidak boleh diatribusikan kepada CLAHE saja;
- wavelet merupakan alternatif transformasi multiskala yang relevan, tetapi bukan baseline utama penelitian;
- RT-DETRv3-R18 hanya menjadi dasar evaluasi transfer antararsitektur yang bersifat tambahan;
- literatur tidak digunakan untuk mengklaim bahwa cacat biji kopi memiliki *frequency signature* yang unik.

## 4. BAB III — Metodologi Penelitian

Authority: `docs/thesis/proposal/BAB_III_METODOLOGI_PENELITIAN.md`.

Struktur formal saat ini:

```text
3.1 Rancangan Umum Penelitian
3.2 Dataset Penelitian
    3.2.1 Sumber, Target Jumlah, dan Karakteristik Dataset Primer
    3.2.2 Pemeriksaan Kecukupan Data dan Penetapan Kelas
    3.2.3 Pembagian Data dan Pencegahan Kebocoran
    3.2.4 Augmentasi Data
3.3 Model Dasar YOLO26n
    3.3.1 Model Acuan dan Pembanding
3.4 Prapemrosesan Citra Berbasis Frekuensi-Angular
    3.4.1 Pembentukan Patch Lokal
    3.4.2 Transformasi Fourier
    3.4.3 Distribusi Angular
    3.4.4 Ambang Adaptif Berdasarkan Entropi
    3.4.5 Pembobotan Respons Spektral
    3.4.6 Rekonstruksi dan Penggabungan Residual
3.5 Analisis Variasi Desain Prapemrosesan
    3.5.1 Variasi Fungsi Jendela
    3.5.2 Variasi Representasi Orientasi
    3.5.3 Variasi Radial-Angular
    3.5.4 Variasi Ambang Lunak
    3.5.5 Variasi Panduan Luminansi
    3.5.6 Analisis Sensitivitas Terbatas
3.6 Rancangan Eksperimen
    3.6.1 Tahap I — Pembentukan Model Acuan
    3.6.2 Tahap II — Pengujian Variasi Prapemrosesan
    3.6.3 Tahap III — Pengujian Ulang dengan Beberapa Seed
    3.6.4 Evaluasi pada Arsitektur Lain — Opsional
    3.6.5 Evaluasi Akhir pada Data Uji
3.7 Konfigurasi Pelatihan
3.8 Evaluasi Kinerja Deteksi
3.9 Analisis Visual
3.10 Analisis Kesalahan dan Kinerja Per Kelas
3.11 Evaluasi Efisiensi Komputasi
3.12 Lingkungan Implementasi
```

Alur Gambar 3.1 harus mengikuti urutan:

```text
Pengumpulan dataset primer
→ anotasi dan pemeriksaan kecukupan
→ pembagian data
→ pembentukan model acuan YOLO26n
→ konfigurasi referensi C0
→ pengujian variasi desain C0–C5
→ pemilihan C*
→ pengujian ulang beberapa seed (B0, B1, B2, B3)
→ evaluasi akhir pada data uji
→ analisis per kelas, kesalahan, visual, dan efisiensi
→ kesimpulan
```

CLAHE tidak ditempatkan sebagai tahap optimasi sebelum C0; CLAHE merupakan kondisi pembanding B1 pada pengujian ulang utama.

Rancangan dataset primer yang berlaku:

```text
Target citra sumber      : sekitar 180–220, nominal sekitar 200
Target kelas awal        : 20 cacat fisik/benda asing + 1 normal
Target anotasi total     : sekitar 6.000–10.000 objek
Minimum awal per kelas   : sekitar 200 objek asli
Target ideal per kelas   : sekitar 300–500 objek
Kemunculan per kelas     : sedikitnya sekitar 30 citra sumber
Split awal               : sekitar 70% / 15% / 15%
Representasi val/test    : diupayakan ≥5 citra sumber per kelas pada masing-masing bagian
```

Jumlah kelas akhir tidak dipaksakan menjadi 21 apabila data primer untuk kelas tertentu tidak memenuhi kriteria kecukupan yang ditetapkan sebelum pelatihan. Split harus berbasis kelompok sumber/spesimen dan sekaligus mempertimbangkan distribusi kelas; bukan sekadar random split terhadap citra atau kelompok.

Pembanding utama:

```text
B0 = YOLO26n tanpa prapemrosesan
B1 = CLAHE + YOLO26n
B2 = konfigurasi frekuensi-angular referensi C0 + YOLO26n
B3 = konfigurasi terpilih C* + YOLO26n
```

Konfigurasi referensi C0 harus konsisten dengan keputusan implementasi yang telah diaudit:

- patch `m=32`, overlap 50%, stride 16;
- *replicate padding* untuk melengkapi grid patch dan hasil dipotong kembali ke ukuran asli;
- FFT ortonormal, spektrum dipusatkan dengan FFT shift dan dikembalikan dengan inverse FFT shift;
- distribusi angular 360 interval pada `[0,2π)` dan pemrosesan RGB per kanal;
- titik pusat/DC dipetakan ke interval angular pertama sebagai aturan indeks, bukan sebagai klaim arah fisik;
- `gamma=0,10`, `epsilon=1e-8`;
- bobot referensi berada pada `[0,1]` sehingga tahap spektral melakukan seleksi/penekanan, bukan amplifikasi koefisien;
- rekonstruksi overlap pada C0 menggunakan perataan daerah tumpang tindih;
- gate dinormalisasi min–maks per citra/per kanal;
- residual `I' = I + I⊙G` tidak diikuti clipping tambahan; untuk input dasar `[0,1]`, keluaran teoritis dapat mencapai `[0,2]`.

Konfigurasi variasi prapemrosesan diuji secara bertahap dan kumulatif:

```text
C0 → C1 → C2 → C3 → C4 → C5
```

Keputusan desain yang berlaku adalah:

```text
C1 = periodic square-root Hann + normalized overlap-add
C2 = orientasi tak bertanda pada [0, π) dengan 180 interval; resolusi tetap 1°/interval
C3 = tiga pita radial radius ternormalisasi: [0,1/3], (1/3,2/3], (2/3,1]
C4 = ambang lunak sigmoid; T=0,02 merupakan nilai awal, bukan nilai optimum dari literatur
C5 = panduan luminansi bersama menggunakan koefisien ITU-R BT.709-6
```

Desain lama berupa 16 orientasi dan pembagian radial berbasis kuantil grid tidak lagi berlaku. Pada C3, entropi, ambang adaptif, dan normalisasi densitas dihitung secara terpisah di setiap pita radial.

Analisis sensitivitas bersifat terbatas dan satu-parameter-pada-satu-waktu, bukan pencarian faktorial penuh:

```text
m     ∈ {16, 32, 64}
gamma ∈ {0,05, 0,10, 0,15}
T     ∈ {0,01, 0,02, 0,05}
```

Jika `m` berubah, overlap tetap 50% sehingga `stride=m/2`. Parameter lain tetap pada nilai referensi selama satu sweep. Nilai terbaik dari sweep yang berbeda tidak boleh digabungkan menjadi konfigurasi baru kecuali kombinasi tersebut benar-benar didefinisikan dan diuji pada data pengembangan sebelum multi-seed dan sebelum data uji digunakan.

Hubungan `C0 → C1 → ... → C5` menunjukkan akumulasi keputusan desain, bukan pewarisan bobot model antar konfigurasi.

Tahap I membentuk model acuan dari `yolo26n.pt` untuk memperoleh baseline validasi, menentukan kelompok tiga kelas sulit, dan memeriksa pipeline. **Checkpoint model acuan tidak digunakan sebagai bobot awal C0–C5.**

Tahap II mengharuskan setiap C0–C5 dibangun kembali langsung dari `yolo26n.pt` menggunakan seed pengembangan 42 dengan kondisi inisialisasi yang dipasangkan. Kandidat struktur dipilih sebagai `C_str` berdasarkan mAP50–95 validation. Selisih absolut mAP50–95 kurang dari 0,001 pada skala 0–1 diperlakukan sebagai seri; pemecah seri berikutnya adalah AP kelompok tiga kelas sulit, kemudian waktu pemrosesan total.

Jika sensitivitas dilakukan, final `C*` dipilih hanya dari `C_str` dan varian sensitivitas yang benar-benar telah dievaluasi menggunakan aturan yang sama. Jika sensitivitas tidak dilakukan, `C*=C_str`.

Tahap III menguji ulang B0–B3 pada seed 42, 123, dan 2026. Pada setiap seed seluruh kondisi kembali dibangun dari `yolo26n.pt`; tidak ada pewarisan checkpoint screening. Setelah konfigurasi dan aturan evaluasi dibekukan, seluruh checkpoint B0–B3 untuk ketiga seed dievaluasi pada data uji yang sama.

RT-DETRv3-R18 hanya merupakan evaluasi tambahan. `C*` dan parameter prapemrosesan tidak boleh dituning ulang khusus untuk RT-DETR.

Metrik utama adalah mAP50–95. Precision/recall mengikuti prosedur evaluasi Ultralytics yang sama dan bersifat sekunder. Kelompok tiga kelas sulit ditentukan sekali dari model acuan pada validation lalu dibekukan. `AP_worst` hanya indikator tambahan. Jika bootstrap dilakukan, gunakan *paired bootstrap* berbasis kelompok sumber.

Visualisasi utama B0/B1/B2/B3 menggunakan seed 42 yang ditetapkan sebelumnya, bukan seed yang dipilih setelah melihat visual. Eigen-CAM diperlakukan sebagai visualisasi respons internal/aktivasi fitur, bukan diklaim sebagai bukti kausal atau selalu class-specific.

Benchmark efisiensi utama menggunakan input 640×640 dan batch 1 pada perangkat/presisi yang sama. Waktu I/O umum dikeluarkan dari perbandingan frontend, sinkronisasi CUDA dilakukan untuk timing GPU, dan memori dilaporkan sebagai peak allocated GPU memory. Jumlah warm-up dan pengulangan harus dibekukan sebelum benchmark.

## 5. Daftar Pustaka dan Source Audit

Authority bibliography formal:

`docs/thesis/proposal/DAFTAR_PUSTAKA.md`

Backend utama yang dipertahankan:

```text
docs/thesis/sources/OFFICIAL_CITATION_AUDIT.md
docs/thesis/sources/CITATION_CROSSWALK.md
docs/thesis/sources/BIBLIOGRAPHY_METADATA_LOCK.md
docs/thesis/sources/BIDIRECTIONAL_CITATION_AUDIT.md
docs/thesis/sources/CLAIM_LEVEL_SOURCE_AUDIT.md
docs/thesis/sources/CROSS_CHAPTER_PROPOSAL_AUDIT.md
```

Set formal terakhir berjumlah 37 sumber unik. Setiap perubahan sitasi harus mengikuti alur:

```text
PERUBAHAN NASKAH
      ↓
CITATION_CROSSWALK
      ↓
OFFICIAL / PRIMARY SOURCE CHECK
      ↓
BIBLIOGRAPHY_METADATA_LOCK
      ↓
DAFTAR_PUSTAKA
      ↓
BIDIRECTIONAL_CITATION_AUDIT
```

Daftar pustaka lengkap tidak sama dengan kebenaran seluruh klaim. Klaim metodologis atau faktual yang sensitif tetap harus ditelusuri ke full text primer.

## 6. Temporal Guardrail Proposal

BAB I–III hanya menjelaskan:

- masalah penelitian;
- teori dan hasil penelitian terdahulu;
- metode yang diusulkan;
- dataset yang akan dikumpulkan;
- rancangan optimasi dan pembanding;
- rancangan eksperimen;
- parameter dan metrik yang akan digunakan;
- analisis visual dan efisiensi yang akan dilakukan.

BAB I–III **tidak boleh** memuat sebagai hasil proposal:

- hasil eksperimen penelitian sendiri;
- hasil pilot satu seed;
- hasil kandidat historis;
- konfigurasi yang disebut telah terbukti terbaik;
- diagnosis pasca-eksperimen;
- klaim bahwa metode usulan telah meningkatkan performa.

## 7. Gaya Bahasa Formal

Bahasa utama proposal adalah bahasa Indonesia yang natural. Istilah teknis Inggris boleh dipertahankan apabila lebih lazim dalam bidang pembelajaran mesin, misalnya *seed*, *optimizer*, *patch*, *backbone*, *neck*, *pretrained*, dan *bootstrap*. Hindari terjemahan literal yang membuat istilah menjadi rancu.

Detail kode yang tidak diperlukan untuk memahami metodologi ditempatkan pada konfigurasi implementasi, bukan memenuhi narasi utama proposal.
