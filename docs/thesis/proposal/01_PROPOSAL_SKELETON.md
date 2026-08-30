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

Untuk keputusan metodologi aktif, `BAB_III_METODOLOGI_PENELITIAN.md` merupakan sumber utama. `RESOLUSI_BLOCKER_TEKNIS_BAB_III.md` dan `AUDIT_FINAL_BAB_III.md` mencatat hasil verifikasi yang melandasi kontrak final. Dokumen `REVISI_*.md` dipertahankan sebagai riwayat peninjauan dan tidak mengalahkan keputusan pada naskah utama apabila terdapat perbedaan historis.

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
- literatur tidak digunakan untuk mengklaim bahwa cacat biji kopi memiliki *frequency signature* yang unik;
- visualisasi aktivasi bersifat analisis pendukung dan tidak digunakan sebagai bukti kausal atau dasar pemilihan konfigurasi.

## 4. BAB III — Metodologi Penelitian

Authority: `docs/thesis/proposal/BAB_III_METODOLOGI_PENELITIAN.md`.

Struktur formal saat ini:

```text
3.1 Rancangan Umum Penelitian
3.2 Dataset Penelitian
    3.2.1 Sumber dan Karakteristik Dataset Primer
    3.2.2 Target Pengumpulan dan Pemeriksaan Kecukupan Data
    3.2.3 Akuisisi Citra dan Anotasi
    3.2.4 Pembagian Data dan Pencegahan Kebocoran
    3.2.5 Augmentasi Data
3.3 Model Dasar YOLO26n
    3.3.1 Kondisi Eksperimen Utama dan Pembanding
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
    3.6.1 Tahap I — Pembentukan Model Acuan Pengembangan
    3.6.2 Tahap II — Pengujian Variasi Prapemrosesan
    3.6.3 Tahap III — Pelatihan Ulang dengan Beberapa Seed Konfirmasi
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
→ anotasi dan pemeriksaan kecukupan data
→ pembagian data berbasis kelompok sumber
→ pembentukan model acuan pengembangan
→ konfigurasi referensi C0
→ pengujian variasi desain C0–C5
→ pemilihan C*
→ pelatihan ulang beberapa seed konfirmasi (B0, B1, B2, B3)
→ evaluasi akhir pada data uji
→ analisis per kelas, kesalahan, visual, dan efisiensi
→ kesimpulan
```

CLAHE tidak ditempatkan sebagai tahap optimasi sebelum C0; CLAHE merupakan kondisi pembanding B1 pada pelatihan ulang utama.

Rancangan dataset primer yang berlaku:

```text
Target citra sumber      : sekitar 180–220, nominal sekitar 200
Target kelas awal        : 20 cacat fisik/benda asing + 1 normal
Target anotasi nominal   : sekitar 6.000–10.000 objek pada sasaran ~200 citra
Rentang teoritis rencana : sekitar 5.400–11.000 objek untuk 180–220 citra × 30–50 objek
Minimum awal per kelas   : sekitar 200 objek asli
Target ideal per kelas   : sekitar 300–500 objek
Kemunculan per kelas     : sedikitnya sekitar 30 citra sumber
Split awal               : sekitar 70% / 15% / 15%
Target validasi          : setiap kelas diupayakan muncul pada ≥5 citra sumber
Target data uji          : ≥10 objek per kelas pada ≥5 citra sumber
```

Jumlah kelas akhir tidak dipaksakan menjadi 21 apabila data primer untuk kelas tertentu tidak memenuhi kriteria kecukupan yang ditetapkan sebelum split dan pelatihan. Audit kecukupan menggunakan jumlah objek, jumlah citra sumber, dan jumlah kelompok sumber per kelas. Split dilakukan berdasarkan `group_id`; citra atau spesimen yang berkaitan tidak boleh tersebar ke train, validation, dan test. Augmentasi hanya dilakukan pada data pelatihan setelah split dibekukan.

Pembanding utama:

```text
B0 = YOLO26n tanpa prapemrosesan
B1 = CLAHE + YOLO26n
B2 = konfigurasi frekuensi-angular referensi C0 + YOLO26n
B3 = konfigurasi terpilih C* + YOLO26n
```

Konfigurasi referensi C0 harus konsisten dengan keputusan implementasi retained AF2 yang telah diaudit:

- input adalah tensor RGB floating point pada rentang dasar `[0,1]` setelah preprocessing umum YOLO;
- patch `m=32`, overlap 50%, stride 16;
- *replicate padding* untuk melengkapi grid patch dan hasil dipotong kembali ke ukuran asli;
- FFT ortonormal, spektrum dipusatkan dengan FFT shift dan dikembalikan dengan inverse FFT shift;
- distribusi angular 360 interval pada `[0,2π)` dan pemrosesan RGB per kanal;
- titik pusat/DC dipetakan ke bin angular `0` sebagai konvensi indeks implementasi, bukan sebagai klaim arah fisik;
- `gamma=0,10`, `epsilon=1e-8`;
- bobot referensi berada pada `[0,1]` sehingga tahap spektral melakukan seleksi/penekanan, bukan amplifikasi koefisien Fourier di atas nilai asal;
- rekonstruksi overlap pada C0 menggunakan perataan daerah tumpang tindih;
- gate dinormalisasi min–maks per citra/per kanal;
- residual `I' = I + I⊙G` tidak diikuti clipping atau renormalisasi tambahan; untuk input dasar `[0,1]`, keluaran teoritis berada pada `[0,2]`.

Konfigurasi variasi prapemrosesan diuji secara bertahap dan kumulatif:

```text
C0 → C1 → C2 → C3 → C4 → C5
```

Keputusan desain yang berlaku adalah:

```text
C1 = periodic square-root Hann + normalized overlap-add
C2 = orientasi tak bertanda pada [0, π) dengan 180 interval; resolusi nominal 1°/interval
C3 = tiga pita radial radius ternormalisasi: [0,1/3], (1/3,2/3], (2/3,1]
C4 = ambang lunak sigmoid; T=0,02 merupakan nilai awal, bukan nilai optimum dari literatur
C5 = panduan luminansi bersama menggunakan koefisien ITU-R BT.709-6
```

Konvensi DC diwariskan antarkonfigurasi agar satu tahap tidak mengubah dua faktor sekaligus. Pada C3, DC berada pada pita radial pertama dan bin orientasi 0. Entropi, ambang adaptif, dan normalisasi densitas dihitung secara terpisah di setiap pita radial.

Analisis sensitivitas bersifat terbatas dan satu-parameter-pada-satu-waktu, bukan pencarian faktorial penuh:

```text
m     ∈ {16, 32, 64}
gamma ∈ {0,05, 0,10, 0,15}
T     ∈ {0,01, 0,02, 0,05}  # hanya jika konfigurasi memakai ambang lunak
```

Jika `m` berubah, overlap tetap 50% sehingga `stride=m/2`. Parameter lain tetap pada nilai referensi selama satu sweep. Nilai terbaik dari sweep yang berbeda tidak boleh digabungkan menjadi konfigurasi baru kecuali kombinasi tersebut benar-benar didefinisikan dan diuji pada data pengembangan sebelum seed konfirmasi dan sebelum data uji digunakan.

Hubungan `C0 → C1 → ... → C5` menunjukkan akumulasi keputusan desain, bukan pewarisan bobot model antar konfigurasi.

Tahap I membentuk `B0_dev` langsung dari `yolo26n.pt` menggunakan seed pengembangan 42 untuk memperoleh baseline validasi, menetapkan kelompok tiga kelas sulit, dan memeriksa pipeline. **Checkpoint model acuan tidak digunakan sebagai bobot awal C0–C5.**

Tahap II mengharuskan setiap C0–C5 dibangun kembali langsung dari `yolo26n.pt` menggunakan seed pengembangan 42 dengan prosedur inisialisasi yang setara. Kandidat struktur dipilih sebagai `C_str` berdasarkan mAP50–95 validation. Selisih absolut mAP50–95 kurang dari 0,001 pada skala 0–1 diperlakukan sebagai seri operasional; pemecah seri berikutnya adalah `AP_H`, kemudian median latency total *end-to-end* berdasarkan protokol efisiensi.

Jika sensitivitas dilakukan, final `C*` dipilih hanya dari `C_str` dan varian sensitivitas yang benar-benar telah dievaluasi menggunakan aturan yang sama. Jika sensitivitas tidak dilakukan, `C*=C_str`. Setelah Tahap II, `C*` dibekukan dan tidak dipilih ulang berdasarkan seed konfirmasi atau data uji.

Tahap III menggunakan seed konfirmasi yang tidak dipakai untuk memilih `C*`:

```text
S_conf = {123, 2026, 31415}
```

Pada setiap seed, B0–B3 dibangun kembali langsung dari `yolo26n.pt`; seed pengembangan 42 tidak dimasukkan ke rerata konfirmasi. Jika `C*=C0`, B2 dan B3 identik sehingga run duplikat tidak dilakukan. Data uji tetap tertutup selama Tahap III dan baru digunakan setelah `C*`, seed, checkpoint rule, metrik, dan prosedur evaluasi dibekukan. Checkpoint terpilih dari setiap seed konfirmasi kemudian dievaluasi pada data uji yang sama.

RT-DETRv3-R18 hanya merupakan evaluasi tambahan. `C*` dan parameter prapemrosesan tidak boleh dituning ulang khusus untuk RT-DETR. Konfigurasi pelatihan yang spesifik RT-DETR harus ditetapkan sama untuk pasangan tanpa prapemrosesan dan dengan `C*`.

Eksperimen utama menggunakan Ultralytics 8.4.96. Untuk deteksi pada versi ini, fitness sama dengan mAP50–95 sehingga `best.pt` dan early stopping selaras dengan metrik utama. `optimizer=Auto` ter-resolve ke AdamW pada skala eksperimen yang direncanakan; optimizer dan learning rate aktual tetap dicatat untuk setiap run. YOLO26 menggunakan jalur `end2end=True`, sehingga output utama tidak menggunakan NMS tambahan seperti head YOLO konvensional.

Metrik utama adalah mAP50–95. mAP50, precision, dan recall bersifat sekunder. Precision/recall ringkasan Ultralytics berasal dari operating point maksimum kurva F1 rata-rata, sedangkan `conf=0.001` pada validator hanya berfungsi sebagai prefilter. Kelompok tiga kelas sulit ditentukan sekali dari `B0_dev` pada validation lalu dibekukan. `AP_worst` hanya indikator tambahan. Pada seed konfirmasi, hasil dan delta terhadap B0 dilaporkan secara berpasangan. Jika bootstrap dilakukan, gunakan *paired bootstrap* berbasis `group_id` dan jangan menjadikannya inferensi utama jika jumlah kelompok independen terlalu sedikit.

Visualisasi utama B0/B1/B2/B3 menggunakan seed konfirmasi yang telah ditetapkan sebelumnya:

```text
s_vis = 123
```

Metode visualisasi aktivasi harus diverifikasi kompatibel dengan YOLO26n sebelum digunakan. Eigen-CAM merupakan kandidat utama, bukan metode yang dipaksakan; metode, target layer, ukuran masukan, dan normalisasi heatmap harus sama antar kondisi. Jika `C*=C0`, kondisi identik tidak perlu divisualisasikan dua kali. Visualisasi tidak digunakan sebagai bukti kausal tunggal.

Benchmark efisiensi utama menggunakan input 640×640 dan batch 1 pada perangkat/presisi yang sama. Dilaporkan `t_pra`, `t_model`, dan latency total *end-to-end* yang diukur langsung. Seluruh overhead khusus frontend, termasuk konversi dan transfer perangkat yang diperlukan, masuk ke biaya metode. Waktu I/O umum dikeluarkan, sinkronisasi GPU dilakukan pada batas timing, dan pekerjaan CPU tercakup dalam wall-clock timing apabila relevan. Median latency total menjadi ukuran utama efisiensi dan tie-break operasional. Jumlah warm-up dan pengulangan dibekukan sebelum benchmark. *Peak allocated GPU memory* dan jumlah parameter dilaporkan sebagai informasi tambahan.

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
