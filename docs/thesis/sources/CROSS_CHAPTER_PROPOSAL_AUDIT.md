# Cross-Chapter Proposal Audit

Status: **CURRENT CONSISTENCY GATE — CLEAN SOURCE-OF-TRUTH SNAPSHOT**

Dokumen ini mengaudit konsistensi artefak formal pada:

```text
docs/thesis/proposal/
├── BAB_I_PENDAHULUAN.md
├── BAB_II_TINJAUAN_PUSTAKA.md
├── BAB_III_METODOLOGI_PENELITIAN.md
└── DAFTAR_PUSTAKA.md
```

BAB I–III duplikat yang sebelumnya berada di root repository telah dihapus. Audit ini tidak menggantikan `OFFICIAL_CITATION_AUDIT.md`, `CLAIM_LEVEL_SOURCE_AUDIT.md`, atau `BIDIRECTIONAL_CITATION_AUDIT.md`.

## 1. Bagian yang dikunci

Subbab berikut pada BAB I berstatus **LOCKED**:

- 1.2 Rumusan Masalah;
- 1.4 Tujuan Penelitian.

Konsistensi BAB II dan BAB III harus mengikuti kedua bagian tersebut. Keduanya tidak boleh diubah tanpa perintah eksplisit pengguna.

## 2. Alignment utama

### 2.1 Masalah penelitian

BAB I menempatkan masalah pada deteksi *fine-grained* cacat biji kopi, terutama ketika beberapa kategori memiliki perbedaan visual yang halus dan kinerja antarkelas dapat berbeda. BAB II menyediakan landasan dari literatur kopi, deteksi objek, prapemrosesan citra, dan representasi frekuensi. BAB III mengoperasionalkan masalah tersebut melalui prapemrosesan frekuensi-angular sebelum YOLO26n.

**Status: CONSISTENT.**

### 2.2 Dataset primer dan jumlah kelas

Dataset utama akan dikumpulkan secara primer. Daftar kelas awal menargetkan 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008 ditambah satu kelas biji normal. Jumlah kelas akhir ditentukan setelah kecukupan data tiap kelas diperiksa.

BAB III merencanakan sekitar 180–220 citra sumber, sekitar 6.000–10.000 anotasi objek, sedikitnya sekitar 200 objek asli per kelas, dan kemunculan kelas pada sekitar 30 citra sumber berbeda. Pembagian awal sekitar 70%/15%/15% dilakukan pada kelompok sumber sebelum augmentasi.

**Status: CONSISTENT.**

### 2.3 Pencegahan kebocoran data

Citra yang berasal dari kelompok sumber yang sama dipertahankan pada bagian data yang sama. Jika spesimen fisik yang sama difoto lebih dari satu kali, seluruh citra yang memuat spesimen tersebut ditempatkan pada split yang sama. Pembagian tidak hanya mengacak kelompok, tetapi juga mempertimbangkan distribusi kelas agar validation dan test mempunyai keterwakilan kelas yang memadai. Data uji disisihkan sejak awal dan tidak digunakan untuk memilih konfigurasi atau parameter.

**Status: CONSISTENT.**

### 2.4 Model dan pembanding

YOLO26n merupakan model utama. Empat kondisi utama adalah:

```text
B0 = YOLO26n tanpa prapemrosesan
B1 = CLAHE + YOLO26n
B2 = C0 + YOLO26n
B3 = C* + YOLO26n
```

CLAHE bukan tahap optimasi sebelum C0, tetapi pembanding B1 pada pengujian ulang utama. Wavelet tidak menjadi pembanding utama. RT-DETRv3-R18 hanya menjadi evaluasi tambahan setelah konfigurasi utama ditetapkan dan tidak digunakan untuk retuning C*.

**Status: CONSISTENT.**

### 2.5 Makna optimasi

Optimasi berarti analisis beberapa variasi desain prapemrosesan yang telah ditetapkan sebelum data uji digunakan, bukan pencarian *global optimum*.

Urutan pengujian utama adalah:

```text
C0 → C1 → C2 → C3 → C4 → C5
```

dengan:

- `C0`: konfigurasi frekuensi-angular referensi;
- `C1`: periodic square-root Hann dan normalized overlap-add;
- `C2`: orientasi tak bertanda pada rentang 0°–180° dengan 180 interval, sehingga resolusi tetap 1° per interval;
- `C3`: tiga pita radial berdasarkan radius ternormalisasi `[0,1/3]`, `(1/3,2/3]`, dan `(2/3,1]`, dengan entropi/ambang dihitung per pita;
- `C4`: ambang lunak dengan nilai awal `T=0,02`, yang diperlakukan sebagai keputusan desain awal dan bukan nilai optimal dari literatur;
- `C5`: panduan luminansi berdasarkan ITU-R BT.709-6.

Desain lama berupa 16 orientasi dan pembagian radial berbasis kuantil grid tidak lagi berlaku.

**Status: CONSISTENT.**

### 2.6 Konfigurasi referensi C0

Konfigurasi referensi menggunakan patch 32, overlap 50%, replicate padding, FFT ortonormal dengan FFT shift, 360 interval angular, `gamma=0,10`, `epsilon=1e-8`, dan gate RGB per kanal. Titik pusat/DC dipetakan ke interval angular pertama hanya sebagai aturan indeks. Bobot referensi berada pada rentang 0–1 sehingga tahap spektral berfungsi sebagai seleksi/penekanan respons. Rekonstruksi overlap C0 menggunakan perataan daerah tumpang tindih. Residual `I'=I+I⊙G` tidak diikuti clipping tambahan.

**Status: CONSISTENT.**

### 2.7 Analisis sensitivitas

Analisis sensitivitas terbatas dilakukan satu parameter pada satu waktu menggunakan data pengembangan, dengan kandidat:

\[
m\in\{16,32,64\},
\]

\[
\gamma\in\{0{,}05,0{,}10,0{,}15\},
\]

\[
T\in\{0{,}01,0{,}02,0{,}05\}.
\]

Ketika `m` berubah, overlap tetap 50% sehingga `stride=m/2`. Parameter lain dipertahankan pada nilai referensi selama satu sweep. Nilai terbaik dari sweep yang berbeda tidak boleh langsung digabungkan menjadi konfigurasi baru yang belum dievaluasi. Keputusan hasil sensitivitas harus dibekukan sebelum pengujian ulang multi-seed dan sebelum data uji digunakan.

**Status: CONSISTENT.**

### 2.8 Pemilihan kandidat

Tahap II tidak menggunakan checkpoint model acuan sebagai bobot awal. Seluruh C0–C5 dibangun langsung dari `yolo26n.pt` dengan seed pengembangan 42 dan kondisi awal yang dipasangkan.

Kandidat struktur dipilih berdasarkan mAP50–95 pada data validasi:

\[
C_{str}=\arg\max_{C_j}mAP_{50:95}^{val}(C_j).
\]

Selisih absolut mAP50–95 kurang dari 0,001 pada skala 0–1 diperlakukan sebagai seri. Pemecah seri berikutnya adalah rerata AP pada kelompok tiga kelas sulit yang telah dibekukan dari model acuan, kemudian waktu pemrosesan total.

Jika analisis sensitivitas dilakukan, final `C*` dipilih hanya dari `C_str` dan varian sensitivitas yang benar-benar telah dievaluasi menggunakan aturan yang sama. Jika tidak dilakukan, `C*=C_str`.

**Status: CONSISTENT.**

### 2.9 Konfirmasi multi-seed dan test

Pada setiap seed 42, 123, dan 2026, kondisi B0–B3 dibangun kembali langsung dari bobot pralatih resmi `yolo26n.pt` dengan kondisi awal model yang dipasangkan. Setelah C*, aturan pemilihan model, metrik, dan prosedur evaluasi dibekukan, seluruh checkpoint B0–B3 untuk ketiga seed dievaluasi pada data uji yang sama. Data uji tidak digunakan untuk memilih ulang konfigurasi.

**Status: CONSISTENT.**

### 2.10 Evaluasi

mAP50–95 merupakan metrik utama. mAP50, precision, recall, AP per kelas, rerata kelompok tiga kelas sulit, AP kelas terendah, hasil per-seed, analisis kesalahan, analisis visual, dan efisiensi merupakan evaluasi tambahan.

Kelompok tiga kelas sulit ditentukan satu kali dari model acuan pada data validasi dan tidak dipilih ulang dari data uji. Precision/recall mengikuti prosedur evaluasi Ultralytics yang sama dan tidak digunakan untuk tuning threshold per kondisi. Jika bootstrap dilakukan, prosedurnya berpasangan dan berbasis kelompok sumber.

**Status: CONSISTENT.**

### 2.11 Analisis visual dan efisiensi

Eigen-CAM diperlakukan sebagai visualisasi respons internal model, bukan bukti kausal dan bukan diasumsikan selalu class-specific. Visualisasi utama B0–B3 menggunakan seed 42 yang ditetapkan sebelumnya. Parameter prediksi visual dibuat sama antar kondisi.

Benchmark efisiensi utama menggunakan input 640×640 dan batch 1 pada perangkat serta presisi yang sama. I/O umum tidak dimasukkan ke overhead frontend, timing GPU menggunakan sinkronisasi CUDA, dan memori dilaporkan sebagai peak allocated GPU memory. Jumlah warm-up dan pengulangan dibekukan sebelum benchmark.

**Status: CONSISTENT.**

### 2.12 Sitasi dan daftar pustaka

Set formal terakhir berjumlah 37 sumber unik. Audit dua arah melaporkan 37/37 sumber tersitasi memiliki entri daftar pustaka dan 37/37 entri daftar pustaka digunakan dalam naskah formal.

**Status: CONSISTENT pada snapshot saat ini.**

## 3. Batas AFAB-2 dan adaptasi penelitian

BAB III membedakan secara eksplisit antara:

1. prinsip yang diadaptasi dari Xu et al. (2025): DFT lokal per patch, distribusi angular, entropi dan ambang adaptif, penekanan respons dengan densitas rendah, pembobotan amplitudo, rekonstruksi dengan fase asli, serta penggabungan ruang asal dan hasil rekonstruksi;
2. keputusan implementasi penelitian: pemrosesan per kanal RGB pada konfigurasi referensi, diskretisasi angular, overlap 50%, konstanta stabilitas numerik, replicate padding, penggabungan patch, dan residual tanpa clipping tambahan;
3. variasi `C1–C5` sebagai rancangan eksperimen penelitian sendiri.

Audit full text mengunci bahwa AFAB-2 berada pada Xu et al. (2025) §3.3.3, Persamaan (9)–(13). Persamaan (14) pada paper Xu merupakan bagian CGFI, bukan AFAB-2.

**Status: CLOSED.**

## 4. Isu yang memang baru diverifikasi saat eksperimen

### 4.1 Kompatibilitas CAM dengan YOLO26

Eigen-CAM merupakan kandidat utama visualisasi. Lapisan target dan prosedur visualisasi harus diverifikasi pada implementasi agar seluruh kondisi dibandingkan dengan prosedur yang sama.

**Status: ACCEPTABLE FOR PROPOSAL / VERIFY BEFORE EXPERIMENT.**

### 4.2 Nilai tetap yang merupakan keputusan desain

Konfigurasi CLAHE, overlap 50%, pembagian tiga pita radial, `T=0,02`, dan toleransi seri mAP 0,001 merupakan keputusan desain yang ditetapkan sebelum data uji digunakan. Nilai tersebut tidak boleh ditulis sebagai nilai optimum dari literatur jika sumber primer tidak menyatakan demikian.

**Status: ACCEPTABLE FOR PROPOSAL.**

## 5. Source-of-truth dan build

Naskah formal hanya berada pada `docs/thesis/proposal/`. Generator `tools/thesis_docx/build_proposal.py` membaca langsung direktori tersebut. Workflow GitHub Actions memverifikasi keberadaan BAB I, BAB II, BAB III, dan daftar pustaka formal sebelum melakukan build.

BAB I–III lama di root telah dihapus dan tidak memiliki jalur fallback pada proses build.

**Status: CLEAN.**

## Hard Rule

Tidak ada isu yang boleh ditutup dengan tebakan atau dengan mengubah hasil eksperimen internal menjadi bukti proposal. Jika suatu klaim memerlukan bukti primer yang belum tersedia, klaim tersebut harus dibatasi atau status auditnya tetap terbuka.
