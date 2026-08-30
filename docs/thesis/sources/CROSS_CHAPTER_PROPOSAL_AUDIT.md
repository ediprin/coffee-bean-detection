# Cross-Chapter Proposal Audit

Status: **CURRENT CONSISTENCY GATE — CLEAN SOURCE-OF-TRUTH SNAPSHOT**

Dokumen ini mengaudit konsistensi artefak formal pada:

```text
docs/thesis/proposal/
├── 01_PROPOSAL_SKELETON.md
├── BAB_I_PENDAHULUAN.md
├── BAB_II_TINJAUAN_PUSTAKA.md
├── BAB_III_METODOLOGI_PENELITIAN.md
└── DAFTAR_PUSTAKA.md
```

BAB I–III duplikat yang sebelumnya berada di root repository telah dihapus. Audit ini tidak menggantikan `OFFICIAL_CITATION_AUDIT.md`, `CLAIM_LEVEL_SOURCE_AUDIT.md`, atau `BIDIRECTIONAL_CITATION_AUDIT.md`.

Untuk keputusan metodologi aktif, `BAB_III_METODOLOGI_PENELITIAN.md` adalah sumber utama. `RESOLUSI_BLOCKER_TEKNIS_BAB_III.md` dan `AUDIT_FINAL_BAB_III.md` mencatat hasil verifikasi kontrak final. Dokumen `REVISI_*.md` merupakan riwayat peninjauan dan tidak mengalahkan keputusan pada naskah utama apabila terdapat perbedaan historis.

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

Dataset utama akan dikumpulkan secara primer. Daftar kelas awal menargetkan 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008 ditambah satu kelas biji normal. Jumlah kelas akhir ditentukan setelah kecukupan data tiap kelas diperiksa dan sebelum split serta pelatihan utama.

BAB III merencanakan sekitar 180–220 citra sumber dengan sasaran nominal sekitar 200 citra. Pada sasaran nominal tersebut, target anotasi sekitar 6.000–10.000 objek; seluruh rentang rencana 180–220 citra dengan 30–50 objek per citra secara matematis setara sekitar 5.400–11.000 objek. Setiap kelas diupayakan memiliki sedikitnya sekitar 200 objek asli, ideal sekitar 300–500 objek, serta muncul pada sekitar 30 citra sumber berbeda.

**Status: CONSISTENT.**

### 2.3 Pencegahan kebocoran data

Split awal sekitar 70%/15%/15% dilakukan sebelum augmentasi dan berbasis `group_id`. Citra yang berasal dari sesi, susunan, atau spesimen fisik yang berkaitan harus berada pada bagian data yang sama. Audit kecukupan menggunakan jumlah objek, jumlah citra sumber, dan jumlah kelompok sumber per kelas.

Validation menargetkan setiap kelas muncul pada sedikitnya sekitar lima citra sumber. Test menargetkan sedikitnya 10 objek per kelas pada sedikitnya lima citra sumber. Data uji disisihkan sejak awal dan tidak digunakan untuk memilih konfigurasi atau parameter.

**Status: CONSISTENT.**

### 2.4 Model dan pembanding

YOLO26n merupakan model utama. Empat kondisi utama adalah:

```text
B0 = YOLO26n tanpa prapemrosesan
B1 = CLAHE + YOLO26n
B2 = C0 + YOLO26n
B3 = C* + YOLO26n
```

CLAHE bukan tahap optimasi sebelum C0, tetapi pembanding B1 pada pelatihan ulang utama. Wavelet tidak menjadi pembanding utama. RT-DETRv3-R18 hanya menjadi evaluasi tambahan setelah konfigurasi utama ditetapkan dan tidak digunakan untuk retuning C*.

**Status: CONSISTENT.**

### 2.5 Makna optimasi

Optimasi berarti analisis beberapa variasi desain prapemrosesan yang telah ditetapkan sebelum data uji digunakan, bukan pencarian *global optimum*.

Urutan pengujian utama adalah:

```text
C0 → C1 → C2 → C3 → C4 → C5
```

dengan:

- `C0`: konfigurasi frekuensi-angular referensi retained AF2;
- `C1`: periodic square-root Hann dan normalized overlap-add;
- `C2`: orientasi tak bertanda pada rentang 0°–180° dengan 180 interval, sehingga resolusi nominal 1° per interval;
- `C3`: tiga pita radial berdasarkan radius ternormalisasi `[0,1/3]`, `(1/3,2/3]`, dan `(2/3,1]`, dengan entropi/ambang dihitung per pita;
- `C4`: ambang lunak dengan nilai awal `T=0,02`, yang diperlakukan sebagai keputusan desain awal dan bukan nilai optimal dari literatur;
- `C5`: panduan luminansi berdasarkan ITU-R BT.709-6.

Rancangan bersifat kumulatif dan bukan eksperimen faktorial penuh. Konvensi DC dan kontrak keluaran residual tidak berubah diam-diam antar tahap.

**Status: CONSISTENT.**

### 2.6 Konfigurasi referensi C0

Konfigurasi referensi mengikuti retained AF2 operator di repo:

- input tensor RGB berada pada rentang dasar `[0,1]` setelah preprocessing umum YOLO;
- patch 32, overlap 50%, stride 16, dan replicate padding;
- FFT ortonormal dengan FFT shift/inverse shift;
- 360 interval angular dan pemrosesan RGB per kanal;
- titik pusat/DC dipetakan ke bin angular `0` hanya sebagai konvensi indeks implementasi, bukan klaim arah fisik;
- `gamma=0,10` dan `epsilon=1e-8`;
- bobot spektral berada pada `[0,1]` sehingga tahap ini tidak memperbesar koefisien Fourier di atas nilai asal;
- rekonstruksi overlap C0 menggunakan perataan daerah tumpang tindih;
- gate dinormalisasi min–maks per citra/per kanal;
- residual `I'=I+I⊙G` tidak diikuti clipping atau renormalisasi pasca-residual; untuk input `[0,1]`, keluaran teoritis berada pada `[0,2]`.

**Status: CONSISTENT.**

### 2.7 Analisis sensitivitas

Analisis sensitivitas terbatas dilakukan satu parameter pada satu waktu menggunakan data pengembangan, dengan kandidat:

\[
m\in\{16,32,64\},
\]

\[
\gamma\in\{0{,}05,0{,}10,0{,}15\},
\]

serta, jika konfigurasi menggunakan ambang lunak:

\[
T\in\{0{,}01,0{,}02,0{,}05\}.
\]

Ketika `m` berubah, overlap tetap 50% sehingga `stride=m/2`. Nilai terbaik dari sweep yang berbeda tidak boleh langsung digabungkan menjadi konfigurasi baru yang belum dievaluasi.

**Status: CONSISTENT.**

### 2.8 Pemilihan kandidat

Tahap I membentuk model acuan pengembangan `B0_dev` langsung dari `yolo26n.pt` menggunakan seed pengembangan 42. Checkpoint tersebut tidak digunakan sebagai bobot awal C0–C5.

Tahap II membangun kembali seluruh C0–C5 langsung dari `yolo26n.pt` menggunakan seed pengembangan 42. Kandidat struktur dipilih berdasarkan mAP50–95 validation:

\[
C_{str}=\arg\max_{C_j}mAP_{50:95}^{val}(C_j).
\]

Selisih absolut mAP50–95 kurang dari 0,001 pada skala 0–1 diperlakukan sebagai seri operasional. Pemecah seri berikutnya adalah `AP_H`, kemudian median latency total *end-to-end*. Jika sensitivitas dilakukan, `C*` dipilih hanya dari `C_str` dan varian sensitivitas yang benar-benar dievaluasi. Setelah Tahap II, `C*` dibekukan.

**Status: CONSISTENT.**

### 2.9 Konfirmasi multi-seed dan test

Seed pengembangan 42 dipisahkan dari seed konfirmasi. Tahap III menggunakan:

```text
S_conf = {123, 2026, 31415}
```

Pada setiap seed konfirmasi, B0–B3 dibangun kembali langsung dari `yolo26n.pt` dengan prosedur inisialisasi yang setara. Seed 42 boleh dilaporkan sebagai hasil pengembangan tetapi tidak dimasukkan ke rerata konfirmasi. Jika `C*=C0`, B2 dan B3 identik sehingga run duplikat tidak dilakukan.

Data uji tetap tertutup selama pengembangan dan Tahap III. Setelah `C*`, seed konfirmasi, aturan checkpoint, metrik, dan prosedur evaluasi dibekukan, checkpoint terpilih dari setiap seed dalam `S_conf` dievaluasi pada data uji yang sama. Setelah test dibuka, tidak dilakukan retuning atau pemilihan ulang konfigurasi.

**Status: CONSISTENT.**

### 2.10 Training dan evaluasi

Eksperimen utama menggunakan Ultralytics 8.4.96. Pada versi ini, fitness deteksi sama dengan mAP50–95 sehingga `best.pt` dan early stopping selaras dengan metrik utama. `optimizer=Auto` ter-resolve ke AdamW pada skala eksperimen yang direncanakan; optimizer dan learning rate aktual tetap dicatat.

YOLO26 menggunakan `end2end=True`; jalur keluaran utama tidak menjalankan NMS tambahan seperti head YOLO konvensional. Validator detect menggunakan `conf=0.001` sebagai prefilter jika confidence tidak ditentukan, sedangkan precision dan recall ringkasan berasal dari operating point maksimum kurva F1 rata-rata. Karena itu P/R bersifat metrik sekunder dan tidak digunakan untuk memilih `C*`.

Metrik utama tetap mAP50–95. mAP50, AP per kelas, `AP_H`, `AP_worst`, precision, recall, hasil per-seed, paired delta, analisis kesalahan, visualisasi, dan efisiensi merupakan evaluasi tambahan. Kelompok tiga kelas sulit ditentukan sekali dari `B0_dev` pada validation dan tidak dipilih ulang dari test. Jika bootstrap digunakan, resampling dilakukan berpasangan berdasarkan `group_id`.

**Status: CONSISTENT.**

### 2.11 Analisis visual dan efisiensi

Visualisasi utama menggunakan seed konfirmasi yang telah dibekukan:

```text
s_vis = 123
```

Eigen-CAM merupakan kandidat utama, bukan metode yang dipaksakan. Metode akhir harus diverifikasi kompatibel dengan YOLO26n dan diterapkan dengan target layer, ukuran masukan, serta normalisasi yang sama pada kondisi yang dibandingkan. Jika `C*=C0`, kondisi identik tidak ditampilkan sebagai dua metode berbeda. Visualisasi tidak digunakan sebagai bukti kausal tunggal.

Benchmark efisiensi utama menggunakan input 640×640 dan batch 1 pada perangkat/presisi yang sama. Dilaporkan waktu prapemrosesan, waktu inferensi model, dan latency total *end-to-end* yang diukur langsung. Seluruh overhead khusus frontend dimasukkan ke biaya metode, sedangkan I/O umum dikeluarkan. Timing GPU menggunakan sinkronisasi pada batas pengukuran dan pekerjaan CPU dicakup oleh wall-clock timing apabila relevan. Median latency total menjadi ukuran utama efisiensi dan tie-break operasional. Jumlah warm-up dan pengulangan dibekukan sebelum benchmark. *Peak allocated GPU memory* dan jumlah parameter dilaporkan sebagai informasi tambahan.

**Status: CONSISTENT.**

### 2.12 Sitasi dan daftar pustaka

Perubahan sinkronisasi BAB I–II tidak menambah atau menghapus sumber. Set formal terakhir tetap 37 sumber unik. Snapshot audit dua arah sebelumnya melaporkan seluruh sumber tersitasi memiliki entri daftar pustaka dan seluruh entri daftar pustaka digunakan dalam naskah formal.

**Status: CONSISTENT; bidirectional source set unchanged.**

## 3. Batas AFAB-2 dan adaptasi penelitian

BAB II dan BAB III membedakan secara eksplisit antara:

1. prinsip yang diadaptasi dari Xu et al. (2025): DFT lokal per patch, distribusi angular, entropi dan ambang adaptif, penekanan respons dengan densitas rendah, pembobotan amplitudo, rekonstruksi dengan fase asli, min–max gate, dan residual fusion;
2. keputusan implementasi penelitian: pemrosesan per kanal RGB, diskretisasi angular, overlap 50%, konstanta stabilitas numerik, replicate padding, penggabungan patch, konvensi diskret DC, dan residual tanpa clipping tambahan;
3. variasi `C1–C5` sebagai rancangan eksperimen penelitian sendiri.

Audit full text mengunci bahwa patch-wise DFT berada pada Xu et al. (2025) §3.3.1 dan AFAB-2/patch-specific chaotic amplitude suppressor berada pada §3.3.3, Persamaan (9)–(13). Paper juga menjelaskan min–max normalization pada recovered spatial domain, perkalian dengan raw spatial domain, dan residual operation. Detail diskret DC serta keputusan tidak melakukan clipping pasca-residual merupakan kontrak transfer implementasi, bukan klaim eksplisit paper.

**Status: CLOSED.**

## 4. Isu yang baru dibekukan pada SOP eksekusi

### 4.1 Kompatibilitas CAM dengan YOLO26n

Eigen-CAM merupakan kandidat utama visualisasi. Lapisan target dan prosedur visualisasi harus diverifikasi pada implementasi agar seluruh kondisi dibandingkan dengan prosedur yang sama.

**Status: ACCEPTABLE FOR PROPOSAL / VERIFY BEFORE VISUAL ANALYSIS.**

### 4.2 Benchmark latency

Jumlah warm-up dan jumlah pengulangan benchmark belum perlu menjadi angka final pada proposal, tetapi keduanya harus dibekukan sebelum hasil latency antar kondisi dibandingkan dan digunakan sebagai tie-break.

**Status: ACCEPTABLE FOR PROPOSAL / FREEZE BEFORE BENCHMARK.**

### 4.3 RT-DETRv3-R18 opsional

Jika analisis lintas arsitektur dilakukan, konfigurasi pelatihan spesifik RT-DETRv3-R18 harus ditetapkan sebelum perbandingan dan digunakan sama untuk kondisi tanpa prapemrosesan versus `C*`. `C*` tidak boleh dituning ulang.

**Status: OPTIONAL / NOT A BLOCKER.**

## 5. Source-of-truth dan build

Naskah formal hanya berada pada `docs/thesis/proposal/`. Generator `tools/thesis_docx/build_proposal.py` membaca langsung direktori tersebut. Workflow GitHub Actions memverifikasi keberadaan BAB I, BAB II, BAB III, dan daftar pustaka formal sebelum melakukan build.

`01_PROPOSAL_SKELETON.md` telah disinkronkan dengan protokol BAB III final. BAB I–III lama di root telah dihapus dan tidak memiliki jalur fallback pada proses build.

**Status: CLEAN.**

## Hard Rule

Tidak ada isu yang boleh ditutup dengan tebakan atau dengan mengubah hasil eksperimen internal menjadi bukti proposal. Jika suatu klaim memerlukan bukti primer yang belum tersedia, klaim tersebut harus dibatasi atau status auditnya tetap terbuka.
