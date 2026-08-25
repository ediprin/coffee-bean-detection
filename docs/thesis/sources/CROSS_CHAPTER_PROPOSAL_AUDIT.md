# Cross-Chapter Proposal Audit

Status: **CURRENT CONSISTENCY GATE — POST-CONSOLIDATION**

Dokumen ini mengaudit konsistensi antara artefak formal di `docs/thesis/proposal/`:

- `BAB_I_PENDAHULUAN.md`;
- `BAB_II_TINJAUAN_PUSTAKA.md`;
- `BAB_III_METODOLOGI_PENELITIAN.md`;
- `DAFTAR_PUSTAKA.md`.

Audit ini tidak menggantikan `OFFICIAL_CITATION_AUDIT.md` atau `BIDIRECTIONAL_CITATION_AUDIT.md`.

## 1. Alignment utama yang sudah konsisten

### 1.1 Masalah penelitian

BAB I menempatkan masalah pada deteksi *fine-grained* cacat biji kopi, terutama ketika beberapa kategori memiliki perbedaan visual yang halus dan kinerja antarkelas dapat berbeda. BAB II mendukung framing melalui literatur kopi, *fine-grained object detection*, prapemrosesan sebelum detector, dan pemrosesan frekuensi. BAB III mempertahankan fokus tersebut.

**Status: CONSISTENT.**

### 1.2 Dataset primer dan jumlah kelas

BAB I menyatakan dataset utama akan dikumpulkan secara primer dan menargetkan 20 kategori cacat fisik serta satu kelas biji normal. BAB II menyediakan konteks dataset multiclass sebelumnya dan contoh skala dataset deteksi multiobjek. BAB III menetapkan target sekitar 180–220 citra sumber, 6.000–10.000 anotasi objek, serta pemeriksaan kecukupan data tiap kelas sebelum jumlah kelas akhir ditetapkan.

**Status: CONSISTENT.**

### 1.3 Model dan pembanding

YOLO26n menjadi model utama. Empat kondisi utama di BAB III adalah:

```text
B0 = YOLO26n tanpa prapemrosesan
B1 = CLAHE + YOLO26n
B2 = C0 + YOLO26n
B3 = C* + YOLO26n
```

Wavelet tidak dijadikan baseline utama. RT-DETRv3-R18 tetap hanya evaluasi tambahan setelah konfigurasi utama ditetapkan.

**Status: CONSISTENT.**

### 1.4 Makna “analisis dan optimasi”

BAB I menjelaskan optimasi sebagai pengujian variasi desain yang telah ditetapkan, bukan pencarian global optimum. BAB III mengoperasionalkannya melalui jalur kumulatif `C0 → C1 → C2 → C3 → C4 → C5`.

**Status: CONSISTENT.**

### 1.5 Data uji akhir

Data uji disisihkan sejak awal dan tidak digunakan untuk memilih konfigurasi atau parameter.

**Status: CLOSED / CONSISTENT.**

### 1.6 Evaluasi

mAP50–95 menjadi metrik utama; mAP50, precision, recall, AP per kelas, kelas sulit yang dibekukan dari model acuan, AP terendah, beberapa seed, analisis visual, dan efisiensi menjadi evaluasi tambahan.

**Status: CONSISTENT.**

### 1.7 Analisis visual

BAB II menyediakan landasan Grad-CAM dan Eigen-CAM. BAB III menggunakan visualisasi hanya sebagai analisis pendukung.

**Status: CONSISTENT.**

### 1.8 Sitasi dan bibliography

Set formal saat ini berjumlah **37 sumber** setelah penambahan ITU-R BT.709-6. Crosswalk dan audit dua arah harus selalu mengikuti snapshot manuscript terbaru.

**Status: CONSISTENT pada snapshot saat ini.**

---

## 2. Isu metodologis yang telah ditutup pada audit terakhir

### 2.1 Batas AFAB-2 vs adaptasi penelitian

BAB III §3.4 sekarang membedakan secara eksplisit:

1. prinsip yang diadaptasi dari Xu et al. (2025): DFT lokal per patch, distribusi angular, entropi dan ambang adaptif, penekanan arah berdensitas rendah, pembobotan amplitudo, rekonstruksi dengan fase asli, serta penggabungan ruang asal dan hasil rekonstruksi melalui normalisasi, perkalian elemen, dan residual;
2. keputusan adaptasi penelitian: pemrosesan RGB per kanal, diskretisasi 360 interval, overlap 50%, konstanta numerik, dan penggabungan patch;
3. variasi C1–C5 sebagai rancangan eksperimen penelitian sendiri.

Audit full text juga mengunci bahwa AFAB-2 berada pada Xu et al. (2025) §3.3.3 dengan Persamaan (9)–(13). **Persamaan (14) pada paper Xu adalah bagian CGFI, bukan AFAB-2.**

**Status: CLOSED.**

### 2.2 Dasar formal koefisien luminansi

BAB III §3.5.5 sekarang mengacu pada **Recommendation ITU-R BT.709-6 (2015)**. Item 3.2 standar tersebut memberikan koefisien pembentukan sinyal luminansi:

\[
Y=0{,}2126R+0{,}7152G+0{,}0722B.
\]

Sumber telah dimasukkan ke daftar pustaka dan crosswalk formal.

**Status: CLOSED.**

### 2.3 Formula DFT/iDFT/amplitudo/fase

Gonzalez & Woods (2018) tetap digunakan sebagai landasan teori umum tanpa mengarang nomor halaman. BAB II sekarang juga mengaitkan bentuk formula yang ditampilkan dengan Xu et al. (2025, §3.1.1, Persamaan 1–4), yang full text primernya tersedia dan telah diperiksa.

Dengan demikian, tidak diperlukan nomor halaman buku yang tidak tersedia untuk menelusuri bentuk formula yang digunakan.

**Status: CLOSED untuk kebutuhan proposal; page locator buku tetap tidak diklaim.**

---

## 3. Isu yang masih perlu diverifikasi saat masuk tahap eksperimen

### 3.1 Kompatibilitas CAM dengan YOLO26

Eigen-CAM masih merupakan kandidat utama visualisasi. Lapisan target dan prosedur visualisasi harus diverifikasi saat implementasi agar perbandingan antar kondisi setara.

**Status: ACCEPTABLE FOR PROPOSAL / VERIFY BEFORE EXPERIMENT.**

### 3.2 Parameter fixed comparator dan variasi desain

Nilai seperti konfigurasi CLAHE tetap, suhu ambang lunak, jumlah orientasi, dan pembagian radial sudah diprespesifikasi dalam metodologi. Nilai tersebut tidak boleh dituning menggunakan data uji. Jika ada perubahan sebelum eksperimen utama, perubahan harus dilakukan sebelum test dibuka dan dicatat sebagai revisi protokol.

**Status: ACCEPTABLE FOR PROPOSAL.**

---

## 4. Isu lama yang telah ditutup

- independent final evaluation set: **CLOSED**;
- label Q1/Q2/SINTA tanpa audit: **CLOSED**;
- dataset Faruq sebagai dataset utama: **CLOSED**;
- 21 kelas sebagai angka yang dipaksakan: **CLOSED**;
- tidak adanya pembanding peningkatan citra: **CLOSED**;
- DETR sebagai baseline utama yang mengaburkan RQ: **CLOSED**;
- batas source-vs-adaptation AFAB-2: **CLOSED**;
- source koefisien BT.709: **CLOSED**;
- traceability formula Fourier tanpa page locator textbook: **CLOSED untuk proposal**.

## 5. Langkah sebelum DOCX

1. Jalankan audit bahasa final BAB I–III.
2. Periksa penomoran tabel, persamaan, dan cross-reference.
3. Pastikan crosswalk, metadata lock, daftar pustaka, dan audit dua arah berada pada snapshot sitasi yang sama.
4. Setelah itu, proposal dapat dipindahkan ke format DOCX sesuai template kampus.

## Hard Rule

Tidak ada isu yang boleh ditutup dengan tebakan atau dengan mengubah hasil eksperimen internal menjadi bukti proposal. Jika sumber yang diperlukan belum tersedia, status tetap OPEN sampai source resmi atau keputusan metodologis tersedia.