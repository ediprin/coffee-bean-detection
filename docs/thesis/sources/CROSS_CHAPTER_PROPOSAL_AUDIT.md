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

BAB I menempatkan masalah pada deteksi *fine-grained* cacat biji kopi, terutama ketika beberapa kategori memiliki perbedaan visual yang halus dan kinerja antarkelas dapat berbeda.

BAB II mendukung framing melalui literatur kopi, *fine-grained object detection*, prapemrosesan sebelum detector, dan pemrosesan frekuensi.

BAB III mempertahankan fokus tersebut dan tidak mengubah penelitian menjadi counting, segmentation, open-set recognition, atau modifikasi arsitektur utama YOLO26n.

**Status: CONSISTENT.**

### 1.2 Dataset primer dan jumlah kelas

BAB I menyatakan dataset utama akan dikumpulkan secara primer dan menargetkan 20 kategori cacat fisik serta satu kelas biji normal.

BAB II menyediakan konteks dataset multiclass sebelumnya dan contoh skala dataset deteksi multiobjek.

BAB III menetapkan target sekitar 180–220 citra sumber, 6.000–10.000 anotasi objek, dan pemeriksaan kecukupan data tiap kelas sebelum jumlah kelas akhir ditetapkan.

Ketiga bab tidak lagi menyatakan bahwa 21 kelas pasti tersedia sebelum pengumpulan data selesai.

**Status: CONSISTENT.**

### 1.3 Model dan pembanding

BAB I menetapkan YOLO26n sebagai model utama, YOLO26n tanpa prapemrosesan sebagai model acuan, CLAHE sebagai pembanding peningkatan kontras, dan RT-DETRv3-R18 sebagai analisis tambahan.

BAB II menyediakan dasar literatur untuk YOLO26, CLAHE/composite preprocessing, wavelet sebagai alternatif konseptual, dan RT-DETRv3 sebagai detector keluarga Transformer.

BAB III mengoperasionalkan empat kondisi utama:

```text
B0 = YOLO26n tanpa prapemrosesan
B1 = CLAHE + YOLO26n
B2 = C0 + YOLO26n
B3 = C* + YOLO26n
```

Wavelet tidak dijadikan baseline utama dan evaluasi RT-DETRv3 tetap opsional.

**Status: CONSISTENT.**

### 1.4 Makna “analisis dan optimasi”

BAB I menjelaskan bahwa optimasi dilakukan melalui pengujian variasi desain yang telah ditetapkan, bukan pencarian global optimum.

BAB III mengoperasionalkannya melalui jalur kumulatif `C0 → C1 → C2 → C3 → C4 → C5` yang mencakup fungsi jendela, representasi orientasi, informasi radial, fungsi ambang, dan panduan luminansi.

**Status: CONSISTENT.**

### 1.5 Data uji akhir

BAB III sekarang menyisihkan data uji sejak awal dengan target sekitar 15% dari citra sumber. Data uji tidak digunakan untuk memilih konfigurasi atau parameter dan baru digunakan setelah metode serta aturan evaluasi ditetapkan.

BAB I tidak menggunakan hasil data uji untuk merumuskan metode, dan BAB II hanya berfungsi sebagai dasar literatur.

**Status: CLOSED / CONSISTENT.**

### 1.6 Evaluasi

BAB I menetapkan mAP50–95 sebagai metrik utama dan mAP50, precision, recall, serta AP per kelas sebagai metrik tambahan.

BAB III menggunakan mAP50–95 sebagai metrik utama, menambahkan analisis tiga kelas sulit yang dibekukan dari model acuan pada validasi, AP kelas terendah, analisis beberapa seed, visualisasi, dan efisiensi komputasi.

**Status: CONSISTENT.**

### 1.7 Analisis visual

BAB II menyediakan landasan Grad-CAM dan Eigen-CAM.

BAB III menggunakan visualisasi hanya sebagai analisis pendukung dan tidak sebagai bukti kausal tunggal.

**Status: CONSISTENT.**

### 1.8 Sitasi dan bibliography

`CITATION_CROSSWALK.md`, `BIBLIOGRAPHY_METADATA_LOCK.md`, `DAFTAR_PUSTAKA.md`, dan `BIDIRECTIONAL_CITATION_AUDIT.md` telah disinkronkan pada set formal 36 sumber.

**Status: CONSISTENT pada snapshot saat ini.**

---

## 2. Isu yang masih terbuka

### ISSUE-01 — Source mechanism vs adaptasi penelitian pada §3.4

BAB III telah menyatakan bahwa mekanisme frekuensi-angular mengadaptasi AFAB-2 Xu et al. (2025), tetapi naskah formal sengaja menyederhanakan sebagian detail implementasi agar tidak berubah menjadi dokumentasi kode.

Sebelum proposal akhir, perlu dipastikan bahwa pembaca tetap dapat membedakan:

1. prinsip yang berasal dari Xu et al. (2025), seperti DFT patch lokal, distribusi angular, entropi, ambang adaptif, dan pembobotan spektral;
2. keputusan implementasi penelitian, seperti overlap 50%, cara pemrosesan kanal RGB, rekonstruksi overlap, normalisasi respons, dan penggabungan residual;
3. variasi desain penelitian pada C1–C5.

Tidak perlu mengembalikan seluruh detail kode, tetapi batas atribusi harus tetap jelas.

**Status: OPEN — HIGH PRIORITY sebelum proposal final.**

### ISSUE-02 — Dasar formal luminansi Rec.709

BAB III menggunakan koefisien luminansi Rec.709 pada variasi C5:

\[
Y=0{,}2126R+0{,}7152G+0{,}0722B.
\]

Jika formulasi tersebut dipertahankan sebagai bagian formal metode, standar atau sumber resmi ITU-R BT.709 sebaiknya masuk source audit. Alternatifnya adalah tidak memberi atribusi standar yang belum diaudit dan menyatakan koefisien sebagai keputusan implementasi yang perlu diberi sumber sebelum proposal final.

**Status: OPEN — MEDIUM PRIORITY.**

### ISSUE-03 — Locator halaman textbook untuk DFT/FFT

Metadata Gonzalez & Woods (2018) telah dikunci dari Pearson dan aman sebagai landasan teori. Namun, exact page/section locator untuk formula DFT/iDFT dan amplitudo/fase belum tersedia dalam project source.

Tidak boleh mengarang nomor halaman.

**Status: OPEN — LOW/MEDIUM PRIORITY.**

### ISSUE-04 — Kompatibilitas teknis CAM dengan YOLO26

BAB II dan BAB III menempatkan Eigen-CAM sebagai kandidat visualisasi utama dan metode CAM lain sebagai alternatif. Sebelum eksekusi tesis, lapisan target dan prosedur visualisasi harus diverifikasi agar perbandingan antar kondisi benar-benar setara.

Ini bukan masalah terhadap proposal selama CAM tetap diposisikan sebagai analisis pendukung.

**Status: ACCEPTABLE FOR PROPOSAL / VERIFY BEFORE EXPERIMENT.**

---

## 3. Isu lama yang telah ditutup

- **Independent final evaluation set:** CLOSED — data uji sekarang disisihkan sejak awal.
- **Label Q1/Q2/SINTA pada tabel penelitian terkait:** CLOSED — tabel menggunakan nama venue/publikasi tanpa klaim indeks yang tidak diaudit.
- **Dataset lama Faruq sebagai dataset utama:** CLOSED — proposal sekarang menggunakan rencana dataset primer.
- **21 kelas sebagai angka yang dipaksakan:** CLOSED — 21 adalah target awal; jumlah kelas akhir mengikuti kecukupan data.
- **Tidak adanya pembanding peningkatan citra:** CLOSED — CLAHE ditambahkan sebagai pembanding konvensional.
- **DETR sebagai baseline utama yang mengaburkan RQ:** CLOSED — RT-DETRv3-R18 hanya analisis tambahan setelah C* ditetapkan.

---

## 4. Urutan penutupan berikutnya

1. Tegaskan atribusi source-vs-adaptation pada §3.4 tanpa mengembalikan gaya bahasa yang terlalu teknis.
2. Audit sumber resmi Rec.709 jika variasi luminansi tetap dipertahankan.
3. Jika tersedia, tambahkan locator halaman/section Gonzalez & Woods tanpa menebak.
4. Jalankan audit bahasa final dan cross-reference tabel/persamaan.
5. Jalankan kembali citation crosswalk apabila ada penambahan sumber.
6. Setelah itu baru siapkan artefak DOCX proposal.

## Hard Rule

Tidak ada isu yang boleh ditutup dengan tebakan atau dengan mengubah hasil eksperimen internal menjadi bukti proposal. Jika sumber yang diperlukan belum tersedia, status tetap OPEN sampai source resmi atau keputusan metodologis tersedia.
