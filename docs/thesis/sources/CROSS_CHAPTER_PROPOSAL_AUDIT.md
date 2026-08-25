# Cross-Chapter Proposal Audit

Status: **WORKING AUTHORITY — proposal consistency gate**

Dokumen ini mengaudit konsistensi antara `BAB_I_PENDAHULUAN.md`, `BAB_II_TINJAUAN_PUSTAKA.md`, `BAB_III_METODOLOGI_PENELITIAN.md`, dan `DAFTAR_PUSTAKA.md`. Audit ini tidak menggantikan `OFFICIAL_CITATION_AUDIT.md` atau `BIDIRECTIONAL_CITATION_AUDIT.md`.

## 1. Alignment utama yang sudah konsisten

### Masalah penelitian

BAB I menempatkan masalah utama pada **deteksi fine-grained cacat biji kopi**, khususnya ketika kategori memiliki karakteristik visual yang relatif serupa dan performa antar kelas tidak seragam.

BAB II mendukung framing tersebut melalui literatur kopi, fine-grained object detection, preprocessing sebelum detector, dan pemrosesan frekuensi.

BAB III tidak mengubah masalah menjadi problem lain seperti open-set recognition, counting, segmentation, atau modifikasi arsitektur YOLO.

**Status: CONSISTENT.**

### Posisi metode

BAB I menggunakan istilah **preprocessing citra berbasis frekuensi-angular** tanpa mengasumsikan pembaca memahami nama internal repository.

BAB II menjelaskan sumber mekanisme: Fourier representation, radial/angular spectrum, dan AFAB/AFAB-2 pada Xu et al. (2025).

BAB III menempatkan preprocessing sebelum YOLO26n dan mempertahankan backbone, neck, dan detection head sebagai detector yang tidak dimodifikasi.

**Status: CONSISTENT.**

### Makna “analisis dan optimasi”

BAB I menyatakan penelitian akan menganalisis dan mengoptimasi rancangan preprocessing.

BAB III mengoperasionalkannya melalui analisis satu faktor pada satu waktu terhadap windowing, representasi arah, struktur spektral, fungsi ambang, dan strategi warna, kemudian sensitivity analysis terbatas.

**Status: CONSISTENT, tetapi provenance setiap variasi harus dibedakan antara source-derived mechanism dan engineering design choice.**

### Evaluasi

BAB I menyebut kinerja deteksi, kinerja per kelas, dan biaya komputasi.

BAB III memetakan hal tersebut ke mAP50, mAP50–95, precision, recall, AP per kelas, ringkasan kelas bawah, analisis kesalahan, analisis visual, latency, throughput, parameter, dan memory.

**Status: CONSISTENT.**

### Analisis visual

BAB II menyediakan landasan Grad-CAM dan Eigen-CAM.

BAB III menggunakan visualisasi sebagai analisis pendukung: visualisasi tahapan preprocessing, respons model, dan prediksi deteksi. Visualisasi tidak diposisikan sebagai bukti kausal tunggal.

**Status: CONSISTENT.**

---

## 2. Isu metodologis yang masih harus ditutup sebelum proposal dianggap final

### ISSUE-01 — Final evaluation set belum didefinisikan secara independen

BAB III saat ini mendeskripsikan `training` dan `validation`, sementara tahap optimasi juga menggunakan hasil evaluasi untuk memilih konfigurasi preprocessing. Jika konfigurasi dipilih dan performa akhir dilaporkan pada validation set yang sama, hasil akhir dapat menjadi optimistik karena validation set berfungsi sekaligus sebagai selection set dan final evaluation set.

**Tidak boleh diselesaikan dengan mengarang adanya test set.**

Keputusan yang harus dibuat secara eksplisit sebelum finalisasi BAB III:

- menyediakan held-out test set yang dibekukan sebelum pemilihan metode; atau
- memakai protokol evaluasi lain yang menjaga data final evaluation terpisah dari data selection.

Status saat ini: **OPEN — HIGH PRIORITY.**

### ISSUE-02 — Source-vs-adaptation pada formula preprocessing perlu dipertegas

BAB III §3.4 mengadaptasi prinsip AFAB-2 Xu et al. (2025), tetapi implementasi penelitian bukan salinan penuh AFAB/LFDet. Naskah final harus membedakan dengan jelas:

1. konsep/formula yang langsung berasal dari source paper;
2. adaptasi yang dilakukan untuk menjadikannya standalone input preprocessing sebelum YOLO26;
3. keputusan implementasi penelitian sendiri seperti residual composition, normalization, patch aggregation, dan parameter default apabila tidak identik dengan source paper.

Tujuannya agar tidak ada formula implementasi penelitian yang terbaca seolah-olah merupakan persamaan asli Xu et al.

Status saat ini: **OPEN — HIGH PRIORITY.**

### ISSUE-03 — Tabel penelitian terkait masih memuat label indeks/quartile yang memerlukan audit terpisah

Tabel 2.1 masih menggunakan label seperti `Q1`, `Q2`, `SINTA 3`, dan `Conference`. Metadata bibliografis paper sudah diaudit melalui official/primary source, tetapi status quartile/indexing adalah klaim berbeda dan harus diverifikasi pada sumber indeks yang relevan untuk periode yang dimaksud.

Sampai audit indeks dilakukan, pilihan paling aman adalah mengganti kolom `Indeks/Venue` menjadi `Sumber Publikasi` dan menampilkan nama jurnal/proceedings tanpa label quartile.

Status saat ini: **OPEN — MEDIUM/HIGH PRIORITY.**

### ISSUE-04 — Rec.709 pada variasi warna belum mempunyai source gate formal

Tabel 3.2 menyebut `Gate berbasis luminance Rec.709`. Jika Rec.709 dipertahankan sebagai istilah/metode formal, standar ITU-R BT.709 harus dimasukkan ke source audit resmi. Jika bukan bagian penting dari kontribusi, wording dapat digeneralisasi menjadi `gate berbasis luminance yang dibagi antar kanal` sampai formulasi final ditetapkan.

Status saat ini: **OPEN — MEDIUM PRIORITY.**

### ISSUE-05 — Page-level locator textbook DFT belum tersedia

Metadata Gonzalez & Woods (2018) sudah dikunci melalui publisher, dan BAB II menggunakannya sebagai theoretical anchor untuk DFT/FFT serta amplitude/phase. Namun exact page/section locator dari edisi yang dipakai belum tersimpan sebagai project source.

Tidak boleh mengarang nomor halaman. Sitasi author–year tetap sah secara bibliografis, tetapi proposal belum boleh disebut `page-level source audit complete` untuk formula fundamental tersebut.

Status saat ini: **OPEN — LOW/MEDIUM PRIORITY.**

### ISSUE-06 — Eigen-CAM bibliography menggunakan primary preprint secara transparan

BAB II dan BAB III menyebut Eigen-CAM. Source primer arXiv telah diverifikasi, sementara publisher IEEE proceedings metadata belum seluruhnya dikunci dalam project source. `DAFTAR_PUSTAKA.md` karena itu menulis Eigen-CAM sebagai preprint, bukan menyamarkannya sebagai paper IEEE.

Ini **bukan error**, tetapi status sumber harus dipertahankan transparan sampai official IEEE record dikunci penuh.

Status saat ini: **ACCEPTABLE WITH DISCLOSURE.**

---

## 3. Citation state snapshot

Menurut `BIDIRECTIONAL_CITATION_AUDIT.md` pada snapshot saat audit ini:

- unique cited sources: 36;
- bibliography entries: 36;
- cited → bibliography: 36/36;
- bibliography → cited: 36/36;
- uncited bibliography entries: 0;
- cited sources without bibliography entry: 0.

Angka tersebut **hanya berlaku untuk snapshot manuscript saat ini**. Setiap perubahan sitasi pada BAB I–III wajib memicu audit ulang.

---

## 4. Urutan penutupan sebelum DOCX

1. Tutup ISSUE-01 tentang independent final evaluation set.
2. Harden §3.4 dengan penandaan source equation vs research adaptation.
3. Hilangkan atau audit label Q1/Q2/SINTA pada Tabel 2.1.
4. Putuskan apakah Rec.709 dipertahankan dan, jika ya, audit standar resminya.
5. Jalankan ulang citation crosswalk dan bidirectional bibliography audit.
6. Audit bahasa proposal dan cross-reference gambar/tabel/persamaan.
7. Baru generate DOCX per bab.

## Hard rule

Tidak ada isu di atas yang boleh “ditutup” dengan tebakan atau dengan mengubah fakta backend menjadi fakta proposal. Jika source atau data yang dibutuhkan belum tersedia, status tetap OPEN sampai keputusan metodologis atau source resmi tersedia.