# Proposal Skeleton — Formal Artifact Contract

Working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

Direktori `docs/thesis/proposal/` diperlakukan sebagai **artefak proposal tesis**, bukan laporan hasil eksperimen repository. Naskah formal harus dapat dibaca oleh pembaca akademik yang tidak mengetahui struktur kode, nama konfigurasi internal, riwayat eksperimen, atau hasil pilot penelitian.

## Artefak formal utama

```text
BAB_I_PENDAHULUAN.md
BAB_II_TINJAUAN_PUSTAKA.md
BAB_III_METODOLOGI_PENELITIAN.md
DAFTAR_PUSTAKA.md                 # BLOCKED sampai official citation audit selesai
```

File lama seperti `02_BACKGROUND.md`, `03_PROBLEM_FORMULATION.md`, `04_LITERATURE_REVIEW.md`, `05_METHODOLOGY.md`, `05_05_AF2_PRIMARY_SOURCE_HARDENED.md`, dan `06_RESEARCH_FLOW.md` adalah bahan kerja/backend teknis, bukan authority naskah formal.

## Official citation gate

Authority audit: `docs/thesis/sources/OFFICIAL_CITATION_AUDIT.md`.

Aturan formal:

- metadata bibliografi final harus diverifikasi dari penerbit/proceedings/badan standar resmi atau primary preprint untuk karya yang memang berupa preprint;
- master workbook dan `CANONICAL_SOURCE_KEYS.md` adalah locator internal, bukan otoritas final bibliografi;
- sumber agregator/indeks hanya boleh menjadi corroboration;
- DOI, author, title, venue, volume, issue, pages/article number, quartile, dan indexing tidak boleh ditebak;
- klaim metodologis harus tetap ditelusuri ke full text primer, bukan disimpulkan dari metadata/abstract saja;
- sumber berstatus `PENDING` tidak boleh dianggap citation-ready;
- `DAFTAR_PUSTAKA.md` baru dibuat setelah semua sumber yang benar-benar dipakai dalam Bab I–III lolos gate atau klaim yang bergantung padanya dihapus/diganti.

## BAB I — Pendahuluan

Authority: `BAB_I_PENDAHULUAN.md`.

```text
1.1 Latar Belakang
1.2 Rumusan Masalah
1.3 Batasan Masalah
1.4 Tujuan Penelitian
1.5 Manfaat Penelitian
```

Bab I tidak memuat hasil eksperimen penelitian sendiri, RQ1–RQ4, nama konfigurasi internal, maupun istilah `AF2` yang belum diperkenalkan. Istilah yang digunakan adalah **preprocessing citra berbasis frekuensi-angular**.

## BAB II — Tinjauan Pustaka

Authority: `BAB_II_TINJAUAN_PUSTAKA.md`.

```text
2.1 Biji Kopi Hijau dan Cacat Fisik Biji Kopi
2.2 Inspeksi Mutu Biji Kopi
2.3 Object Detection
2.4 You Only Look Once (YOLO)
2.5 YOLO26
2.6 Fine-Grained Object Detection
2.7 Preprocessing Citra untuk Object Detection
2.8 Representasi Citra pada Domain Frekuensi
    2.8.1 Discrete Fourier Transform dan Fast Fourier Transform
    2.8.2 Amplitudo dan Fase
    2.8.3 Representasi Radial dan Angular
    2.8.4 Pemrosesan Frekuensi pada Computer Vision
2.9 Visualisasi Aktivasi Model
2.10 Penelitian Terkait
```

Bab II boleh memuat hasil penelitian terdahulu yang telah diverifikasi. Citation key internal (`COF-01`, `FG-01`, dan sebagainya), status audit, path repository, dan hasil eksperimen penelitian sendiri tidak masuk naskah formal. AFAB/AFAB-2 disebut hanya sebagai terminologi metode sumber Xu et al.

Subbab 2.9 menjadi landasan untuk rencana analisis visual pada Bab III. Eigen-CAM menjadi kandidat utama visualisasi respons model, sedangkan Grad-CAM dan Grad-CAM++ menjadi alternatif apabila target kelas, layer target, dan aliran gradient pada YOLO26 dapat didefinisikan secara konsisten. Metadata dan klaim masing-masing metode tetap tunduk pada official citation gate. Visualisasi diposisikan sebagai analisis interpretatif pendukung dan bukan bukti kausal tunggal.

## BAB III — Metodologi Penelitian

Authority: `BAB_III_METODOLOGI_PENELITIAN.md`.

Struktur formal:

```text
3.1 Arsitektur Umum Penelitian
3.2 Dataset Penelitian
    3.2.1 Sumber dan Karakteristik Dataset
    3.2.2 Pembagian Dataset dan Pencegahan Kebocoran Data
    3.2.3 Augmentasi Data
3.3 Model Dasar YOLO26n
3.4 Preprocessing Citra Berbasis Frekuensi-Angular
    3.4.1 Pembentukan Patch Lokal
    3.4.2 Transformasi Fourier
    3.4.3 Distribusi Angular
    3.4.4 Ambang Adaptif Berdasarkan Entropi
    3.4.5 Pembobotan Respons Spektral
    3.4.6 Inverse Fourier Transform dan Rekonstruksi Citra
3.5 Analisis dan Optimasi Preprocessing
3.6 Rancangan Eksperimen
3.7 Konfigurasi Pelatihan
3.8 Evaluasi Kinerja Deteksi
3.9 Analisis Visual
    3.9.1 Visualisasi Tahapan Preprocessing
    3.9.2 Visualisasi Respons Model
    3.9.3 Visualisasi Prediksi Deteksi
3.10 Analisis Kesalahan dan Kinerja Per Kelas
3.11 Evaluasi Efisiensi Komputasi
3.12 Lingkungan Implementasi
```

Analisis visual digunakan sebagai **analisis pendukung**, bukan sebagai bukti kausal tunggal. Visualisasi yang direncanakan mencakup tahapan preprocessing, respons/aktivasi model menggunakan Eigen-CAM atau metode CAM lain yang kompatibel dengan YOLO26, serta perbandingan prediksi deteksi pada citra yang sama. Pemilihan contoh visual harus mengikuti kriteria yang konsisten agar tidak hanya menampilkan kasus yang menguntungkan metode yang diusulkan.

Bab III menjelaskan apa yang **akan dilakukan**. Detail seperti checkpoint hash, commit SHA, D0/D0FT, historical factorization genealogy, promotion gate, RNG fork, dan hasil pilot tidak dimasukkan ke naskah formal. Rancangan eksperimen, parameter pelatihan, formula metode, dan rencana evaluasi boleh dicantumkan karena merupakan bagian dari metodologi proposal.

Nama konfigurasi internal seperti `AF2C`, `AF2WIN`, `AF2ORI`, `AF2POL`, `AF2SOFT`, dan `AF2LUM` tidak digunakan. Variasi dijelaskan berdasarkan faktor akademiknya: windowing, representasi arah, struktur radial-angular, fungsi ambang, dan strategi pemrosesan warna.

## Temporal guardrail proposal

```text
BOLEH
- masalah penelitian
- teori dan hasil penelitian terdahulu yang terverifikasi
- metode yang diusulkan
- rancangan optimasi
- rancangan eksperimen
- parameter/metrik yang akan digunakan
- analisis visual yang akan dilakukan
- evaluasi yang akan dilakukan

TIDAK BOLEH SEBAGAI HASIL PROPOSAL
- hasil eksperimen penelitian sendiri
- hasil pilot satu seed
- historical candidate results
- diagnosis pasca-eksperimen
- klaim bahwa metode usulan sudah meningkatkan performa
```

## Prinsip source-of-truth

```text
proposal/*.md formal artifact
    = apa yang dibaca dosen/penguji

foundation/ + sources/ + protocol/config/code
    = alasan, bukti, dan detail teknis internal
```

Setiap revisi substansial pada percakapan harus dipindahkan ke artefak formal yang sesuai agar keputusan tidak hanya hidup di chat.
