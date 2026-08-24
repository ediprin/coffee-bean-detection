# Proposal Skeleton — Synchronized

Working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

Status: working title. Dokumen proposal formal harus ditulis sebagai **rencana penelitian**, bukan laporan hasil eksperimen yang sudah dilakukan di repository.

---

## Bab I — Pendahuluan

Struktur Bab I mengikuti proposal kampus yang dijadikan acuan:

```text
1.1 Latar Belakang
1.2 Rumusan Masalah
1.3 Batasan Masalah
1.4 Tujuan Penelitian
1.5 Manfaat Penelitian
```

Bab I formal **tidak** menggunakan subbab `Identifikasi Masalah`, daftar `RQ1–RQ4`, `Kontribusi yang Diharapkan`, maupun `Sistematika Penulisan` kecuali kemudian diwajibkan secara eksplisit oleh program studi.

### 1.1 Latar Belakang

Authoritative draft: `02_BACKGROUND.md`.

Rantai argumentasi:

```text
mutu dan inspeksi fisik biji kopi
-> keterbatasan inspeksi visual manusia
-> computer vision dan deep learning
-> kelayakan keluarga YOLO pada domain kopi
-> taxonomy yang lebih rinci memperlihatkan visual similarity dan class-wise disparity
-> kebutuhan diskriminasi fine-grained
-> penelitian kopi yang ditinjau banyak meningkatkan representasi di dalam model
-> preprocessing citra sebagai ruang solusi alternatif
-> frequency/angular processing sebagai mekanisme yang layak diuji
-> usulan preprocessing frekuensi-angular sebelum YOLO26
-> analisis dan optimasi rancangan preprocessing
-> evaluasi kinerja deteksi dan biaya komputasi
```

Aturan keras Bab I:

- jangan memasukkan hasil eksperimen penelitian sendiri;
- jangan memasukkan seed, hasil pilot, D0/D0FT, Bottom-3/Worst hasil aktual, proposal accessibility, atau istilah internal repository;
- jangan mengasumsikan singkatan `AF2` sudah dikenal pembaca;
- gunakan istilah deskriptif **preprocessing citra berbasis frekuensi-angular** pada Bab I;
- jangan menyatakan bahwa frekuensi adalah bottleneck cacat kopi yang sudah terbukti;
- hasil penelitian terdahulu boleh digunakan sebagai bukti masalah dan precedent metode.

### 1.2 Rumusan Masalah

Authoritative draft: `03_PROBLEM_FORMULATION.md`.

Ditulis sebagai **satu paragraf naratif**, mengikuti gaya proposal kampus. Rumusan merangkum:

- kesulitan diskriminasi pada kategori cacat yang rinci;
- kecenderungan literatur kopi yang ditinjau memodifikasi komponen internal model;
- kebutuhan mengkaji preprocessing berbasis frekuensi dan arah;
- kebutuhan menganalisis dan mengoptimasi preprocessing tersebut pada YOLO26.

Tidak menggunakan daftar RQ pada naskah formal.

### 1.3 Batasan Masalah

Ditulis sebagai daftar bernomor yang ringkas dan dapat dipahami pembaca tanpa mengetahui repository. Ruang lingkup utama:

- object detection biji kopi hijau, 21 kelas pada dataset penelitian;
- YOLO26n sebagai detector;
- optimasi pada preprocessing input, bukan backbone/neck/head;
- faktor dan parameter preprocessing dijelaskan rinci di Bab III;
- evaluasi deteksi dan efisiensi;
- tidak membahas cita rasa, roasting, dan keseluruhan proses grading kopi.

Nama kandidat internal seperti `AF2C`, `AF2WIN`, `AF2ORI`, `AF2POL`, `AF2SOFT`, dan `AF2LUM` tidak dimasukkan ke Bab I.

### 1.4 Tujuan Penelitian

Ditulis sebagai satu paragraf ringkas seperti proposal acuan: menganalisis dan mengoptimasi preprocessing frekuensi-angular pada YOLO26, lalu mengevaluasi pengaruhnya terhadap kinerja deteksi dan biaya komputasi.

### 1.5 Manfaat Penelitian

Ditulis sebagai beberapa poin sederhana mengenai manfaat kajian metode, informasi dampak preprocessing terhadap deteksi, dan referensi bagi penelitian lanjutan.

---

## Bab II — Tinjauan Pustaka

Main draft: `04_LITERATURE_REVIEW.md`.

Struktur kerja:

```text
2.1 Biji Kopi Hijau dan Cacat Fisik Biji Kopi
2.2 Inspeksi Mutu Biji Kopi: Metode Konvensional dan Tantangannya
2.3 Object Detection
2.4 YOLO
2.5 YOLO26
2.6 Fine-Grained Object Detection
2.7 Preprocessing Citra untuk Object Detection
2.8 Representasi Citra pada Domain Frekuensi
2.9 Penelitian Terkait
```

Bab II boleh memuat hasil **penelitian terdahulu** yang telah diverifikasi. Catatan audit internal, citation key, dan hasil eksperimen repository sendiri tidak masuk naskah formal.

---

## Bab III — Metode Penelitian

Base technical source: `05_METHODOLOGY.md` dan `05_05_AF2_PRIMARY_SOURCE_HARDENED.md`.

Kedua file tersebut adalah **sumber teknis internal**, bukan teks yang boleh ditempel mentah ke proposal formal. Bab III formal harus ditulis ulang dengan bahasa akademik yang menjelaskan metode kepada pembaca yang belum mengetahui repository.

Urutan konseptual yang dipertahankan:

```text
kerangka penelitian
-> dataset
-> YOLO26 sebagai baseline
-> preprocessing frekuensi-angular yang diusulkan
-> mekanisme preprocessing
-> analisis dan optimasi rancangan
-> rancangan perbandingan eksperimen
-> konfigurasi pelatihan
-> metrik evaluasi
-> analisis hasil yang direncanakan
-> evaluasi efisiensi
```

Nama `AF2` hanya boleh digunakan setelah asal mekanismenya dan definisi istilah tersebut diperkenalkan secara eksplisit pada Bab III. Detail repository seperti commit SHA, checkpoint hash, RNG fork, historical D0 genealogy, dan hasil pilot tidak dimasukkan ke naskah utama proposal kecuali diperlukan sebagai lampiran teknis terpisah.

---

## Proposal-only temporal guardrail

Proposal menjelaskan **apa yang akan dilakukan**. Karena itu:

```text
HASIL PENELITIAN SENDIRI -> tidak masuk Bab I–III proposal formal
PROTOKOL / RENCANA EKSPERIMEN -> boleh masuk Bab III
HASIL PENELITIAN TERDAHULU -> boleh dipakai sebagai landasan teori / gap
```

Eksperimen yang telah dilakukan di repository tetap disimpan sebagai evidence internal untuk pengembangan penelitian, tetapi tidak digunakan sebagai hasil pada naskah proposal.
