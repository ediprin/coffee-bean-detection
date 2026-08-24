# Proposal Skeleton — Formal Artifact Contract

Working title:

**Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi**

Dokumen pada direktori ini diperlakukan sebagai **artefak proposal tesis**, bukan laporan hasil eksperimen repository. Naskah formal harus dapat dibaca oleh pembaca akademik yang tidak mengetahui struktur kode, nama konfigurasi internal, riwayat eksperimen, atau hasil pilot penelitian.

---

## Artefak formal utama

```text
BAB_I_PENDAHULUAN.md
BAB_II_TINJAUAN_PUSTAKA.md        # akan menjadi authority Bab II setelah rewrite formal
BAB_III_METODOLOGI_PENELITIAN.md # akan menjadi authority Bab III setelah rewrite formal
DAFTAR_PUSTAKA.md                 # akan dibangun dari sumber yang benar-benar disitasi
```

File lama seperti `02_BACKGROUND.md`, `03_PROBLEM_FORMULATION.md`, `04_LITERATURE_REVIEW.md`, `05_METHODOLOGY.md`, `05_05_AF2_PRIMARY_SOURCE_HARDENED.md`, dan `06_RESEARCH_FLOW.md` diperlakukan sebagai **bahan kerja / sumber teknis internal** sampai kontennya dipindahkan ke artefak formal yang sesuai.

---

## BAB I — Pendahuluan

Authority: `BAB_I_PENDAHULUAN.md`.

Struktur mengikuti proposal kampus acuan:

```text
1.1 Latar Belakang
1.2 Rumusan Masalah
1.3 Batasan Masalah
1.4 Tujuan Penelitian
1.5 Manfaat Penelitian
```

Bab I formal tidak menggunakan subbab `Identifikasi Masalah`, daftar `RQ1–RQ4`, `Kontribusi yang Diharapkan`, maupun `Sistematika Penulisan` kecuali kemudian diwajibkan secara eksplisit oleh program studi.

Aturan keras Bab I:

- tidak memuat hasil eksperimen penelitian sendiri;
- tidak memuat seed, D0/D0FT, hasil pilot, promotion gate, proposal accessibility, atau diagnosis hasil internal;
- tidak mengasumsikan pembaca mengetahui istilah `AF2`;
- menggunakan istilah deskriptif **preprocessing citra berbasis frekuensi-angular**;
- tidak menyatakan bahwa informasi frekuensi merupakan bottleneck cacat kopi yang sudah terbukti;
- hasil penelitian terdahulu boleh dipakai sebagai landasan masalah dan precedent metode.

---

## BAB II — Tinjauan Pustaka

Target authority: `BAB_II_TINJAUAN_PUSTAKA.md`.

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

Bab II boleh memuat hasil **penelitian terdahulu** yang telah diverifikasi. Bab II formal tidak memuat status audit, citation key internal (`COF-01`, `FG-01`, dan sebagainya), path repository, atau hasil eksperimen penelitian sendiri.

Nama implementasi internal `AF2` tidak digunakan sebagai konsep teori umum. Jika mekanisme Xu et al. dibahas, gunakan terminologi sumber seperti AFAB/AFAB-2. Penelitian yang diusulkan dijelaskan secara deskriptif sebagai preprocessing citra berbasis frekuensi-angular sebelum YOLO26.

---

## BAB III — Metodologi Penelitian

Target authority: `BAB_III_METODOLOGI_PENELITIAN.md`.

`05_METHODOLOGY.md`, `05_05_AF2_PRIMARY_SOURCE_HARDENED.md`, protokol eksperimen, konfigurasi, dan kode operator berfungsi sebagai **backend teknis** untuk memastikan metode yang ditulis benar dan reproducible. File tersebut bukan naskah yang ditempel mentah ke proposal.

Bab III formal harus menjelaskan metode dari sudut pandang pembaca akademik, dengan urutan konseptual:

```text
arsitektur / alur umum penelitian
-> dataset
-> preprocessing citra yang diusulkan
-> transformasi Fourier dan analisis angular
-> rekonstruksi citra
-> optimasi faktor/parameter preprocessing
-> YOLO26 sebagai detector
-> proses pelatihan
-> rancangan perbandingan
-> metrik evaluasi
-> evaluasi efisiensi
```

Detail internal seperti commit SHA, checkpoint hash, `D0`, `D0FT`, historical factorization genealogy, `PROMOTE_TO_3_SEED`, RNG fork, dan hasil pilot tidak dimasukkan ke naskah utama proposal.

Jika nama pendek untuk metode adaptasi ingin digunakan, istilah tersebut hanya boleh diperkenalkan setelah asal mekanisme dan perbedaannya dengan AFAB-2 dijelaskan secara eksplisit. Nama config repository tidak otomatis menjadi nama akademik metode.

---

## Temporal guardrail proposal

```text
BOLEH
- masalah penelitian
- teori dan hasil penelitian terdahulu
- metode yang diusulkan
- rancangan optimasi
- rancangan eksperimen
- parameter/metrik yang akan digunakan
- evaluasi yang akan dilakukan

TIDAK BOLEH SEBAGAI HASIL PROPOSAL
- hasil eksperimen penelitian sendiri
- hasil pilot satu seed
- historical candidate results
- diagnosis pasca-eksperimen
- klaim bahwa metode usulan sudah meningkatkan performa
```

Eksperimen yang sudah pernah dilakukan tetap dipertahankan di backend repository sebagai catatan pengembangan, tetapi tidak diubah menjadi narasi hasil proposal.

---

## Prinsip source-of-truth

```text
proposal/*.md formal artifact
    = apa yang dibaca dosen/penguji

foundation/ + sources/ + protocol/config/code
    = alasan dan bukti internal bahwa proposal tersebut benar
```

Setiap revisi substansial pada percakapan harus dipindahkan ke artefak formal yang sesuai agar keputusan tidak hanya hidup di chat.
