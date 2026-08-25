# Formal Citation Coverage Audit — BAB I–III

Status: **active proposal gate**

Dokumen ini memetakan sitasi yang benar-benar dipakai pada artefak formal proposal ke sumber primer/resmi yang sudah diaudit. Daftar ini bukan daftar pustaka; fungsinya memastikan tidak ada sitasi author-year di BAB I–III yang berdiri tanpa sumber yang dapat dipertanggungjawabkan.

## Aturan

1. Sitasi formal hanya boleh dipertahankan jika identitas paper/standar telah diverifikasi melalui `OFFICIAL_CITATION_AUDIT.md`.
2. Klaim teknis harus sesuai full text primer, bukan hanya metadata.
3. Jika publisher landing tidak dapat diakses tetapi primary publisher PDF tersedia, metadata yang dipakai dibatasi pada informasi yang benar-benar tercetak pada PDF tersebut.
4. Primary preprint boleh dipakai secara transparan sebagai preprint; tidak boleh disamarkan sebagai versi publisher.
5. Sumber yang belum lolos gate tidak boleh dipakai hanya karena ada di master workbook atau pernah disebut dalam percakapan.

## BAB I — coverage

| Sitasi teks | Key | Status sumber |
|---|---|---|
| García et al. (2019) | COF-17 | FINAL — OFFICIAL VERIFIED |
| Hong et al. (2026) | COF-01 | FINAL — OFFICIAL VERIFIED |
| Gope et al. (2024) | COF-06 | FINAL — OFFICIAL VERIFIED |
| Bahy dan Rifai (2026) | COF-02 | FINAL — OFFICIAL VERIFIED |
| Jundullah et al. (2026) | COF-05 | FINAL — OFFICIAL VERIFIED |
| Hebert dan Alamsyah (2026) | COF-04 | FINAL — OFFICIAL VERIFIED |
| Kesiman et al. (2023) | COF-07 | FINAL — PRIMARY PUBLISHER PDF VERIFIED |
| Hu et al. (2025) | COF-13 | FINAL — OFFICIAL VERIFIED |
| Jiao et al. (2025) | COF-12 | FINAL — OFFICIAL VERIFIED |
| Liu et al. (2022) | PRE-01 | FINAL — OFFICIAL VERIFIED |
| Qin et al. (2022) | PRE-02 | FINAL — OFFICIAL VERIFIED |
| Li et al. (2025) | PRE-03 | FINAL — OFFICIAL VERIFIED |
| Syauqi et al. (2025) | PRE-04 | FINAL — PRIMARY PUBLISHER PDF VERIFIED |
| Chen et al. (2024) | PRE-05 | FINAL — OFFICIAL VERIFIED |
| Cao et al. (2019) | SPEC-01 | FINAL — OFFICIAL VERIFIED |
| Zhang dan Tan (2003) | SPEC-02 | FINAL — OFFICIAL VERIFIED |
| Xu et al. (2025) | FG-01 | FINAL — OFFICIAL VERIFIED |

**Keputusan hardening:** sitasi Samudra dan Rachmawati (2025) dihapus dari BAB I karena publisher landing IEEE belum tertutup secara penuh pada audit, sementara argumen fine-grained sudah didukung oleh sumber lain yang lolos gate.

## BAB II — coverage

| Sitasi teks | Key | Status sumber |
|---|---|---|
| Badan Standardisasi Nasional (2008) | STD-01 | FINAL — OFFICIAL VERIFIED |
| Kesiman et al. (2023) | COF-07 | FINAL — PRIMARY PUBLISHER PDF VERIFIED |
| Arwatchananukul et al. (2024) | COF-08 | FINAL — OFFICIAL VERIFIED |
| Bahy dan Rifai (2026) | COF-02 | FINAL — OFFICIAL VERIFIED |
| de Oliveira et al. (2016) | COF-10 | FINAL — OFFICIAL VERIFIED |
| Ren et al. (2015) | DET-02 | FINAL — OFFICIAL VERIFIED |
| Redmon et al. (2016) | DET-03 | FINAL — OFFICIAL VERIFIED |
| Feng et al. (2021) | DIAG-01 | FINAL — OFFICIAL VERIFIED |
| Wu et al. (2020) | DIAG-02 | FINAL — OFFICIAL VERIFIED |
| Jiang et al. (2018) | DIAG-03 | FINAL — OFFICIAL VERIFIED |
| Gope et al. (2024) | COF-06 | FINAL — OFFICIAL VERIFIED |
| Hong et al. (2026) | COF-01 | FINAL — OFFICIAL VERIFIED |
| Jocher et al. (2026) | DET-01 | FINAL — PRIMARY PREPRINT VERIFIED |
| Xie et al. (2025) | FG-02 | FINAL — PRIMARY PUBLISHER PDF VERIFIED |
| Jundullah et al. (2026) | COF-05 | FINAL — OFFICIAL VERIFIED |
| Hebert dan Alamsyah (2026) | COF-04 | FINAL — OFFICIAL VERIFIED |
| Hu et al. (2025) | COF-13 | FINAL — OFFICIAL VERIFIED |
| Liu et al. (2022) | PRE-01 | FINAL — OFFICIAL VERIFIED |
| Qin et al. (2022) | PRE-02 | FINAL — OFFICIAL VERIFIED |
| Li et al. (2025) | PRE-03 | FINAL — OFFICIAL VERIFIED |
| Syauqi et al. (2025) | PRE-04 | FINAL — PRIMARY PUBLISHER PDF VERIFIED |
| Chen et al. (2024) | PRE-05 | FINAL — OFFICIAL VERIFIED |
| Yang dan Soatto (2020) | PRE-08 | FINAL — OFFICIAL VERIFIED |
| Xu et al. (2025) | FG-01 | FINAL — OFFICIAL VERIFIED |
| Cao et al. (2019) | SPEC-01 | FINAL — OFFICIAL VERIFIED |
| Zhang dan Tan (2003) | SPEC-02 | FINAL — OFFICIAL VERIFIED |
| Chi et al. (2020) | FREQ-01 | FINAL — OFFICIAL VERIFIED |
| Li et al. (2024) | FREQ-02 | FINAL — OFFICIAL VERIFIED |
| Chen et al. (2025) | FREQ-03 | FINAL — OFFICIAL VERIFIED |
| Selvaraju et al. (2017) | XAI-01 | FINAL — OFFICIAL VERIFIED |
| Muhammad dan Yeasin (2020) | XAI-03 | FINAL — PRIMARY PREPRINT VERIFIED; bibliography harus transparan sebagai primary preprint kecuali IEEE record dikunci kemudian |

**Keputusan hardening:** Samudra dan Rachmawati (2025) dihapus dari narasi dan Tabel 2.1. Grad-CAM++ juga tidak dipakai sebagai sitasi formal; BAB II hanya menyebut Grad-CAM, Eigen-CAM, dan kemungkinan varian CAM lain setelah kompatibilitas teknis diverifikasi.

## BAB III — coverage

Sumber eksplisit yang dipakai pada metodologi saat ini:

| Sitasi teks | Key | Status sumber |
|---|---|---|
| Jocher et al. (2026) | DET-01 | FINAL — PRIMARY PREPRINT VERIFIED |
| Xu et al. (2025) | FG-01 | FINAL — OFFICIAL VERIFIED |
| Lin et al. (2014) | EVAL-01 | FINAL — OFFICIAL VERIFIED |
| Muhammad & Yeasin (2020) | XAI-03 | FINAL — PRIMARY PREPRINT VERIFIED |
| Selvaraju et al. (2017) | XAI-01 | FINAL — OFFICIAL VERIFIED |

Formula adaptasi frekuensi-angular pada BAB III harus tetap dilacak ke mekanisme sumber Xu et al. dan implementasi repository. Formula yang merupakan definisi/operasionalisasi penelitian sendiri harus dibedakan dari formula yang dikutip langsung dari paper.

## Gate berikutnya

Sebelum membangun `DAFTAR_PUSTAKA.md`:

1. lakukan audit exact author spelling, title, venue, year, volume/issue/article number/pages dan DOI untuk setiap key di atas;
2. jangan menambahkan indeks Q1/Q2/SINTA ke daftar pustaka karena indeks bukan elemen APA;
3. untuk `DET-01` dan `XAI-03`, gunakan status preprint secara transparan jika publisher version belum dikunci;
4. lakukan audit dua arah setelah bibliography dibentuk: **setiap sitasi harus punya entri** dan **setiap entri harus benar-benar disitasi**.
