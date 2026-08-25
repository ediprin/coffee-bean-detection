# Formal Citation Inventory — Proposal

Status: **working citation inventory; not a bibliography**

Dokumen ini memetakan seluruh sitasi author-year yang saat ini muncul pada artefak formal `BAB_I_PENDAHULUAN.md`, `BAB_II_TINJAUAN_PUSTAKA.md`, dan `BAB_III_METODOLOGI_PENELITIAN.md` ke key canonical dan status pada `OFFICIAL_CITATION_AUDIT.md`.

Aturan: daftar ini tidak boleh dipakai untuk menebak metadata APA. Metadata final tetap harus diambil dari sumber resmi/primer yang tercatat pada audit.

## BAB I — Pendahuluan

| Sitasi dalam teks | Key | Status gate |
|---|---|---|
| García et al. (2019) | COF-17 | FINAL — OFFICIAL VERIFIED |
| Hong et al. (2026) | COF-01 | FINAL — OFFICIAL VERIFIED |
| Gope et al. (2024) | COF-06 | FINAL — OFFICIAL VERIFIED |
| Bahy dan Rifai (2026) | COF-02 | FINAL — OFFICIAL VERIFIED |
| Jundullah et al. (2026) | COF-05 | FINAL — OFFICIAL VERIFIED |
| Hebert dan Alamsyah (2026) | COF-04 | FINAL — OFFICIAL VERIFIED |
| Kesiman et al. (2023) | COF-07 | FINAL — PRIMARY PUBLISHER PDF VERIFIED |
| Samudra dan Rachmawati (2025) | COF-03 | PRIMARY VERIFIED / PUBLISHER METADATA PENDING |
| Hu et al. (2025) | COF-13 | FINAL — OFFICIAL VERIFIED |
| Jiao et al. (2025) | COF-12 | FINAL — OFFICIAL VERIFIED |
| Liu et al. (2022) | PRE-01 | FINAL — OFFICIAL VERIFIED |
| Qin et al. (2022) | PRE-02 | FINAL — OFFICIAL VERIFIED |
| Li et al. (2025), FE-YOLO | PRE-03 | FINAL — OFFICIAL VERIFIED |
| Syauqi et al. (2025) | PRE-04 | FINAL — PRIMARY PUBLISHER PDF VERIFIED |
| Chen et al. (2024), maize seed cracks | PRE-05 | FINAL — OFFICIAL VERIFIED |
| Cao et al. (2019) | SPEC-01 | FINAL — OFFICIAL VERIFIED |
| Zhang dan Tan (2003) | SPEC-02 | FINAL — OFFICIAL VERIFIED |
| Xu et al. (2025) | FG-01 | FINAL — OFFICIAL VERIFIED |

**BAB I blocker:** `COF-03` masih harus ditutup metadata publisher-nya atau bibliografi harus ditulis hanya berdasarkan metadata yang benar-benar tampak pada primary publisher PDF tanpa menambahkan field dari tebakan.

## BAB II — Tinjauan Pustaka

| Sitasi dalam teks | Key | Status gate |
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
| Syauqi et al. (2025) | PRE-04 | FINAL — PRIMARY PUBLISHER PDF VERIFIED |
| Chen et al. (2024) | PRE-05 | FINAL — OFFICIAL VERIFIED |
| Liu et al. (2022) | PRE-01 | FINAL — OFFICIAL VERIFIED |
| Qin et al. (2022) | PRE-02 | FINAL — OFFICIAL VERIFIED |
| Li et al. (2025), FE-YOLO | PRE-03 | FINAL — OFFICIAL VERIFIED |
| Yang dan Soatto (2020) | PRE-08 | FINAL — OFFICIAL VERIFIED |
| Xu et al. (2025) | FG-01 | FINAL — OFFICIAL VERIFIED |
| Cao et al. (2019) | SPEC-01 | FINAL — OFFICIAL VERIFIED |
| Zhang dan Tan (2003) | SPEC-02 | FINAL — OFFICIAL VERIFIED |
| Chi et al. (2020) | FREQ-01 | FINAL — OFFICIAL VERIFIED |
| Li et al. (2024), FDADNet | FREQ-02 | FINAL — OFFICIAL VERIFIED |
| Chen et al. (2025), FDConv | FREQ-03 | FINAL — OFFICIAL VERIFIED |
| Selvaraju et al. (2017) | XAI-01 | FINAL — OFFICIAL VERIFIED |
| Muhammad dan Yeasin (2020) | XAI-03 | FINAL — PRIMARY PREPRINT VERIFIED |
| Hu et al. (2025) | COF-13 | FINAL — OFFICIAL VERIFIED |

Grad-CAM++ / `XAI-02` **tidak lagi disitasi secara eksplisit dalam artefak formal**. Ia tetap boleh disimpan sebagai backend source, tetapi tidak boleh otomatis dimasukkan ke daftar pustaka.

## BAB III — Metodologi Penelitian

| Sitasi dalam teks | Key | Status gate |
|---|---|---|
| Jocher et al. (2026) | DET-01 | FINAL — PRIMARY PREPRINT VERIFIED |
| Xu et al. (2025) | FG-01 | FINAL — OFFICIAL VERIFIED |
| Lin et al. (2014) | EVAL-01 | FINAL — OFFICIAL VERIFIED |
| Muhammad & Yeasin (2020) | XAI-03 | FINAL — PRIMARY PREPRINT VERIFIED |
| Selvaraju et al. (2017) | XAI-01 | FINAL — OFFICIAL VERIFIED |

## Unique cited-source gate

Saat inventaris ini dibuat, terdapat **35 sumber unik** yang benar-benar disitasi pada BAB I–III formal.

- 34 sumber telah memiliki status `FINAL` pada official/primary-source gate.
- 1 sumber, `COF-03` (Samudra & Rachmawati, 2025), masih `PRIMARY VERIFIED / PUBLISHER METADATA PENDING`.
- `XAI-02` Grad-CAM++ bukan bagian dari cited-source set saat ini.

Angka di atas hanya menyatakan **status gate sumber**, bukan bahwa seluruh entri APA sudah lengkap. Sebelum `DAFTAR_PUSTAKA.md` dibuat, setiap sumber masih harus memiliki author list, tahun, judul persis, venue, volume/issue/article/pages jika berlaku, dan DOI/identifier yang diambil dari source resmi/primer tanpa tebakan.

## Citation-safety correction pada artefak formal

### Selesai

1. Tabel 2.1 BAB II tidak lagi menampilkan klaim `Q1`, `Q2`, atau `SINTA 3`. Kolom tersebut telah diubah menjadi **Sumber Publikasi/Venue** dan hanya menggunakan nama jurnal/konferensi yang terkait dengan paper.
2. Nomenklatur standar pada narasi BAB II telah dinormalisasi menjadi **SNI 2907:2008**. Bentuk `SNI 01-2907-2008` hanya boleh dipertahankan apabila merupakan bagian persis dari judul publikasi terdahulu.
3. Grad-CAM++ tidak lagi menjadi sitasi eksplisit pada BAB II/BAB III sehingga `XAI-02` tidak menjadi dependency daftar pustaka formal.

### Masih diperlukan

1. Nomenklatur SNI pada BAB III masih perlu diselaraskan menjadi **SNI 2907:2008** pada narasi penelitian sendiri.
2. `COF-03` tetap tidak dipromosikan ke `FINAL` hanya berdasarkan metadata sekunder. Primary PDF mengonfirmasi judul, dua penulis, metode, dan klaim visual-similarity; venue/pages/DOI telah terkorroborasi, tetapi publisher landing IEEE belum tertutup melalui source resmi yang dapat dibaca.
3. Metadata APA lengkap setiap sumber masih harus diekstrak dari source resmi/primer sebelum bibliography formal dibuat.

## Gate berikutnya

1. Selaraskan nomenklatur SNI pada BAB III.
2. Tutup `COF-03` bila primary publisher PDF atau IEEE record memberikan metadata final yang dapat diverifikasi; jika tidak, jangan menambah metadata dari tebakan.
3. Ekstrak metadata APA lengkap hanya dari source yang lolos gate.
4. Bangun `DAFTAR_PUSTAKA.md` setelah metadata aman lengkap.
5. Lakukan audit dua arah: `cited -> bibliography` dan `bibliography -> cited`.
