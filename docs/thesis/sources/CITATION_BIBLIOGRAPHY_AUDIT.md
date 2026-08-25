# Citation ↔ Bibliography Audit — Proposal

Status: **CURRENT FORMAL CORPUS AUDITED — NOT SUBMISSION-FINAL**

Audit ini membandingkan sitasi author–year yang digunakan pada `BAB_I_PENDAHULUAN.md`, `BAB_II_TINJAUAN_PUSTAKA.md`, dan `BAB_III_METODOLOGI_PENELITIAN.md` dengan `DAFTAR_PUSTAKA.md`.

## 1. Ringkasan

- Unique bibliography entries saat ini: **35**.
- Unique canonical references yang dipetakan pada current formal corpus: **35**.
- Missing bibliography entry untuk sitasi current formal corpus: **0 berdasarkan crosswalk saat ini**.
- Bibliography entry yang sengaja tidak mempunyai sitasi current formal corpus: **0**.
- `XAI-02` / Grad-CAM++: **tidak masuk bibliography**, karena tidak lagi disitasi eksplisit pada BAB II/BAB III.

Audit author–year ↔ key menggunakan `CITATION_CROSSWALK.md`; metadata entry menggunakan `APA_METADATA_LOCK.md`.

## 2. Cited → bibliography

| Key | Sitasi formal | Bibliography | Status |
|---|---|---|---|
| STD-01 | Badan Standardisasi Nasional (2008) | Ada | MATCH |
| COF-17 | García et al. (2019) | Ada | MATCH |
| COF-01 | Hong et al. (2026) | Ada | MATCH |
| COF-02 | Bahy dan Rifai (2026) | Ada | MATCH |
| COF-03 | Samudra dan Rachmawati (2025) | Ada | MATCH |
| COF-04 | Hebert dan Alamsyah (2026) | Ada | MATCH |
| COF-05 | Jundullah et al. (2026) | Ada | MATCH |
| COF-06 | Gope et al. (2024) | Ada | MATCH |
| COF-07 | Kesiman et al. (2023) | Ada | MATCH |
| COF-08 | Arwatchananukul et al. (2024) | Ada | MATCH |
| COF-10 | de Oliveira et al. (2016) | Ada | MATCH |
| COF-12 | Jiao et al. (2025) | Ada | MATCH |
| COF-13 | Hu et al. (2025) | Ada | MATCH |
| DET-01 | Jocher et al. (2026) | Ada | MATCH |
| DET-02 | Ren et al. (2015) | Ada | MATCH |
| DET-03 | Redmon et al. (2016) | Ada | MATCH |
| DIAG-01 | Feng et al. (2021) | Ada | MATCH |
| DIAG-02 | Wu et al. (2020) | Ada | MATCH |
| DIAG-03 | Jiang et al. (2018) | Ada | MATCH |
| FG-01 | Xu et al. (2025) | Ada | MATCH |
| FG-02 | Xie et al. (2025) | Ada | MATCH |
| EVAL-01 | Lin et al. (2014) | Ada | MATCH |
| PRE-01 | Liu et al. (2022) | Ada | MATCH |
| PRE-02 | Qin et al. (2022) | Ada | MATCH |
| PRE-03 | Li et al. (2025), FE-YOLO context | Ada | MATCH |
| PRE-04 | Syauqi et al. (2025) | Ada | MATCH |
| PRE-05 | Chen et al. (2024), maize context | Ada | MATCH |
| PRE-08 | Yang dan Soatto (2020) | Ada | MATCH |
| SPEC-01 | Cao et al. (2019) | Ada | MATCH |
| SPEC-02 | Zhang dan Tan (2003) | Ada | MATCH |
| FREQ-01 | Chi et al. (2020) | Ada | MATCH |
| FREQ-02 | Li et al. (2024), FDADNet context | Ada | MATCH |
| FREQ-03 | Chen et al. (2025), FDConv context | Ada | MATCH |
| XAI-01 | Selvaraju et al. (2017) | Ada | MATCH |
| XAI-03 | Muhammad dan Yeasin (2020) | Ada | MATCH |

## 3. Bibliography → cited

Seluruh 35 entry di `DAFTAR_PUSTAKA.md` berasal dari 35 key di tabel di atas. Tidak ada reference atlas/backlog yang dimasukkan hanya untuk memperbesar jumlah referensi.

## 4. Metadata-risk checks

- COF-10 memakai DOI Elsevier yang benar: `10.1016/j.jfoodeng.2015.10.009`.
- PRE-05 tetap bertahun 2024 walaupun string DOI memuat `2023`.
- EVAL-01 memakai author list published Springer (8 author), bukan 10-author arXiv preprint.
- DIAG-03 memakai Springer version-of-record pagination **816–832**. CVF author-created copy mempunyai pagination **784–799**; kedua pagination tidak dicampur.
- FREQ-01 tidak diberi page range karena accessible NeurIPS official metadata tidak menampilkannya.
- XAI-03 pada bibliography saat ini menggunakan **primary arXiv version**, bukan metadata IEEE yang belum dikunci langsung dari IEEE landing.
- XAI-02/Grad-CAM++ tidak dimasukkan karena tidak disitasi current formal text.

## 5. Kepatuhan jumlah referensi

Pedoman tesis yang menjadi authority proyek sebelumnya dicatat mensyaratkan **minimum 50 referensi** dan sedikitnya **40% berupa jurnal penelitian**, dengan daftar pustaka hanya memuat sumber yang benar-benar disitasi.

Current formal corpus baru memiliki **35** referensi. Dengan klasifikasi source-type saat ini, **17/35 ≈ 48,6%** merupakan artikel jurnal, sehingga proporsi jurnal berada di atas 40%, tetapi **jumlah total 50 belum terpenuhi**.

Konsekuensi:

- proposal **belum boleh disebut submission-final** hanya karena bibliography sudah citation-matched;
- kekurangan 15 referensi tidak boleh dipenuhi dengan filler atau sumber yang tidak digunakan;
- penambahan sumber harus dilakukan melalui penguatan BAB II/BAB I yang memang memerlukan evidence tambahan, lalu sumber tersebut disitasi dalam teks dan melewati official citation gate yang sama.

## 6. Next gate

Sebelum DOCX final proposal:

1. perluas corpus secara substantif sampai memenuhi jumlah referensi yang diwajibkan, apabila ketentuan minimum 50 memang berlaku pada naskah proposal yang diserahkan;
2. setiap referensi tambahan harus melewati `OFFICIAL_CITATION_AUDIT.md` dan `APA_METADATA_LOCK.md`;
3. jalankan kembali audit dua arah setelah setiap perubahan sitasi;
4. jangan menyebut bibliography final sebelum jumlah, metadata, dan text-citation audit semuanya lolos.
