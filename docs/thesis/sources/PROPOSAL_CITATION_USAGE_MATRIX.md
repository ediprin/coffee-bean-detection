# Proposal Citation Usage Matrix

Purpose: memastikan setiap sitasi yang masih muncul pada artefak formal BAB I–III memiliki satu identitas sumber yang jelas dan status verifikasi resmi/primer yang transparan.

Status mengikuti `OFFICIAL_CITATION_AUDIT.md`. File ini harus disinkronkan dengan artefak formal, bukan dengan riwayat draft lama.

| Sitasi dalam teks | Key | Bab | Status saat ini | Catatan |
|---|---|---|---|---|
| Badan Standardisasi Nasional (2008) | STD-01 | II | FINAL | Standar resmi BSN. BAB III menyebut SNI sebagai konteks dataset, tetapi citation author-year eksplisit berada di BAB II. |
| García et al. (2019) | COF-17 | I | FINAL | MDPI official verified. |
| Hong et al. (2026) | COF-01 | I, II | FINAL | Elsevier official verified. |
| Gope et al. (2024) | COF-06 | I, II | FINAL | Nature official verified. |
| Bahy dan Rifai (2026) | COF-02 | I, II | FINAL | Official journal article/PDF verified. |
| Jundullah et al. (2026) | COF-05 | I, II | FINAL | Official journal page verified. |
| Hebert dan Alamsyah (2026) | COF-04 | I, II | FINAL | Official journal page verified. |
| Kesiman et al. (2023) | COF-07 | I, II | FINAL — PRIMARY PUBLISHER PDF VERIFIED | Conference paper/IEEE-linked record verified. |
| Arwatchananukul et al. (2024) | COF-08 | II | FINAL | Elsevier official verified. |
| de Oliveira et al. (2016) | COF-10 | II | FINAL | Elsevier official verified; DOI lama pada workbook tidak digunakan. |
| Jiao et al. (2025) | COF-12 | I | FINAL | PLOS official verified. |
| Hu et al. (2025) | COF-13 | I, II | FINAL | Elsevier official verified. |
| Liu et al. (2022) | PRE-01 | I, II | FINAL | AAAI official verified. |
| Qin et al. (2022) | PRE-02 | I, II | FINAL | ACCV/CVF official verified. |
| Li et al. (2025) — FE-YOLO | PRE-03 | I, II | FINAL | Elsevier official verified. |
| Syauqi et al. (2025) | PRE-04 | I, II | FINAL — PRIMARY IEEE PDF VERIFIED | Primary IEEE PDF memuat title, authors, conference, dan DOI. |
| Chen et al. (2024) — maize seed | PRE-05 | I, II | FINAL | Elsevier official verified. |
| Yang dan Soatto (2020) | PRE-08 | II | FINAL | CVF official verified. |
| Cao et al. (2019) | SPEC-01 | I, II | FINAL | Publisher official verified. |
| Zhang dan Tan (2003) | SPEC-02 | I, II | FINAL | Elsevier official verified. |
| Xu et al. (2025) | FG-01 | I, II, III | FINAL | Elsevier official verified; parent AFAB/AFAB-2. |
| Xie et al. (2025) | FG-02 | II | FINAL — PRIMARY IEEE PDF VERIFIED | IEEE TCSVT publisher-format PDF verified. |
| Ren et al. (2015) | DET-02 | II | FINAL | NeurIPS official verified. |
| Redmon et al. (2016) | DET-03 | II | FINAL | CVF official verified. |
| Jocher et al. (2026) | DET-01 | II, III | FINAL — PRIMARY PREPRINT | Primary arXiv verified; harus tetap disebut sebagai preprint jika belum ada publisher version yang dikunci. |
| Feng et al. (2021) | DIAG-01 | II | FINAL | CVF official verified. |
| Wu et al. (2020) | DIAG-02 | II | FINAL | CVF official verified. |
| Jiang et al. (2018) | DIAG-03 | II | FINAL | ECCV/Springer official verified. |
| Chi et al. (2020) | FREQ-01 | II | FINAL | NeurIPS official verified. |
| Li et al. (2024) — FDADNet | FREQ-02 | II | FINAL | MDPI official verified. |
| Chen et al. (2025) — FDConv | FREQ-03 | II | FINAL | CVF official verified. |
| Lin et al. (2014) | EVAL-01 | III | FINAL | Springer official verified. |
| Selvaraju et al. (2017) | XAI-01 | II, III | FINAL | CVF official verified. |
| Muhammad dan Yeasin (2020) | XAI-03 | II, III | FINAL — PRIMARY PREPRINT VERIFIED | Primary Eigen-CAM paper verified; official IEEE landing juga teridentifikasi, tetapi bibliografi tidak boleh mengisi metadata yang belum dikunci dari record resmi. |

## Backend source yang sengaja tidak menjadi dependency formal

`COF-03` Samudra & Rachmawati (2025) tetap dipertahankan sebagai evidence backend. Full text primer mendukung adanya misclassification antara *black bean* dan *partially black bean* karena kemiripan visual, tetapi sumber ini telah dihapus dari BAB I dan BAB II formal agar bibliografi proposal tidak bergantung pada metadata publisher yang belum tertutup sepenuhnya.

`XAI-02` Grad-CAM++ juga tetap sebagai backend source. Sitasi eksplisitnya telah dikeluarkan; naskah formal menggunakan Grad-CAM, Eigen-CAM, dan istilah umum "varian CAM lain".

## Current gate

Cited-source set formal saat ini berjumlah **34 sumber unik**. Tidak ada lagi sumber `PRIMARY VERIFIED / PUBLISHER METADATA PENDING` di dalam cited-source set formal.

Ini **belum** berarti `DAFTAR_PUSTAKA.md` boleh diisi dari ingatan. Tahap berikutnya adalah ekstraksi metadata APA lengkap dari publisher/primary source untuk setiap satu dari 34 sumber tersebut, kemudian audit dua arah `cited -> bibliography` dan `bibliography -> cited`.

## Known discrepancies / corrections

- `COF-10`: DOI yang benar menurut ScienceDirect adalah `10.1016/j.jfoodeng.2015.10.009`; DOI workbook lama yang berakhiran `.030` tidak digunakan.
- `PRE-05`: artikel berada pada tahun 2024 walaupun DOI resmi memuat string `2023`; DOI tetap `10.1016/j.compag.2023.108475`.
- `XAI-02`: penulis pertama adalah **Aditya Chattopadhyay**. Karena sumber ini bukan dependency formal, ia tidak masuk bibliography proposal saat ini.
