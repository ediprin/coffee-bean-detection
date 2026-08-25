# Citation Crosswalk — Proposal Formal

Status: **AUDIT WORKING AUTHORITY**

Dokumen ini memetakan sitasi author–year yang saat ini muncul pada artefak formal `BAB_I_PENDAHULUAN.md`, `BAB_II_TINJAUAN_PUSTAKA.md`, dan `BAB_III_METODOLOGI_PENELITIAN.md` ke key canonical dan status verifikasi pada `OFFICIAL_CITATION_AUDIT.md`.

Aturan keras:

1. Crosswalk ini **bukan daftar pustaka**.
2. `FINAL` berarti metadata yang akan ditulis harus mengikuti sumber resmi/primer yang dikunci di `OFFICIAL_CITATION_AUDIT.md`.
3. `PRIMARY VERIFIED / PUBLISHER METADATA PENDING` boleh mendukung klaim yang telah dibaca dari full text primer, tetapi field bibliografis yang belum terlihat pada sumber primer tidak boleh ditebak.
4. `OPTIONAL / UNRESOLVED` harus dihapus dari naskah formal atau ditutup auditnya sebelum `DAFTAR_PUSTAKA.md` dinyatakan citation-ready.
5. Jika file proposal berubah, crosswalk ini harus diaudit ulang dua arah.

## A. Sitasi domain kopi dan standar

| Sitasi formal | Key | Status gate | Sumber authority | Keputusan |
|---|---|---|---|---|
| Badan Standardisasi Nasional (2008) | STD-01 | FINAL — OFFICIAL VERIFIED | BSN | Aman |
| García et al. (2019) | COF-17 | FINAL — OFFICIAL VERIFIED | MDPI Applied Sciences | Aman |
| Hong et al. (2026) | COF-01 | FINAL — OFFICIAL VERIFIED | Elsevier / Current Research in Food Science | Aman |
| Bahy dan Rifai (2026) | COF-02 | FINAL — OFFICIAL VERIFIED | IJoICT official article/PDF | Aman |
| Samudra dan Rachmawati (2025) | COF-03 | PRIMARY VERIFIED / PUBLISHER METADATA PENDING | Primary project PDF; IEEE DOI/proceedings metadata corroborated | Klaim full-text aman; bibliography jangan menambah metadata yang belum dikunci |
| Hebert dan Alamsyah (2026) | COF-04 | FINAL — OFFICIAL VERIFIED | INOVTEK Polbeng official page | Aman |
| Jundullah et al. (2026) | COF-05 | FINAL — OFFICIAL VERIFIED | Brilliance official page | Aman |
| Gope et al. (2024) | COF-06 | FINAL — OFFICIAL VERIFIED | Nature / Scientific Reports | Aman |
| Kesiman et al. (2023) | COF-07 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | Primary conference PDF / IEEE record identified | Aman sesuai metadata primer yang dikunci |
| Arwatchananukul et al. (2024) | COF-08 | FINAL — OFFICIAL VERIFIED | Elsevier / Smart Agricultural Technology | Aman |
| de Oliveira et al. (2016) | COF-10 | FINAL — OFFICIAL VERIFIED | Elsevier / Journal of Food Engineering | Aman; gunakan DOI resmi `10.1016/j.jfoodeng.2015.10.009` |
| Jiao et al. (2025) | COF-12 | FINAL — OFFICIAL VERIFIED | PLOS ONE | Aman |
| Hu et al. (2025) | COF-13 | FINAL — OFFICIAL VERIFIED | Elsevier / LWT | Aman |

## B. Object detection, fine-grained, preprocessing, dan evaluasi

| Sitasi formal | Key | Status gate | Sumber authority | Keputusan |
|---|---|---|---|---|
| Ren et al. (2015) | DET-02 | FINAL — OFFICIAL VERIFIED | NeurIPS proceedings | Aman |
| Redmon et al. (2016) | DET-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Jocher et al. (2026) | DET-01 | FINAL — PRIMARY PREPRINT VERIFIED | arXiv primary preprint | Aman jika ditulis sebagai preprint, bukan publication venue yang tidak ada |
| Feng et al. (2021) | DIAG-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Wu et al. (2020) | DIAG-02 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Jiang et al. (2018) | DIAG-03 | FINAL — OFFICIAL VERIFIED | ECCV/Springer proceedings | Aman |
| Xie et al. (2025) | FG-02 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE TCSVT publisher-format PDF | Aman |
| Xu et al. (2025) | FG-01 | FINAL — OFFICIAL VERIFIED | Elsevier / Neural Networks | Aman |
| Liu et al. (2022) | PRE-01 | FINAL — OFFICIAL VERIFIED | AAAI official proceedings | Aman |
| Qin et al. (2022) | PRE-02 | FINAL — OFFICIAL VERIFIED | CVF ACCV Open Access | Aman |
| Li et al. (2025) — FE-YOLO | PRE-03 | FINAL — OFFICIAL VERIFIED | Elsevier / Digital Signal Processing | Aman |
| Syauqi et al. (2025) | PRE-04 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE publisher PDF | Aman; sebut pipeline komposit, bukan CLAHE saja |
| Chen et al. (2024) — maize seed | PRE-05 | FINAL — OFFICIAL VERIFIED | Elsevier / Computers and Electronics in Agriculture | Aman; tahun publikasi 2024, DOI tetap mengandung 2023 |
| Yang dan Soatto (2020) | PRE-08 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Cao et al. (2019) | SPEC-01 | FINAL — OFFICIAL VERIFIED | Publisher official record | Aman |
| Zhang dan Tan (2003) | SPEC-02 | FINAL — OFFICIAL VERIFIED | Elsevier / Pattern Recognition | Aman |
| Chi et al. (2020) | FREQ-01 | FINAL — OFFICIAL VERIFIED | NeurIPS proceedings | Aman |
| Li et al. (2024) — FDADNet | FREQ-02 | FINAL — OFFICIAL VERIFIED | MDPI Processes | Aman |
| Chen et al. (2025) — Frequency Dynamic Convolution | FREQ-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Lin et al. (2014) | EVAL-01 | FINAL — OFFICIAL VERIFIED | Springer / ECCV | Aman |

## C. Visualisasi aktivasi / XAI

| Sitasi formal | Key | Status gate | Sumber authority | Keputusan |
|---|---|---|---|---|
| Selvaraju et al. (2017) | XAI-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Muhammad dan Yeasin (2020) | XAI-03 | FINAL — PRIMARY PREPRINT VERIFIED | Primary Eigen-CAM paper | Aman jika versi yang disitasi ditulis transparan sesuai source primer; jangan mengarang metadata IEEE |
| Chattopadhyay et al. (2018) | XAI-02 | PRIMARY VERIFIED / OPTIONAL CITATION | Primary author preprint; IEEE landing belum dapat ditutup oleh crawler | **Belum bersih untuk bibliography publisher final. Rekomendasi: hapus sitasi eksplisit dan gunakan frasa “Grad-CAM atau variannya” kecuali official IEEE metadata ditutup.** |

## D. Status kesiapan daftar pustaka

Pada audit saat ini, hampir seluruh sitasi formal telah mempunyai source authority yang cukup. Dua perhatian tersisa adalah:

1. **COF-03 (Samudra & Rachmawati, 2025):** klaim paper telah didukung oleh full text primer. Metadata conference/DOI telah teridentifikasi, tetapi bibliography harus hanya memakai metadata yang sudah dikunci dari paper/record yang dapat diverifikasi.
2. **XAI-02 (Grad-CAM++):** sifatnya hanya alternatif metodologis. Untuk meminimalkan risiko sitasi, sitasi eksplisit sebaiknya dihapus dari naskah formal sampai official IEEE record dapat dikunci.

`DAFTAR_PUSTAKA.md` hanya boleh dibangun setelah dua keputusan di atas diselesaikan dan audit cited → bibliography / bibliography → cited dijalankan.
