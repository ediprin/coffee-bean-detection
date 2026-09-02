# Citation Crosswalk — Proposal Formal

Status: **AUDIT WORKING AUTHORITY**

Dokumen ini memetakan sitasi author–year yang saat ini muncul pada artefak formal `docs/thesis/proposal/BAB_I_PENDAHULUAN.md`, `BAB_II_TINJAUAN_PUSTAKA.md`, dan `BAB_III_METODOLOGI_PENELITIAN.md` ke key canonical dan status verifikasi pada source-audit repository.

Aturan keras:

1. Crosswalk ini **bukan daftar pustaka**.
2. Status source harus mengikuti sumber resmi/primer yang dikunci pada audit repository.
3. Sumber yang tidak lagi disitasi pada artefak formal tidak boleh masuk `DAFTAR_PUSTAKA.md` hanya karena pernah muncul pada drafting lama.
4. Jika file proposal berubah, crosswalk ini harus diaudit ulang dua arah.

## A. Sitasi domain kopi dan standar

| Sitasi formal | Key | Status gate | Sumber authority | Keputusan |
|---|---|---|---|---|
| Badan Standardisasi Nasional (2008) | STD-01 | FINAL — OFFICIAL VERIFIED | BSN | Aman |
| International Telecommunication Union (2015) | STD-02 | FINAL — OFFICIAL VERIFIED | ITU-R Recommendation BT.709-6 | Aman; Item 3.2 memuat koefisien 0,2126, 0,7152, dan 0,0722 untuk pembentukan sinyal luminansi |
| García et al. (2019) | COF-17 | FINAL — OFFICIAL VERIFIED | MDPI Applied Sciences | Aman |
| Hong et al. (2026) | COF-01 | FINAL — OFFICIAL VERIFIED | Elsevier / Current Research in Food Science | Aman |
| Bahy dan Rifai (2026) | COF-02 | FINAL — OFFICIAL VERIFIED | IJoICT official article/PDF | Aman |
| Samudra dan Rachmawati (2025) | COF-03 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE ICoDSA primary paper | Aman; dipakai untuk contoh kebingungan black vs partially black yang dikaitkan dengan visual similarity |
| Hebert dan Alamsyah (2026) | COF-04 | FINAL — OFFICIAL VERIFIED | INOVTEK Polbeng official page | Aman |
| Jundullah et al. (2026) | COF-05 | FINAL — OFFICIAL VERIFIED | Brilliance official page + primary PDF | Aman; klaim per kelas harus mengikuti paper |
| Gope et al. (2024) | COF-06 | FINAL — OFFICIAL VERIFIED | Nature / Scientific Reports | Aman |
| Kesiman et al. (2023) | COF-07 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | Primary conference PDF / IEEE record | Aman |
| Arwatchananukul et al. (2024) | COF-08 | FINAL — OFFICIAL VERIFIED | Elsevier / Smart Agricultural Technology | Aman |
| de Oliveira et al. (2016) | COF-10 | FINAL — OFFICIAL VERIFIED | Elsevier / Journal of Food Engineering | Aman; DOI resmi `10.1016/j.jfoodeng.2015.10.009` |
| Jiao et al. (2025) | COF-12 | FINAL — OFFICIAL VERIFIED | PLOS ONE | Aman |
| Hu et al. (2025) | COF-13 | FINAL — OFFICIAL VERIFIED | Elsevier / LWT | Aman |
| Tarekegn dan Debelee (2025) | COF-18 | FINAL — OFFICIAL VERIFIED | Tech Science Press / Journal on Artificial Intelligence | Aman; digunakan untuk konteks skala dataset deteksi primer |

## B. Object detection, fine-grained, preprocessing, teori, dan evaluasi

| Sitasi formal | Key | Status gate | Sumber authority | Keputusan |
|---|---|---|---|---|
| Ren et al. (2015) | DET-02 | FINAL — OFFICIAL VERIFIED | NeurIPS proceedings | Aman |
| Redmon et al. (2016) | DET-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Jocher et al. (2026) | DET-01 | FINAL — PRIMARY PREPRINT VERIFIED | arXiv primary preprint | Aman jika ditulis sebagai preprint |
| Wang et al. (2025) — RT-DETRv3 | DET-04 | FINAL — OFFICIAL VERIFIED | CVF WACV 2025 Open Access | Aman; digunakan hanya sebagai landasan evaluasi lintas arsitektur opsional |
| Feng et al. (2021) | DIAG-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Wu et al. (2020) | DIAG-02 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Jiang et al. (2018) | DIAG-03 | FINAL — OFFICIAL VERIFIED | ECCV/Springer proceedings | Aman |
| Xie et al. (2025) | FG-02 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE TCSVT publisher-format PDF | Aman |
| Xu et al. (2025) | FG-01 | FINAL — OFFICIAL VERIFIED | Elsevier / Neural Networks | Aman; §3.1.1 memuat DFT/iDFT/amplitudo/fase dan §3.3.3 memuat AFAB-2 Eq. (9)–(13) |
| Liu et al. (2022) | PRE-01 | FINAL — OFFICIAL VERIFIED | AAAI official proceedings | Aman |
| Qin et al. (2022) | PRE-02 | FINAL — OFFICIAL VERIFIED | CVF ACCV Open Access | Aman |
| Li et al. (2025) — FE-YOLO | PRE-03 | FINAL — OFFICIAL VERIFIED | Elsevier / Digital Signal Processing | Aman |
| Syauqi et al. (2025) | PRE-04 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE publisher PDF | Aman; sebut pipeline komposit, bukan CLAHE saja |
| Chen et al. (2024) — maize seed | PRE-05 | FINAL — OFFICIAL VERIFIED | Elsevier / Computers and Electronics in Agriculture | Aman |
| Yang dan Soatto (2020) | PRE-08 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Gonzalez dan Woods (2018) | THEORY-01 | OFFICIAL PUBLISHER METADATA VERIFIED | Pearson Global Edition, 4th ed. | Aman sebagai landasan teori umum; formula yang ditampilkan juga dapat dilacak langsung pada Xu et al. (2025) §3.1.1 sehingga nomor halaman buku tidak direka |
| Cao et al. (2019) | SPEC-01 | FINAL — OFFICIAL VERIFIED | Publisher official record | Aman |
| Zhang dan Tan (2003) | SPEC-02 | FINAL — OFFICIAL VERIFIED | Elsevier / Pattern Recognition | Aman |
| Chi et al. (2020) | FREQ-01 | FINAL — OFFICIAL VERIFIED | NeurIPS proceedings | Aman |
| Li et al. (2024) — FDADNet | FREQ-02 | FINAL — OFFICIAL VERIFIED | MDPI Processes | Aman |
| Chen et al. (2025) — Frequency Dynamic Convolution | FREQ-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |

`EVAL-01` / Lin et al. (2014) tetap dapat digunakan sebagai backend evaluasi COCO, tetapi **tidak mempunyai sitasi author–year pada naskah formal saat ini**, sehingga tidak masuk daftar pustaka proposal.

## C. Visualisasi aktivasi / XAI

| Sitasi formal | Key | Status gate | Sumber authority | Keputusan |
|---|---|---|---|---|
| Selvaraju et al. (2017) | XAI-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Aman |
| Muhammad dan Yeasin (2020) | XAI-03 | FINAL — PRIMARY PREPRINT VERIFIED | Primary Eigen-CAM paper | Aman jika status preprint ditulis transparan |

`XAI-02` / Grad-CAM++ tidak disitasi secara eksplisit pada BAB II atau BAB III formal dan tidak masuk daftar pustaka.

## D. Status daftar pustaka

Set formal saat ini berjumlah **38 sumber unik**.

Audit dua arah yang harus berlaku pada snapshot ini:

- `cited → bibliography`: **38/38**;
- `bibliography → cited`: **38/38**;
- cited source tanpa bibliography: **0**;
- bibliography entry tanpa sitasi formal: **0**.

Perubahan terbaru pada sinkronisasi ini:

- `COF-03` Samudra dan Rachmawati (2025) masuk ke set formal karena sekarang disitasi pada Tabel 2.1 sebagai bukti langsung kemiripan visual antarkelas pada deteksi cacat kopi;
- `EVAL-01` Lin et al. (2014) tetap backend-only pada snapshot formal saat ini.

Kondisi ini hanya berlaku untuk snapshot naskah saat ini. Penambahan, penghapusan, atau perubahan sitasi pada BAB I–III wajib memicu audit ulang metadata dan audit dua arah.