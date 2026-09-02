# Official Citation Audit — Proposal

Status: **HARD GATE untuk sumber formal**

Dokumen ini mencatat sumber resmi atau primer yang boleh mendukung artefak formal pada `docs/thesis/proposal/`. Metadata aggregator hanya digunakan sebagai locator; jika tersedia sumber penerbit, proceedings, standard body, atau PDF primer, sumber tersebut menjadi authority.

## Aturan

- **FINAL — OFFICIAL VERIFIED**: identitas bibliografis telah diverifikasi pada sumber resmi.
- **FINAL — PRIMARY PUBLISHER PDF VERIFIED**: identitas dan isi relevan telah diperiksa pada PDF primer/publisher.
- **FINAL — PRIMARY PREPRINT VERIFIED**: sumber primer yang tersedia adalah preprint dan harus ditulis sebagai preprint.
- **OFFICIAL PUBLISHER METADATA VERIFIED**: metadata resmi cukup untuk sumber teori, tetapi nomor halaman tidak boleh direka.
- Klaim metodologis sensitif harus diperiksa dari full text primer, bukan metadata saja.
- DOI, halaman, volume, issue, indeks, dan quartile tidak boleh ditebak.

## A. Sumber formal yang lolos gate

| Key | Status | Authority | Batas penggunaan / catatan |
|---|---|---|---|
| STD-01 | FINAL — OFFICIAL VERIFIED | BSN | SNI 2907:2008 — Biji kopi. |
| STD-02 | FINAL — OFFICIAL VERIFIED | ITU-R official recommendation + PDF | Recommendation ITU-R BT.709-6 (06/2015), *Parameter values for the HDTV standards for production and international programme exchange*. Item 3.2 memberi koefisien luminansi 0.2126, 0.7152, 0.0722. |
| COF-17 | FINAL — OFFICIAL VERIFIED | MDPI Applied Sciences | García et al. (2019). |
| COF-01 | FINAL — OFFICIAL VERIFIED | Elsevier / CRFS | Hong et al. (2026), improved YOLOv10 coffee defect detection. |
| COF-02 | FINAL — OFFICIAL VERIFIED | IJoICT official article/PDF | Bahy & Rifai (2026), 20-category SNI-based detection. |
| COF-03 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE ICoDSA primary paper | Samudra & Rachmawati (2025); aman untuk contoh kebingungan black vs partially black yang dikaitkan penulis dengan visual similarity. |
| COF-04 | FINAL — OFFICIAL VERIFIED | INOVTEK official page | Hebert & Alamsyah (2026). |
| COF-05 | FINAL — OFFICIAL VERIFIED | Brilliance official article/PDF | Jundullah et al. (2026); klaim per kelas harus mengikuti tabel/diskusi primer. |
| COF-06 | FINAL — OFFICIAL VERIFIED | Nature / Scientific Reports | Gope et al. (2024). |
| COF-07 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE primary paper/record | Kesiman et al. (2023). |
| COF-08 | FINAL — OFFICIAL VERIFIED | Elsevier / Smart Agricultural Technology | Arwatchananukul et al. (2024). |
| COF-10 | FINAL — OFFICIAL VERIFIED | Elsevier / Journal of Food Engineering | de Oliveira et al. (2016); DOI resmi berakhiran `.009`. |
| COF-12 | FINAL — OFFICIAL VERIFIED | PLOS | Jiao et al. (2025). |
| COF-13 | FINAL — OFFICIAL VERIFIED | Elsevier / LWT | Hu et al. (2025). |
| COF-18 | FINAL — OFFICIAL VERIFIED | Tech Science Press official article/full text | Tarekegn & Debelee (2025); angka dataset harus mengikuti full text primer. |
| FG-01 | FINAL — OFFICIAL VERIFIED | Elsevier / Neural Networks + primary PDF | Xu et al. (2025). §3.1.1 Eq. (1)–(4) memuat DFT, amplitudo, fase, iDFT. §3.3.3 Eq. (9)–(13) memuat AFAB-2. Eq. (14) adalah CGFI, **bukan AFAB-2**. |
| FG-02 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE TCSVT primary paper | Xie et al. (2025). |
| PRE-01 | FINAL — OFFICIAL VERIFIED | AAAI proceedings | Liu et al. (2022), IA-YOLO. |
| PRE-02 | FINAL — OFFICIAL VERIFIED | CVF ACCV Open Access | Qin et al. (2022), DENet. |
| PRE-03 | FINAL — OFFICIAL VERIFIED | Elsevier / DSP | Li et al. (2025), FE-YOLO. |
| PRE-04 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE primary PDF | Syauqi et al. (2025); pipeline komposit, bukan CLAHE-only. |
| PRE-05 | FINAL — OFFICIAL VERIFIED | Elsevier / CEA | Chen et al. (2024), maize crack enhancement + YOLOv8. |
| PRE-08 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Yang & Soatto (2020), FDA. |
| THEORY-01 | OFFICIAL PUBLISHER METADATA VERIFIED | Pearson | Gonzalez & Woods (2018), *Digital Image Processing*, 4th Global Edition. Tidak menulis locator halaman yang tidak tersedia. |
| SPEC-01 | FINAL — OFFICIAL VERIFIED | Publisher official record | Cao et al. (2019), radial/angular spectrum. |
| SPEC-02 | FINAL — OFFICIAL VERIFIED | Elsevier | Zhang & Tan (2003). |
| FREQ-01 | FINAL — OFFICIAL VERIFIED | NeurIPS proceedings | Chi et al. (2020). |
| FREQ-02 | FINAL — OFFICIAL VERIFIED | MDPI Processes | Li et al. (2024), FDADNet. |
| FREQ-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Chen et al. (2025), Frequency Dynamic Convolution. |
| DET-01 | FINAL — PRIMARY PREPRINT VERIFIED | arXiv primary paper | Jocher et al. (2026), YOLO26; bibliography harus transparan sebagai preprint. |
| DET-02 | FINAL — OFFICIAL VERIFIED | NeurIPS proceedings | Ren et al. (2015). |
| DET-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Redmon et al. (2016). |
| DET-04 | FINAL — OFFICIAL VERIFIED | CVF WACV 2025 | Wang et al. (2025), RT-DETRv3; hanya dasar evaluasi lintas arsitektur opsional. |
| DIAG-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Feng et al. (2021), TOOD. |
| DIAG-02 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Wu et al. (2020). |
| DIAG-03 | FINAL — OFFICIAL VERIFIED | Springer ECCV | Jiang et al. (2018). |
| XAI-01 | FINAL — OFFICIAL VERIFIED | CVF/IEEE | Selvaraju et al. (2017), Grad-CAM. |
| XAI-03 | FINAL — PRIMARY PREPRINT VERIFIED | arXiv primary paper | Muhammad & Yeasin (2020), Eigen-CAM. |

## B. Backend-only yang tetap terverifikasi

Sumber berikut tetap boleh digunakan untuk audit/konteks internal tetapi tidak berada pada daftar pustaka formal selama tidak disitasi di BAB I–III:

| Key | Sumber | Status |
|---|---|---|
| EVAL-01 | Lin et al. (2014), Microsoft COCO | OFFICIAL VERIFIED |
| XAI-02 | Grad-CAM++ | PRIMARY VERIFIED / OPTIONAL |

## C. Guardrail metodologis aktif

1. **AFAB-2**: atribusi formal hanya untuk mekanisme yang benar-benar ada di Xu et al. (2025). Paper mendefinisikan angular density Eq. (9), entropy Eq. (10), adaptive threshold Eq. (11), hard normalized-density suppression Eq. (12), dan amplitude remodeling Eq. (13). Rekonstruksi menggunakan fase asli dijelaskan setelah persamaan tersebut. Eq. (14) tidak boleh disebut AFAB-2 karena berada pada bagian CGFI.
2. **Overlap patch**: Xu menyatakan penggunaan *large overlap* pada definisi metode dan kemudian menguji 0.5 serta 0.75. Proposal boleh memakai 50% sebagai keputusan adaptasi, bukan sebagai nilai universal dari source.
3. **Gamma**: Xu menguji 0–0.2 dan memakai 0.1 pada eksperimen mereka. Proposal menggunakan 0.1 sebagai konfigurasi referensi, bukan nilai yang diasumsikan optimal untuk kopi.
4. **BT.709**: Item 3.2 ITU-R BT.709-6 merupakan authority untuk koefisien luminansi C5.
5. **Fourier fundamentals**: Gonzalez & Woods tetap landasan teori umum; bentuk formula yang ditampilkan juga dapat diverifikasi langsung pada Xu et al. §3.1.1 tanpa mengarang nomor halaman buku.
6. **CLAHE literature**: Syauqi et al. menggunakan pipeline komposit, sehingga hasilnya tidak boleh dinarasikan sebagai efek CLAHE saja.
7. **Proposal temporal rule**: tidak ada hasil eksperimen internal tesis yang boleh masuk BAB I–III sebagai temuan.

## D. Status formal

Set sitasi formal saat ini: **38 sumber unik**. `CITATION_CROSSWALK.md`, `BIBLIOGRAPHY_METADATA_LOCK.md`, `DAFTAR_PUSTAKA.md`, dan `BIDIRECTIONAL_CITATION_AUDIT.md` harus mengikuti angka yang sama.