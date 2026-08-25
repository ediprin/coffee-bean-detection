# Bidirectional Citation Audit — Proposal

Status: **CURRENT FORMAL CITATION GATE**

Audit ini menghubungkan tiga lapisan:

1. sitasi author–year yang benar-benar dipertahankan pada artefak formal BAB I–III;
2. `CITATION_CROSSWALK.md` sebagai pemetaan ke canonical key;
3. `DAFTAR_PUSTAKA.md` sebagai bibliography proposal-facing.

Metadata bibliografis hanya boleh berasal dari source resmi/primer yang telah dikunci pada audit repository. Jika manuscript berubah, audit ini harus dijalankan ulang.

## Hasil audit dua arah

- Unique cited-source set: **36 sumber**.
- Entri pada `DAFTAR_PUSTAKA.md`: **36 sumber**.
- `cited → bibliography`: **36/36 terpetakan**.
- `bibliography → cited`: **36/36 terpetakan**.
- Uncited bibliography entries: **0**.
- Cited sources tanpa bibliography entry: **0**.
- `XAI-02` / Grad-CAM++: **tidak disitasi eksplisit dan tidak masuk bibliography**.
- `EVAL-02` / official COCOeval: **sudah diverifikasi sebagai source implementasi**, tetapi belum menjadi sitasi author–year terpisah sehingga belum dihitung sebagai bibliography entry.

## Cross-check set formal

| Key | Sitasi author–year formal | Bibliography | Source gate |
|---|---|---|---|
| STD-01 | Badan Standardisasi Nasional (2008) | Ada | Official BSN |
| COF-17 | García et al. (2019) | Ada | Official publisher |
| COF-01 | Hong et al. (2026) | Ada | Official publisher + primary full text |
| COF-02 | Bahy dan Rifai (2026) | Ada | Official journal + primary full text |
| COF-03 | Samudra dan Rachmawati (2025) | Ada | Primary IEEE paper |
| COF-04 | Hebert dan Alamsyah (2026) | Ada | Official journal + primary full text |
| COF-05 | Jundullah et al. (2026) | Ada | Official journal + Table 3 p.319 / Discussion p.320 checked |
| COF-06 | Gope et al. (2024) | Ada | Official publisher |
| COF-07 | Kesiman et al. (2023) | Ada | Primary IEEE paper/record + p.79–80 checked |
| COF-08 | Arwatchananukul et al. (2024) | Ada | Official publisher |
| COF-10 | de Oliveira et al. (2016) | Ada | Official publisher |
| COF-12 | Jiao et al. (2025) | Ada | Official publisher |
| COF-13 | Hu et al. (2025) | Ada | Official publisher + primary full text |
| DET-02 | Ren et al. (2015) | Ada | Official NeurIPS proceedings |
| DET-03 | Redmon et al. (2016) | Ada | CVF primary proceedings page/PDF |
| DET-01 | Jocher et al. (2026) | Ada | Primary arXiv preprint |
| DIAG-01 | Feng et al. (2021) | Ada | CVF primary proceedings page/PDF |
| DIAG-02 | Wu et al. (2020) | Ada | CVF primary proceedings page/PDF |
| DIAG-03 | Jiang et al. (2018) | Ada | Springer official chapter |
| FG-02 | Xie et al. (2025) | Ada | Primary IEEE publisher-format paper |
| FG-01 | Xu et al. (2025) | Ada | Official publisher + AFAB-2 formula-level audit |
| PRE-01 | Liu et al. (2022) | Ada | AAAI official proceedings |
| PRE-02 | Qin et al. (2022) | Ada | CVF ACCV primary proceedings convention |
| PRE-03 | Li et al. (2025) | Ada | Official publisher |
| PRE-04 | Syauqi et al. (2025) | Ada | Primary IEEE paper, pp. 18–23 |
| PRE-05 | Chen et al. (2024) | Ada | Official publisher |
| PRE-08 | Yang dan Soatto (2020) | Ada | CVF primary proceedings page/PDF |
| THEORY-01 | Gonzalez dan Woods (2018) | Ada | Pearson official 4th Global Edition metadata; exact formula-page project locator still open |
| SPEC-01 | Cao et al. (2019) | Ada | Official publisher |
| SPEC-02 | Zhang dan Tan (2003) | Ada | Official publisher |
| FREQ-01 | Chi et al. (2020) | Ada | Official NeurIPS proceedings |
| FREQ-02 | Li et al. (2024) | Ada | Official publisher |
| FREQ-03 | Chen et al. (2025) | Ada | CVF primary proceedings page/PDF |
| EVAL-01 | Lin et al. (2014) | Ada | Springer official chapter |
| XAI-01 | Selvaraju et al. (2017) | Ada | CVF/IEEE official record |
| XAI-03 | Muhammad dan Yeasin (2020) | Ada | Primary Eigen-CAM preprint |

## Corrections closed during audit

### COF-05 / Jundullah et al. (2026)

Claim-level gate yang sebelumnya pending sudah ditutup dari primary PDF:

- Table 3, p. 319: metrik per kelas;
- p. 320: confusion matrix dan Discussion;
- paper secara langsung membahas kesulitan pada kategori dengan kemiripan visual, termasuk varian biji hitam.

Temuan tersebut boleh mendukung argumen fine-grained pada dataset mereka, tetapi tidak boleh diubah menjadi klaim bahwa semua YOLO memiliki bottleneck klasifikasi atau bahwa frequency preprocessing pasti efektif.

### THEORY-01 / Gonzalez & Woods

BAB II sekarang menggunakan `(Gonzalez & Woods, 2018)` sebagai landasan fundamental DFT/FFT/amplitudo/fase.

Metadata bibliography dikunci sebagai:

- Rafael C. Gonzalez & Richard E. Woods;
- *Digital Image Processing*;
- 4th edition, Global Edition;
- Pearson;
- 2018;
- ISBN `9781292223049` pada metadata lock.

Nomor halaman formula **tidak ditulis** karena selected official full-text pages belum tersimpan sebagai project source.

### COCO evaluation detail

Official `cocodataset/cocoapi` telah diperiksa. `pycocotools/cocoeval.py` secara eksplisit mendefinisikan `iouThrs` pada 0.50 sampai 0.95 dengan langkah 0.05. Ini menutup source gap untuk rentang threshold COCO-style evaluation pada level implementasi.

`EVAL-02` tidak otomatis dimasukkan sebagai bibliography entry terpisah karena manuscript saat ini belum mempunyai sitasi formal terpisah untuk software tersebut.

### COF-03

Samudra dan Rachmawati (2025) masih muncul pada manuscript formal dan karena itu **bukan backend-only**. Entri bibliography dipertahankan sesuai primary IEEE record.

### PRE-04

Primary IEEE PDF mengunci author order, pages 18–23, dan DOI. Deskripsi metode harus tetap menyebut preprocessing komposit, bukan CLAHE saja.

### XAI-03

Eigen-CAM ditulis sebagai primary preprint agar status sumber transparan. Tidak boleh disamarkan sebagai conference publication sampai convention IEEE proceedings benar-benar dikunci.

## Remaining scholarly-quality check

Audit bibliography dan sebagian besar claim-level gate inti sekarang bersih. Satu gap yang sengaja **tidak ditutup dengan tebakan** adalah:

- selected page/section locator dari *Digital Image Processing* untuk formula DFT/iDFT dan amplitude/phase. Metadata Pearson dan citation source sudah resmi, tetapi page-level project-source verification masih menunggu akses halaman buku yang dipakai.

Ini tidak membatalkan sumber bibliografis Gonzalez & Woods, tetapi proposal belum boleh disebut **page-level citation audit complete** untuk rumus fundamental tersebut.

## Hard rule

```text
MANUSCRIPT CHANGE
    ↓
CITATION_CROSSWALK
    ↓
OFFICIAL / PRIMARY SOURCE CHECK
    ↓
CLAIM-LEVEL SOURCE CHECK
    ↓
BIBLIOGRAPHY METADATA LOCK
    ↓
DAFTAR_PUSTAKA
    ↓
BIDIRECTIONAL AUDIT
```

Tidak ada metadata baru yang boleh masuk melalui inferensi atau ingatan.