# Bidirectional Citation Audit — Proposal

Status: **CURRENT FORMAL CITATION GATE**

Audit ini menghubungkan tiga lapisan:

1. sitasi author–year yang benar-benar dipertahankan pada artefak formal `docs/thesis/proposal/BAB_I–III`;
2. `CITATION_CROSSWALK.md` sebagai pemetaan ke canonical key;
3. `docs/thesis/proposal/DAFTAR_PUSTAKA.md` sebagai bibliography proposal-facing.

Jika manuscript berubah, audit ini harus dijalankan ulang.

## Hasil audit dua arah

- Unique cited-source set: **38 sumber**.
- Entri pada `DAFTAR_PUSTAKA.md`: **38 sumber**.
- `cited → bibliography`: **38/38 terpetakan**.
- `bibliography → cited`: **38/38 terpetakan**.
- Uncited bibliography entries: **0**.
- Cited sources tanpa bibliography entry: **0**.

Perubahan terbaru:

- `COF-03` Samudra & Rachmawati (2025) masuk ke set formal karena disitasi pada Tabel 2.1 sebagai bukti langsung kebingungan antarkelas kopi yang dikaitkan dengan visual similarity;
- `STD-02` International Telecommunication Union (2015), `COF-18` Tarekegn & Debelee (2025), dan `DET-04` Wang et al. (2025) tetap berada pada set formal;
- `EVAL-01` Lin et al. (2014) dan `XAI-02` Grad-CAM++ tetap backend-only.

## Cross-check set formal

| Key | Sitasi formal | Bibliography | Source gate |
|---|---|---|---|
| STD-01 | Badan Standardisasi Nasional (2008) | Ada | Official BSN |
| STD-02 | International Telecommunication Union (2015) | Ada | Official ITU-R BT.709-6 |
| COF-17 | García et al. (2019) | Ada | Official publisher |
| COF-01 | Hong et al. (2026) | Ada | Official publisher + primary full text |
| COF-02 | Bahy dan Rifai (2026) | Ada | Official journal + primary full text |
| COF-03 | Samudra dan Rachmawati (2025) | Ada | Primary IEEE conference paper/full text |
| COF-04 | Hebert dan Alamsyah (2026) | Ada | Official journal + primary full text |
| COF-05 | Jundullah et al. (2026) | Ada | Official journal + primary full text |
| COF-06 | Gope et al. (2024) | Ada | Official publisher |
| COF-07 | Kesiman et al. (2023) | Ada | Primary IEEE paper/record |
| COF-08 | Arwatchananukul et al. (2024) | Ada | Official publisher |
| COF-10 | de Oliveira et al. (2016) | Ada | Official publisher |
| COF-12 | Jiao et al. (2025) | Ada | Official publisher |
| COF-13 | Hu et al. (2025) | Ada | Official publisher + primary full text |
| COF-18 | Tarekegn dan Debelee (2025) | Ada | Tech Science Press official article/full text |
| DET-02 | Ren et al. (2015) | Ada | NeurIPS official proceedings |
| DET-03 | Redmon et al. (2016) | Ada | CVF primary proceedings |
| DET-01 | Jocher et al. (2026) | Ada | Primary arXiv preprint |
| DET-04 | Wang et al. (2025) | Ada | CVF WACV 2025 Open Access |
| DIAG-01 | Feng et al. (2021) | Ada | CVF primary proceedings |
| DIAG-02 | Wu et al. (2020) | Ada | CVF primary proceedings |
| DIAG-03 | Jiang et al. (2018) | Ada | Springer official chapter |
| FG-02 | Xie et al. (2025) | Ada | Primary IEEE publisher-format paper |
| FG-01 | Xu et al. (2025) | Ada | Official publisher + full-text formula audit |
| PRE-01 | Liu et al. (2022) | Ada | AAAI official proceedings |
| PRE-02 | Qin et al. (2022) | Ada | CVF ACCV Open Access |
| PRE-03 | Li et al. (2025) | Ada | Official publisher |
| PRE-04 | Syauqi et al. (2025) | Ada | Primary IEEE paper, pp. 18–23 |
| PRE-05 | Chen et al. (2024) | Ada | Official publisher |
| PRE-08 | Yang dan Soatto (2020) | Ada | CVF Open Access |
| THEORY-01 | Gonzalez dan Woods (2018) | Ada | Pearson official publisher metadata |
| SPEC-01 | Cao et al. (2019) | Ada | Official publisher |
| SPEC-02 | Zhang dan Tan (2003) | Ada | Official publisher |
| FREQ-01 | Chi et al. (2020) | Ada | NeurIPS official proceedings |
| FREQ-02 | Li et al. (2024) | Ada | Official publisher |
| FREQ-03 | Chen et al. (2025) | Ada | CVF primary proceedings |
| XAI-01 | Selvaraju et al. (2017) | Ada | CVF/IEEE official record |
| XAI-03 | Muhammad dan Yeasin (2020) | Ada | Primary Eigen-CAM preprint |

## Backend-only yang sengaja tidak dihitung

| Key | Sumber | Alasan tidak masuk bibliography formal saat ini |
|---|---|---|
| EVAL-01 | Lin et al. (2014) | Sumber evaluasi backend, tetapi tidak disitasi formal pada naskah |
| XAI-02 | Grad-CAM++ | Tidak lagi disitasi eksplisit |

## Scholarly-quality guardrail

Audit bibliografis dua arah **bukan** bukti bahwa setiap klaim telah diverifikasi sampai halaman dan persamaan. Guardrail yang berlaku:

- AFAB-2 mengikuti Xu et al. (2025): distribusi angular Eq. (9), entropi/ambang/pembobotan Eq. (10)–(13); Eq. (14) pada paper adalah bagian CGFI dan tidak boleh diatribusikan kepada AFAB-2;
- Samudra & Rachmawati (2025) hanya digunakan untuk bukti tiga kelas dan kebingungan black vs partially black yang dikaitkan dengan visual similarity; jangan digeneralisasi ke seluruh taksonomi kopi;
- Syauqi et al. (2025) tetap disebut sebagai pipeline komposit, bukan bukti CLAHE-only;
- angka dataset Tarekegn & Debelee mengikuti full text primer;
- RT-DETRv3 dipakai sebagai dasar evaluasi tambahan, bukan klaim superioritas arsitektur;
- koefisien luminansi pada C5 mengikuti ITU-R BT.709-6 Item 3.2;
- Gonzalez & Woods tetap menjadi landasan teori umum, sementara formula DFT/iDFT/amplitudo/fase juga dapat dilacak pada Xu et al. (2025) §3.1.1 tanpa mengarang nomor halaman buku.

## Hard Rule

```text
MANUSCRIPT CHANGE
    ↓
CITATION_CROSSWALK
    ↓
OFFICIAL / PRIMARY SOURCE CHECK
    ↓
CLAIM-LEVEL SOURCE CHECK (untuk klaim sensitif)
    ↓
BIBLIOGRAPHY_METADATA_LOCK
    ↓
DAFTAR_PUSTAKA
    ↓
BIDIRECTIONAL AUDIT
```

Tidak ada metadata baru yang boleh masuk melalui inferensi atau ingatan.