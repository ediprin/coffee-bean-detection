# Bidirectional Citation Audit — Proposal

Status: **CURRENT FORMAL CITATION GATE**

Audit ini menghubungkan tiga lapisan:

1. sitasi author–year yang benar-benar dipertahankan pada artefak formal BAB I–III;
2. `CITATION_CROSSWALK.md` sebagai pemetaan ke canonical key;
3. `DAFTAR_PUSTAKA.md` sebagai bibliography proposal-facing.

Metadata bibliografis hanya boleh berasal dari `BIBLIOGRAPHY_METADATA_LOCK.md` beserta koreksi authority pada `BIBLIOGRAPHY_METADATA_LOCK_CORRECTIONS.md`. Jika manuscript berubah, audit ini harus dijalankan ulang.

## Hasil audit dua arah

- Unique cited-source set: **35 sumber**.
- Entri pada `DAFTAR_PUSTAKA.md`: **35 sumber**.
- `cited → bibliography`: **35/35 terpetakan**.
- `bibliography → cited`: **35/35 terpetakan**.
- Uncited bibliography entries: **0**.
- Cited sources tanpa bibliography entry: **0**.
- `XAI-02` / Grad-CAM++: **tidak disitasi eksplisit dan tidak masuk bibliography**.

## Cross-check set formal

| Key | Sitasi author–year formal | Bibliography | Source gate |
|---|---|---|---|
| STD-01 | Badan Standardisasi Nasional (2008) | Ada | Official BSN |
| COF-17 | García et al. (2019) | Ada | Official publisher |
| COF-01 | Hong et al. (2026) | Ada | Official publisher |
| COF-02 | Bahy dan Rifai (2026) | Ada | Official journal |
| COF-03 | Samudra dan Rachmawati (2025) | Ada | Primary IEEE paper |
| COF-04 | Hebert dan Alamsyah (2026) | Ada | Official journal |
| COF-05 | Jundullah et al. (2026) | Ada | Official journal |
| COF-06 | Gope et al. (2024) | Ada | Official publisher |
| COF-07 | Kesiman et al. (2023) | Ada | Primary IEEE paper/record |
| COF-08 | Arwatchananukul et al. (2024) | Ada | Official publisher |
| COF-10 | de Oliveira et al. (2016) | Ada | Official publisher |
| COF-12 | Jiao et al. (2025) | Ada | Official publisher |
| COF-13 | Hu et al. (2025) | Ada | Official publisher |
| DET-02 | Ren et al. (2015) | Ada | Official NeurIPS proceedings |
| DET-03 | Redmon et al. (2016) | Ada | CVF primary proceedings page/PDF |
| DET-01 | Jocher et al. (2026) | Ada | Primary arXiv preprint |
| DIAG-01 | Feng et al. (2021) | Ada | CVF primary proceedings page/PDF |
| DIAG-02 | Wu et al. (2020) | Ada | CVF primary proceedings page/PDF |
| DIAG-03 | Jiang et al. (2018) | Ada | Springer official chapter |
| FG-02 | Xie et al. (2025) | Ada | Primary IEEE publisher-format paper |
| FG-01 | Xu et al. (2025) | Ada | Official publisher |
| PRE-01 | Liu et al. (2022) | Ada | AAAI official proceedings |
| PRE-02 | Qin et al. (2022) | Ada | CVF ACCV primary proceedings convention |
| PRE-03 | Li et al. (2025) | Ada | Official publisher |
| PRE-04 | Syauqi et al. (2025) | Ada | Primary IEEE paper, pp. 18–23 |
| PRE-05 | Chen et al. (2024) | Ada | Official publisher |
| PRE-08 | Yang dan Soatto (2020) | Ada | CVF primary proceedings page/PDF |
| SPEC-01 | Cao et al. (2019) | Ada | Official publisher |
| SPEC-02 | Zhang dan Tan (2003) | Ada | Official publisher |
| FREQ-01 | Chi et al. (2020) | Ada | Official NeurIPS proceedings |
| FREQ-02 | Li et al. (2024) | Ada | Official publisher |
| FREQ-03 | Chen et al. (2025) | Ada | CVF primary proceedings page/PDF |
| EVAL-01 | Lin et al. (2014) | Ada | Springer official chapter |
| XAI-01 | Selvaraju et al. (2017) | Ada | CVF/IEEE official record |
| XAI-03 | Muhammad dan Yeasin (2020) | Ada | Primary Eigen-CAM preprint |

## Corrections closed during audit

### COF-03

Samudra dan Rachmawati (2025) masih muncul pada manuscript formal dan karena itu **bukan backend-only**. Entri bibliography dipertahankan:

- Melyna Nura Samudra
- Ema Rachmawati
- *Deep Learning-Based Defect Detection in Arabica Green Coffee Beans Using LSKNet*
- ICoDSA 2025
- pp. 692–697
- DOI `10.1109/ICoDSA67155.2025.11157423`

### PRE-04

Primary IEEE PDF mengunci author order dan pagination:

- Faturrahman Syauqi
- Maulisa Oktiana
- Kahlil Muchtar
- Al Bahri
- Safrizal Razali
- pp. 18–23
- DOI `10.1109/ICONS-IOT65216.2025.11211242`

Author order dari primary paper menang terhadap metadata sekunder yang berbeda.

### XAI-03

Eigen-CAM ditulis sebagai primary preprint agar status sumber transparan. Tidak boleh disamarkan sebagai conference publication sampai IEEE proceedings metadata dipilih dan dikunci sebagai convention bibliography.

## Remaining scholarly-quality checks

Audit dua arah bibliography sudah bersih, tetapi **citation completeness tidak sama dengan claim correctness**. Sebelum proposal dinyatakan citation-ready, tetap diperlukan:

1. page/section-level verification untuk klaim metodologis yang sensitif, terutama Xu et al. (2025), Hong et al. (2026), Syauqi et al. (2025), dan paper coffee per-class yang menjadi dasar fine-grained argument;
2. source fundamental untuk definisi DFT/FFT/amplitude/phase bila rumus fundamental dipertahankan di BAB II;
3. official COCO evaluation specification jika BAB III ingin mengklaim detail implementasi AP@[0.50:0.95], bukan sekadar konteks benchmark COCO;
4. audit ulang setiap kali sitasi ditambah, dihapus, atau tahunnya berubah.

## Hard rule

```text
MANUSCRIPT CHANGE
    ↓
CITATION_CROSSWALK
    ↓
OFFICIAL / PRIMARY SOURCE CHECK
    ↓
BIBLIOGRAPHY_METADATA_LOCK
    ↓
DAFTAR_PUSTAKA
    ↓
BIDIRECTIONAL AUDIT
```

Tidak ada metadata baru yang boleh masuk melalui inferensi atau ingatan.