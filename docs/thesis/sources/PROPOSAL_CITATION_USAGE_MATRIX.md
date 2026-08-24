# Proposal Citation Usage Matrix

Purpose: memastikan setiap sitasi yang muncul pada artefak formal Bab I–III memiliki satu identitas sumber yang jelas dan satu status verifikasi resmi.

Status mengikuti `OFFICIAL_CITATION_AUDIT.md`.

| Sitasi dalam teks | Key | Bab | Status saat ini | Catatan |
|---|---|---|---|---|
| Badan Standardisasi Nasional (2008) | STD-01 | II, III | FINAL | Standar resmi BSN. |
| García et al. (2019) | COF-17 | I | FINAL | MDPI official verified. |
| Hong et al. (2026) | COF-01 | I, II, III | FINAL | Elsevier official verified. |
| Gope et al. (2024) | COF-06 | I, II | FINAL | Nature official verified. |
| Bahy dan Rifai (2026) | COF-02 | I, II | FINAL | Official journal article/PDF verified. |
| Jundullah et al. (2026) | COF-05 | I, II | FINAL | Official journal page verified. |
| Hebert dan Alamsyah (2026) | COF-04 | I, II | FINAL | Official journal page verified. |
| Kesiman et al. (2023) | COF-07 | I, II | FINAL | Primary conference paper + IEEE official landing. |
| Samudra dan Rachmawati (2025) | COF-03 | I, II | CORROBORATED | Belum masuk bibliography final sampai IEEE metadata ditutup. |
| Arwatchananukul et al. (2024) | COF-08 | II | FINAL | Elsevier official verified. |
| de Oliveira et al. (2016) | COF-10 | II | FINAL | Elsevier official verified; DOI workbook lama salah. |
| Jiao et al. (2025) | COF-12 | I, II | FINAL | PLOS official verified. |
| Hu et al. (2025) | COF-13 | I, II | FINAL | Elsevier official verified. |
| Liu et al. (2022) | PRE-01 | I, II | FINAL | AAAI official verified. |
| Qin et al. (2022) | PRE-02 | I, II | FINAL | ACCV/CVF official verified. |
| Li et al. (2025) — FE-YOLO | PRE-03 | I, II | FINAL | Elsevier official verified. |
| Syauqi et al. (2025) | PRE-04 | I, II | CORROBORATED | IEEE official bibliographic record belum ditutup. |
| Chen et al. (2024) — maize seed | PRE-05 | I, II | FINAL | Elsevier official verified. |
| Yang dan Soatto (2020) | PRE-08 | II | FINAL | CVF official verified. |
| Cao et al. (2019) | SPEC-01 | I, II | FINAL | Wiley official verified. |
| Zhang dan Tan (2003) | SPEC-02 | I, II | FINAL | Elsevier official verified. |
| Xu et al. (2025) | FG-01 | I, II, III | FINAL | Elsevier official verified; parent AFAB/AFAB-2. |
| Xie et al. (2025) | FG-02 | II | CORROBORATED | DOI/volume/pages kuat; IEEE official record belum ditutup. |
| Ren et al. (2015) | DET-02 | II | FINAL | NeurIPS official verified. |
| Redmon et al. (2016) | DET-03 | II | FINAL | CVF official verified. |
| Jocher et al. (2026) | DET-01 | II, III | FINAL | Primary arXiv preprint verified; status preprint harus dipertahankan. |
| Feng et al. (2021) | DIAG-01 | II | FINAL | CVF official verified. |
| Wu et al. (2020) | DIAG-02 | II | FINAL | CVF official verified. |
| Jiang et al. (2018) | DIAG-03 | II | FINAL | ECCV/Springer official verified. |
| Chi et al. (2020) | FREQ-01 | II | FINAL | NeurIPS official verified. |
| Li et al. (2024) — FDADNet | FREQ-02 | II | FINAL | MDPI official verified. |
| Chen et al. (2025) — FDConv | FREQ-03 | II | FINAL | CVF official verified. |
| Lin et al. (2014) | EVAL-01 | III | FINAL | Springer official verified. |
| Selvaraju et al. (2017) | XAI-01 | II, III | FINAL | CVF official verified. |
| Chattopadhyay et al. (2018) | XAI-02 | II, III | CORROBORATED | Primary arXiv + DOI kuat; IEEE record masih hard gate. |
| Muhammad dan Yeasin (2020) | XAI-03 | II, III | CORROBORATED | Primary arXiv + DOI kuat; IEEE record masih hard gate. |

## Current hard blockers

Sebelum `DAFTAR_PUSTAKA.md` dinyatakan final, minimal sumber berikut harus ditutup atau dikeluarkan dari naskah formal:

1. `COF-03` — Samudra & Rachmawati (2025)
2. `FG-02` — Xie et al. (2025)
3. `PRE-04` — Syauqi et al. (2025)
4. `XAI-02` — Grad-CAM++
5. `XAI-03` — Eigen-CAM

Jika salah satu tidak dapat diverifikasi dari official record, jangan menebak metadata. Opsi yang sah adalah mempertahankan status belum-final atau menghapus/ganti sitasi tersebut dari artefak formal.

## Known discrepancy

`COF-10` pada master workbook lama mencantumkan DOI `10.1016/j.jfoodeng.2015.10.030`, sedangkan ScienceDirect resmi menunjukkan DOI `10.1016/j.jfoodeng.2015.10.009`. Nilai resmi harus digunakan.
