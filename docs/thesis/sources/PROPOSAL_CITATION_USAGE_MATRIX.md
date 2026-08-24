# Proposal Citation Usage Matrix

Purpose: memastikan setiap sitasi yang muncul pada artefak formal Bab I–III memiliki satu identitas sumber yang jelas dan satu status verifikasi resmi/primer yang transparan.

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
| Kesiman et al. (2023) | COF-07 | I, II | FINAL | Primary conference paper + IEEE landing identified. |
| Samudra dan Rachmawati (2025) | COF-03 | PRIMARY VERIFIED / PUBLISHER HTML PENDING | I, II | Full project PDF and DOI/page range are verified; IEEE publisher HTML remains crawler-blocked, so status is stated transparently. |
| Arwatchananukul et al. (2024) | COF-08 | II | FINAL | Elsevier official verified. |
| de Oliveira et al. (2016) | COF-10 | II | FINAL | Elsevier official verified; DOI workbook lama salah dan sudah dikoreksi. |
| Jiao et al. (2025) | COF-12 | I, II | FINAL | PLOS official verified. |
| Hu et al. (2025) | COF-13 | I, II | FINAL | Elsevier official verified. |
| Liu et al. (2022) | PRE-01 | I, II | FINAL | AAAI official verified. |
| Qin et al. (2022) | PRE-02 | I, II | FINAL | ACCV/CVF official verified. |
| Li et al. (2025) — FE-YOLO | PRE-03 | I, II | FINAL | Elsevier official verified. |
| Syauqi et al. (2025) | PRE-04 | I, II | FINAL — PRIMARY IEEE PDF VERIFIED | PDF yang diunduh dari IEEE Xplore memuat title, authors, conference, copyright/footer IEEE, dan DOI `10.1109/ICONS-IOT65216.2025.11211242`. |
| Chen et al. (2024) — maize seed | PRE-05 | I, II | FINAL | Elsevier official verified. |
| Yang dan Soatto (2020) | PRE-08 | II | FINAL | CVF official verified. |
| Cao et al. (2019) | SPEC-01 | I, II | FINAL | Publisher official verified. |
| Zhang dan Tan (2003) | SPEC-02 | I, II | FINAL | Elsevier official verified. |
| Xu et al. (2025) | FG-01 | I, II, III | FINAL | Elsevier official verified; parent AFAB/AFAB-2. |
| Xie et al. (2025) | FG-02 | II | FINAL — PRIMARY IEEE PDF VERIFIED | First page explicitly gives IEEE TCSVT 35(8), August 2025, p.8197, full authors, and DOI `10.1109/TCSVT.2025.3544741`. |
| Ren et al. (2015) | DET-02 | II | FINAL | NeurIPS official verified. |
| Redmon et al. (2016) | DET-03 | II | FINAL | CVF official verified. |
| Jocher et al. (2026) | DET-01 | II, III | FINAL — PRIMARY PREPRINT | Primary arXiv preprint verified; status preprint harus dipertahankan. |
| Feng et al. (2021) | DIAG-01 | II | FINAL | CVF official verified. |
| Wu et al. (2020) | DIAG-02 | II | FINAL | CVF official verified. |
| Jiang et al. (2018) | DIAG-03 | II | FINAL | ECCV/Springer official verified. |
| Chi et al. (2020) | FREQ-01 | II | FINAL | NeurIPS official verified. |
| Li et al. (2024) — FDADNet | FREQ-02 | II | FINAL | MDPI official verified. |
| Chen et al. (2025) — FDConv | FREQ-03 | II | FINAL | CVF official verified. |
| Lin et al. (2014) | EVAL-01 | III | FINAL | Springer official verified. |
| Selvaraju et al. (2017) | XAI-01 | II, III | FINAL | CVF official verified. |
| Chattopadhyay et al. (2018) | XAI-02 | II, III | PRIMARY PREPRINT VERIFIED / PUBLISHER HTML PENDING | Primary author preprint verifies method and author spelling; WACV DOI `10.1109/WACV.2018.00097` is corroborated. Prefer removing this optional citation from formal proposal unless needed. |
| Muhammad dan Yeasin (2020) | XAI-03 | II, III | FINAL — PRIMARY PREPRINT | Primary arXiv paper verifies title, authors, year, and Eigen-CAM mechanism. If bibliography uses this source, cite it explicitly as the primary preprint rather than inventing IEEE metadata. |

## Remaining hard gate

`DAFTAR_PUSTAKA.md` tetap belum boleh disebut final sebelum dua hal berikut selesai:

1. **COF-03**: bibliography metadata harus diambil dari primary PDF/DOI yang sudah dipetakan; jangan mengarang detail IEEE yang tidak terlihat pada source.
2. **XAI-02**: Grad-CAM++ bukan komponen wajib penelitian. Pilihan paling aman adalah menghapus sitasi eksplisitnya dari artefak formal dan menulis "Grad-CAM atau variannya". Jika tetap dipakai, bibliography harus menyatakan sumber primer yang benar dan tidak mengklaim publisher metadata yang belum diverifikasi.

`PRE-04`, `FG-02`, dan `XAI-03` bukan lagi blocker: bukti primer sudah tersedia.

## Known discrepancies / corrections

- `COF-10`: master workbook lama mencantumkan DOI `10.1016/j.jfoodeng.2015.10.030`, sedangkan ScienceDirect resmi menunjukkan DOI yang benar `10.1016/j.jfoodeng.2015.10.009`.
- `XAI-02`: primary author preprint mengeja penulis pertama **Aditya Chattopadhyay**. Jangan mengikuti variasi ejaan sekunder tanpa memeriksa source yang dipilih untuk bibliography.
- `PRE-05`: tahun artikel adalah 2024, sedangkan DOI resmi mengandung tahun 2023 (`10.1016/j.compag.2023.108475`). DOI tidak boleh diubah.
