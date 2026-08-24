# Proposal Citation Usage Matrix

Purpose: memastikan setiap sitasi yang muncul pada artefak formal Bab I–III memiliki satu identitas sumber yang jelas dan satu status verifikasi resmi/primer yang transparan.

Status mengikuti `OFFICIAL_CITATION_AUDIT.md`.

| Sitasi dalam teks | Key | Bab | Status saat ini | Catatan |
|---|---|---|---|---|
| Badan Standardisasi Nasional (2008) | STD-01 | II, III | FINAL | Standar resmi BSN. |
| García et al. (2019) | COF-17 | I | FINAL | MDPI official verified. |
| Hong et al. (2026) | COF-01 | I, II | FINAL | Elsevier official verified. BAB III tidak lagi mengatribusikan analisis visual kepada Hong tanpa bukti metode spesifik. |
| Gope et al. (2024) | COF-06 | I, II | FINAL | Nature official verified. |
| Bahy dan Rifai (2026) | COF-02 | I, II | FINAL | Official journal article/PDF verified. |
| Jundullah et al. (2026) | COF-05 | I, II | FINAL | Official journal page verified. |
| Hebert dan Alamsyah (2026) | COF-04 | I, II | FINAL | Official journal page verified. |
| Kesiman et al. (2023) | COF-07 | I, II | FINAL | Primary conference paper + IEEE record identified. |
| Samudra dan Rachmawati (2025) | COF-03 | I, II | PRIMARY VERIFIED / PUBLISHER METADATA PENDING | Full text dan DOI/page range tersedia; jangan menambah metadata publisher yang belum terlihat. |
| Arwatchananukul et al. (2024) | COF-08 | II | FINAL | Elsevier official verified. |
| de Oliveira et al. (2016) | COF-10 | II | FINAL | Elsevier official verified; DOI workbook lama salah dan sudah dikoreksi. |
| Jiao et al. (2025) | COF-12 | I, II | FINAL | PLOS official verified. |
| Hu et al. (2025) | COF-13 | I, II | FINAL | Elsevier official verified. |
| Liu et al. (2022) | PRE-01 | I, II | FINAL | AAAI official verified. |
| Qin et al. (2022) | PRE-02 | I, II | FINAL | ACCV/CVF official verified. |
| Li et al. (2025) — FE-YOLO | PRE-03 | I, II | FINAL | Elsevier official verified. |
| Syauqi et al. (2025) | PRE-04 | I, II | FINAL — PRIMARY IEEE PDF VERIFIED | IEEE Xplore-downloaded PDF memuat title, authors, conference, dan DOI `10.1109/ICONS-IOT65216.2025.11211242`. |
| Chen et al. (2024) — maize seed | PRE-05 | I, II | FINAL | Elsevier official verified. |
| Yang dan Soatto (2020) | PRE-08 | II | FINAL | CVF official verified. |
| Cao et al. (2019) | SPEC-01 | I, II | FINAL | Publisher official verified. |
| Zhang dan Tan (2003) | SPEC-02 | I, II | FINAL | Elsevier official verified. |
| Xu et al. (2025) | FG-01 | I, II, III | FINAL | Elsevier official verified; parent AFAB/AFAB-2. |
| Xie et al. (2025) | FG-02 | II | FINAL — PRIMARY IEEE PDF VERIFIED | IEEE TCSVT PDF explicitly gives 35(8), 8197–8208 and DOI `10.1109/TCSVT.2025.3544741`. |
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
| Muhammad dan Yeasin (2020) | XAI-03 | II, III | FINAL — PRIMARY PREPRINT | Primary Eigen-CAM paper verified; bibliography harus menyatakan source primer yang benar. |

## Sumber backend yang tidak lagi wajib masuk daftar pustaka formal

`XAI-02` (Grad-CAM++) tetap disimpan pada evidence backend, tetapi sitasi eksplisitnya telah dihapus dari BAB II dan BAB III. Naskah formal sekarang menyebut **Grad-CAM atau varian CAM lain** tanpa menggantungkan argumen pada Grad-CAM++.

## Remaining hard gate

Sebelum `DAFTAR_PUSTAKA.md` disebut final, blocker bibliografis utama yang tersisa adalah `COF-03` (Samudra & Rachmawati, 2025): entri harus dibentuk hanya dari metadata yang benar-benar terverifikasi pada primary paper/DOI.

Setelah itu tetap diperlukan audit dua arah seluruh artefak: **cited → bibliography** dan **bibliography → cited**.

## Known discrepancies / corrections

- `COF-10`: master workbook lama mencantumkan DOI `10.1016/j.jfoodeng.2015.10.030`, sedangkan ScienceDirect resmi menunjukkan DOI yang benar `10.1016/j.jfoodeng.2015.10.009`.
- `PRE-05`: tahun artikel 2024, DOI resmi `10.1016/j.compag.2023.108475`; DOI tidak boleh diubah.
- `XAI-02`: primary author preprint mengeja penulis pertama **Aditya Chattopadhyay**; variasi ejaan pada metadata sekunder tidak digunakan dalam naskah formal karena sitasinya telah dikeluarkan.
