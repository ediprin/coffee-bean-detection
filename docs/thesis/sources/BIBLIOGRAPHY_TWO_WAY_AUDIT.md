# Bibliography Two-Way Audit — Proposal

Status: **WORKING QUALITY GATE — NOT YET FINAL-SUBMISSION READY**

Dokumen ini memeriksa hubungan dua arah antara sitasi pada artefak formal proposal dan `docs/thesis/proposal/DAFTAR_PUSTAKA.md`.

## Prinsip

1. Sitasi yang muncul di BAB I–III harus mempunyai entri daftar pustaka.
2. Entri daftar pustaka hanya boleh dipertahankan jika benar-benar disitasi pada BAB I–III.
3. Metadata bibliografis tidak boleh diambil dari ingatan. Otoritas metadata mengikuti `OFFICIAL_CITATION_AUDIT.md`.
4. Jika publisher/primary PDF berbeda dengan workbook atau catatan lama, publisher/primary PDF menang.
5. Sumber yang hanya digunakan sebagai locator internal tidak otomatis masuk daftar pustaka.

## Sitasi formal yang dipertahankan

### Standar dan domain kopi

- Badan Standardisasi Nasional (2008) → STD-01
- García et al. (2019) → COF-17
- Hong et al. (2026) → COF-01
- Bahy dan Rifai (2026) → COF-02
- Samudra dan Rachmawati (2025) → COF-03
- Hebert dan Alamsyah (2026) → COF-04
- Jundullah et al. (2026) → COF-05
- Gope et al. (2024) → COF-06
- Kesiman et al. (2023) → COF-07
- Arwatchananukul et al. (2024) → COF-08
- de Oliveira et al. (2016) → COF-10
- Jiao et al. (2025) → COF-12
- Hu et al. (2025) → COF-13

### Detector, fine-grained, preprocessing, frequency, evaluasi

- Ren et al. (2015) → DET-02
- Redmon et al. (2016) → DET-03
- Jocher et al. (2026) → DET-01
- Feng et al. (2021) → DIAG-01
- Wu et al. (2020) → DIAG-02
- Jiang et al. (2018) → DIAG-03
- Xie et al. (2025) → FG-02
- Xu et al. (2025) → FG-01
- Liu et al. (2022) → PRE-01
- Qin et al. (2022) → PRE-02
- Li et al. (2025) / FE-YOLO → PRE-03
- Syauqi et al. (2025) → PRE-04
- Chen et al. (2024) / maize seed → PRE-05
- Yang dan Soatto (2020) → PRE-08
- Cao et al. (2019) → SPEC-01
- Zhang dan Tan (2003) → SPEC-02
- Chi et al. (2020) → FREQ-01
- Li et al. (2024) / FDADNet → FREQ-02
- Chen et al. (2025) / Frequency Dynamic Convolution → FREQ-03
- Lin et al. (2014) → EVAL-01

### Visualisasi aktivasi

- Selvaraju et al. (2017) → XAI-01
- Muhammad dan Yeasin (2020) → XAI-03

`XAI-02` / Grad-CAM++ **tidak dipertahankan sebagai sitasi eksplisit** pada artefak formal saat ini, sehingga tidak boleh masuk daftar pustaka hanya karena pernah dibahas pada drafting sebelumnya.

## Cross-check yang sudah dikunci

- COF-02 Bahy & Rifai: DOI `10.21108/ijoict.v12i1.10584` dikonfirmasi langsung pada PDF resmi IJoICT, halaman pertama.
- PRE-03 FE-YOLO: *Digital Signal Processing*, volume 166, article 105355, DOI `10.1016/j.dsp.2025.105355`, dikonfirmasi pada ScienceDirect.
- COF-13 Hu et al.: *LWT*, volume 235, article 118631, DOI `10.1016/j.lwt.2025.118631`, dikonfirmasi pada ScienceDirect.
- SPEC-02 Zhang & Tan: *Pattern Recognition*, 36(3), 657–664, DOI `10.1016/S0031-3203(02)00099-7`, dikonfirmasi pada ScienceDirect.
- SPEC-01 Cao et al.: *Journal of Spectroscopy* 2019, article 4970376, DOI `10.1155/2019/4970376`, author list dikonfirmasi pada publisher record.
- PRE-01 IA-YOLO: *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(2), 1792–1800, DOI `10.1609/aaai.v36i2.20072`, dikonfirmasi pada AAAI official article page.
- FREQ-01 Fast Fourier Convolution: NeurIPS 2020, volume 33, pp. 4479–4488; author Lu Chi, Borui Jiang, Yadong Mu dikonfirmasi melalui NeurIPS proceedings locator/primary paper record.
- FREQ-03 Frequency Dynamic Convolution: CVPR 2025, pp. 30178–30188; author Linwei Chen, Lin Gu, Liang Li, Chenggang Yan, Ying Fu dikonfirmasi pada CVF Open Access.

## Keputusan saat ini

`DAFTAR_PUSTAKA.md` sudah dapat dipakai sebagai **working formal bibliography**, tetapi belum diberi label final-submission-ready sampai pemeriksaan field-by-field seluruh entri selesai. Tidak boleh menambahkan DOI, pages, volume, issue, atau full author list baru tanpa sumber resmi/primer.
