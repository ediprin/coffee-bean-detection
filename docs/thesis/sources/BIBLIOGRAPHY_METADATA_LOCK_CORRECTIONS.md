# Bibliography Metadata Lock — Corrections

Status: **AUTHORITATIVE OVERRIDE** untuk `BIBLIOGRAPHY_METADATA_LOCK.md` sampai file induk dikonsolidasikan.

Dokumen ini dibuat karena audit silang menemukan dua masalah pada readiness section `BIBLIOGRAPHY_METADATA_LOCK.md`. Jika terjadi konflik, koreksi di file ini menang.

## 1. COF-03 tidak boleh dikeluarkan dari cited-source set

`COF-03` masih disitasi pada artefak formal dan tercatat pada `CITATION_CROSSWALK.md`. Karena itu, pernyataan lama yang menempatkan `COF-03` sebagai backend-only adalah salah.

### COF-03 — LOCKED
- Authors: Melyna Nura Samudra; Ema Rachmawati
- Year: 2025
- Title: *Deep Learning-Based Defect Detection in Arabica Green Coffee Beans Using LSKNet*
- Proceedings: *2025 International Conference on Data Science and Its Applications (ICoDSA)*
- Pages: 692–697
- DOI: `10.1109/ICoDSA67155.2025.11157423`
- Authority: primary IEEE publisher-format paper / IEEE Xplore record already identified in project evidence.
- Decision: **WAJIB masuk `DAFTAR_PUSTAKA.md` selama sitasi Samudra dan Rachmawati (2025) tetap ada di BAB I–III.**

## 2. PRE-04 sekarang tertutup dari primary IEEE PDF

File Library project berisi primary PDF IEEE:

`Syauqi et al. - 2025 - Edge AI-Based Defect Detection in White Pepper (Piper Nigrum L.) Using CLAHE-Based Preprocessing and.pdf`

Primary PDF mengunci urutan penulis pada halaman pertama sebagai berikut:

1. Faturrahman Syauqi
2. Maulisa Oktiana
3. Kahlil Muchtar
4. Al Bahri
5. Safrizal Razali

Urutan ini **mengalahkan** urutan berbeda pada repository penulis atau metadata sekunder.

### PRE-04 — LOCKED
- Authors: Faturrahman Syauqi; Maulisa Oktiana; Kahlil Muchtar; Al Bahri; Safrizal Razali
- Year: 2025
- Title: *Edge AI-Based Defect Detection in White Pepper (Piper nigrum L.) Using CLAHE-Based Pre-processing and YOLO*
- Proceedings: *2025 IEEE International Conference on Networking, Intelligent Systems, and IoT (ICONS-IoT)*
- Pages: 18–23
- DOI: `10.1109/ICONS-IOT65216.2025.11211242`
- Authority: primary PDF downloaded from IEEE Xplore. Page 18 contains the title/author block and IEEE DOI footer; page 23 is the final paper page.
- Method guardrail: preprocessing adalah pipeline komposit dengan CLAHE sebagai komponen utama; jangan direduksi menjadi klaim bahwa metodenya hanya CLAHE.

## 3. Corrected cited-source count

Set formal cited-source saat ini adalah:

- Coffee/domain/standard: 13
- Detection/fine-grained/preprocessing/spectral/evaluation: 20
- XAI/activation visualization: 2
- **TOTAL: 35 sumber unik**

Status setelah koreksi:

- Metadata cukup untuk pembentukan bibliography: **35/35**
- `XAI-02` Grad-CAM++: backend/optional dan **tidak disitasi eksplisit**, sehingga tidak masuk bibliography.

## 4. Hard gate tetap berlaku

Koreksi ini tidak mengizinkan metadata baru dari tebakan. Setiap perubahan sitasi pada BAB I–III harus memicu audit ulang `CITATION_CROSSWALK.md`, `OFFICIAL_CITATION_AUDIT.md`, dan bibliography formal.