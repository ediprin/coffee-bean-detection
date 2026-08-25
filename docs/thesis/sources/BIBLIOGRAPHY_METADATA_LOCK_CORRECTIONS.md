# Bibliography Metadata Lock — Corrections

Status: **AUTHORITATIVE OVERRIDE** untuk `BIBLIOGRAPHY_METADATA_LOCK.md` sampai file induk dikonsolidasikan.

Dokumen ini dibuat karena audit silang menemukan beberapa hal yang harus dikunci secara eksplisit. Jika terjadi konflik, koreksi di file ini menang.

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

## 3. THEORY-01 — Gonzalez & Woods 4th edition convention

Sumber fundamental DFT/FFT harus memakai satu edition convention yang konsisten. Metadata berikut dikunci untuk proposal berdasarkan Pearson Global Edition dan record bibliografis ISBN yang sama.

### THEORY-01 — METADATA LOCKED, FORMULA-PAGE AUDIT OPEN
- Authors: Rafael C. Gonzalez; Richard E. Woods
- Year used in citation: **2018**
- Title: *Digital Image Processing*
- Edition: 4th edition, Global Edition
- Publisher: Pearson
- ISBN-13: `9781292223049`
- Authority: Pearson official Global Edition catalog; Pearson identifies the 4th edition and ©2018. Bibliographic record for this ISBN identifies Pearson, 2018, 1024 pages.
- Scope: authoritative image-processing foundation for DFT/IDFT, Fourier-domain representation, frequency-domain filtering, magnitude/spectrum and phase terminology.
- Hard limitation: exact formula-page locator remains open until the selected chapter/pages are available directly as project source. Do not fabricate page numbers.
- Decision: once `(Gonzalez & Woods, 2018)` is inserted into formal BAB II, add the corresponding APA entry to `DAFTAR_PUSTAKA.md` and rerun both citation audits.

Do not mix this ISBN/year convention with newer Pearson digital-update listings or a different US-edition ISBN.

## 4. EVAL-02 — official COCOeval implementation

### EVAL-02 — OFFICIAL IMPLEMENTATION VERIFIED
- Source: `cocodataset/cocoapi`
- File: `PythonAPI/pycocotools/cocoeval.py`
- Authority: official `cocodataset` GitHub organization / COCO API repository.
- Source-code statement: detection evaluation defaults define `iouThrs` as `[.5:.05:.95]`, i.e. 10 IoU thresholds from 0.50 through 0.95.
- Source code also supports `bbox` as an evaluation `iouType`.
- Safe use: support the exact IoU-threshold range when explaining COCO-style AP/mAP evaluation.
- Guardrail: do not claim the thesis evaluator is fully identical to all COCOeval defaults unless the actual evaluation path is checked. In particular, thesis `max_det` settings are not automatically COCOeval defaults.

This source may be cited as an official software/specification source if the formal proposal needs exact implementation-level support beyond Lin et al. (2014).

## 5. Current cited-source count

Set formal cited-source **yang sudah berada di BAB I–III saat file ini diperbarui** adalah:

- Coffee/domain/standard: 13
- Detection/fine-grained/preprocessing/spectral/evaluation: 20
- XAI/activation visualization: 2
- **TOTAL: 35 sumber unik**

Status:

- Metadata cukup untuk bibliography saat ini: **35/35**
- `THEORY-01` telah dikunci tetapi belum dihitung sampai benar-benar disitasi di artefak formal.
- `EVAL-02` telah diverifikasi sebagai official implementation tetapi belum dihitung sebagai entri bibliography formal kecuali sitasi formal ditambahkan.
- `XAI-02` Grad-CAM++: backend/optional dan **tidak disitasi eksplisit**, sehingga tidak masuk bibliography.

## 6. Hard gate tetap berlaku

Koreksi ini tidak mengizinkan metadata baru dari tebakan. Setiap perubahan sitasi pada BAB I–III harus memicu audit ulang `CITATION_CROSSWALK.md`, `OFFICIAL_CITATION_AUDIT.md`, dan bibliography formal.