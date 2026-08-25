# Foundational Citation Gaps — Proposal

Status: **OPEN HARDENING ITEMS, BUKAN BLOKIR METADATA BIBLIOGRAPHY 35-SUMBER SAAT INI**

Dokumen ini mencatat klaim fundamental/implementasi yang masih perlu penguatan sumber primer/otoritatif sebelum proposal disebut citation-ready secara penuh.

## G-01 — DFT / FFT / amplitude / phase foundation

### Kondisi manuscript

BAB II §2.8.1–2.8.2 saat ini memuat:

- definisi DFT 2-D;
- inverse DFT;
- pernyataan bahwa FFT merupakan algoritma untuk menghitung DFT secara lebih efisien;
- definisi amplitude dan phase dari koefisien kompleks.

### Source status

`THEORY-01` Gonzalez & Woods, *Digital Image Processing*, 4th ed., dan `THEORY-02` Bracewell, *The Fourier Transform and Its Applications*, 3rd ed., sudah memiliki bibliographic locator pada master workbook. Akan tetapi, pencarian File Library saat audit ini **tidak menemukan full-text/book pages**; yang ditemukan hanya workbook yang merekam metadata bibliografis dan secara eksplisit menyatakan exact cited pages masih perlu diakses/upload.

Karena itu:

- THEORY-01/THEORY-02 **belum ditambahkan sebagai sitasi formal**;
- keduanya **belum masuk `DAFTAR_PUSTAKA.md`**;
- jangan mengarang nomor halaman buku;
- sampai halaman sumber primer tersedia, formulasi yang dipertahankan harus dicocokkan dengan primary method papers yang memang tersedia di Project/File Library.

### Primary papers yang tersedia

- Xu et al. (2025), *Neural Networks* 187, 107402: full primary PDF tersedia. §3.3.1 secara eksplisit menjelaskan DFT untuk transformasi spatial→frequency, patch-wise DFT, dan patch-wise iDFT; §3.3.3 membahas amplitude/phase dan mempertahankan phase asli saat amplitude diremodel.
- Li et al. (2025), *Digital Signal Processing* 166, 105355: full primary PDF tersedia dan membahas Fourier decomposition serta amplitude/phase processing pada FE-YOLO.

Primary method papers tersebut boleh menopang pernyataan tentang **bagaimana metode mereka menggunakan Fourier**, tetapi tidak ideal sebagai satu-satunya authority untuk teori transformasi Fourier umum.

### Stop criterion

G-01 ditutup hanya jika salah satu terjadi:

1. halaman buku otoritatif yang relevan tersedia dan exact page/section diverifikasi; atau
2. subbab fundamental ditulis ulang sehingga setiap definisi/rumus yang dipertahankan dapat ditelusuri secara eksplisit ke primary source yang tersedia, tanpa mengklaim general theory melebihi sumber.

## G-02 — Exact COCO evaluation settings

### Official source verified

Official `cocodataset/cocoapi` implementation `PythonAPI/pycocotools/cocoeval.py` telah diperiksa. Header implementasi mendefinisikan default:

- IoU thresholds `[.5:.05:.95]` (10 thresholds);
- recall thresholds `[0:.01:1]` (101 thresholds);
- `maxDets` default `[1, 10, 100]`;
- evaluasi dapat menggunakan `bbox`, `segm`, atau `keypoints`.

Source code menyatakan: “Code written by Piotr Dollar and Tsung-Yi Lin, 2015.”

### Decision

- `EVAL-01` Lin et al. (2014) tetap menjadi scholarly benchmark reference.
- `EVAL-02` official COCOeval menjadi authority implementasi untuk exact metric settings.
- Jika BAB III mempertahankan klaim implementasi AP@[0.50:0.95], manuscript sebaiknya menyebut/merujuk official COCOeval secara eksplisit pada revisi berikutnya.
- Jangan menganggap `mAP50–95` otomatis identik dengan setiap evaluator/library tanpa mengecek evaluator yang benar-benar digunakan dalam eksperimen.

## G-03 — Efficiency measurement protocol

BAB III mencantumkan latency, throughput, memory, dan parameter count. Agar pengukuran dapat direproduksi, protokol hasil akhir harus mengunci minimal:

- hardware;
- batch size;
- input size;
- numerical precision;
- warm-up;
- jumlah pengulangan timing;
- statistik agregasi;
- cakupan waktu: preprocessing-only, detector-only, atau end-to-end.

Source eksternal tidak wajib bila protokol internal didefinisikan secara eksplisit dan konsisten, tetapi manuscript tidak boleh membandingkan latency/throughput yang diukur dengan kondisi berbeda.

## Current decision

`DAFTAR_PUSTAKA.md` saat ini memuat **35 sumber yang memang disitasi pada snapshot BAB I–III saat ini**. Dokumen ini tidak menambah sumber baru ke bibliography. Penambahan THEORY-01/THEORY-02 atau EVAL-02 hanya dilakukan setelah manuscript benar-benar menggunakan sitasi tersebut dan source gate yang diperlukan ditutup.