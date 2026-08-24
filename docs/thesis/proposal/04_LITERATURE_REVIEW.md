# BAB II — TINJAUAN PUSTAKA

Status: structural draft following the adopted USU/campus proposal pattern.

Do not finalize bibliography wording until each citation key has been resolved to the verified primary source.

---

## 2.1 Biji Kopi Hijau dan Cacat Fisik Biji Kopi

Biji kopi hijau (*green coffee bean*) merupakan biji kopi yang telah melalui proses pascapanen tertentu tetapi belum mengalami proses penyangraian. Pada tahap ini, kondisi fisik biji menjadi salah satu aspek penting dalam penilaian mutu karena berbagai jenis cacat dapat dikenali melalui karakteristik warna, bentuk, permukaan, kerusakan lokal, atau keberadaan material asing.

Dalam penelitian ini, istilah cacat fisik digunakan untuk merujuk pada kategori kondisi biji yang menjadi target inspeksi visual pada dataset penelitian. Konteks standar mutu seperti SNI dan SCA digunakan untuk menjelaskan relevansi kategori cacat, tetapi penelitian tidak melakukan rekonstruksi penuh proses grading manual berbasis nilai cacat.

**Evidence to insert:** coffee standards / taxonomy sources; verified dataset taxonomy; direct coffee-defect literature.

**Do not overclaim:** taxonomy in the thesis must match the actual dataset labels used in the experiment.

---

## 2.2 Inspeksi Mutu Biji Kopi: Metode Konvensional dan Tantangannya

Inspeksi fisik biji kopi secara konvensional dilakukan melalui pengamatan visual untuk membedakan biji normal, cacat, dan material asing berdasarkan karakteristik yang tampak. Pendekatan manual memiliki keunggulan berupa kemudahan implementasi, tetapi dapat dipengaruhi pengalaman operator, konsistensi pengamatan, kelelahan, kondisi pencahayaan, dan jumlah sampel yang harus diperiksa.

Literatur coffee-defect recognition menunjukkan bahwa kesulitan menjadi semakin nyata ketika kategori cacat diperinci. Beberapa kelas dapat memiliki perbedaan visual yang kecil, sedangkan kategori lain mempunyai ciri yang lebih jelas. Hal ini menyebabkan performa antarkelas dapat sangat berbeda walaupun metrik agregat model terlihat tinggi. [COF-01][COF-02][COF-03][COF-04][COF-05][COF-07]

Kondisi tersebut mendorong penggunaan computer vision dan deep learning untuk menghasilkan inspeksi yang lebih konsisten, terukur, dan dapat diotomatisasi.

---

## 2.3 Object Detection

Object detection merupakan tugas computer vision yang bertujuan menentukan **kelas objek** sekaligus **lokasi objek** pada citra. Secara umum, keluaran detector mencakup bounding box, skor kepercayaan, dan prediksi kelas.

Untuk suatu objek dengan bounding box prediksi \(B_p\) dan ground-truth \(B_g\), kualitas tumpang tindih dapat dinilai menggunakan *Intersection over Union* (IoU):

\[
IoU = \frac{|B_p \cap B_g|}{|B_p \cup B_g|}.
\]

Object detection memuat dua submasalah yang saling berhubungan tetapi tidak identik, yaitu klasifikasi dan lokalisasi. Literatur menunjukkan bahwa kualitas skor klasifikasi tidak selalu ekuivalen dengan kualitas lokalisasi, dan kedua tugas dapat membutuhkan representasi fitur yang berbeda. [DIAG-01][DIAG-02][DIAG-03]

Pemisahan konseptual ini penting dalam penelitian karena peningkatan metrik deteksi tidak langsung diasumsikan sebagai peningkatan kemampuan lokalisasi.

---

## 2.4 YOLO (You Only Look Once)

YOLO merupakan keluarga *one-stage object detector* yang melakukan prediksi kelas dan bounding box dalam satu alur inferensi. Pendekatan ini banyak digunakan pada aplikasi yang memerlukan kecepatan dan efisiensi, termasuk inspeksi pertanian dan deteksi cacat visual.

Berbagai penelitian kopi telah menggunakan keluarga YOLO untuk mendeteksi biji dan cacat kopi. Studi dengan taxonomy relatif kecil menunjukkan bahwa YOLO dapat mencapai performa tinggi pada domain green coffee bean, sedangkan studi dengan 15–20 kelas menunjukkan variasi performa antarkelas yang lebih besar. [COF-01][COF-02][COF-04][COF-05][COF-06]

**Drafting note:** historical YOLO timeline should remain concise. This thesis does not need a version-by-version encyclopedia.

---

## 2.5 YOLO26

YOLO26 digunakan sebagai detector utama dalam penelitian ini. Subbab ini akan menjelaskan arsitektur dan karakteristik YOLO26 hanya sejauh diperlukan untuk memahami rancangan eksperimen, meliputi komponen backbone, feature aggregation/neck, detection head, proses prediksi, dan karakteristik pelatihan yang relevan terhadap perbandingan baseline dan metode usulan.

Rancangan penelitian tidak mengubah YOLO26 sebagai bagian utama kontribusi. Perbandingan dikendalikan dengan menggunakan detector, inisialisasi, dataset, dan budget pelatihan yang sama, sedangkan perbedaan utama terletak pada kondisi input sebelum detector.

**Sources required before final prose:** original YOLO26 paper/documentation + repository configuration used in the experiment.

**Boundary:** AF2 must not be described as an internal YOLO26 module.

---

## 2.6 Fine-Grained Object Detection

Fine-grained recognition dan fine-grained object detection berhubungan dengan pengenalan kategori yang memiliki perbedaan visual antarkelas relatif kecil. Dalam kondisi seperti ini, model tidak cukup hanya mengenali keberadaan objek umum, tetapi harus mempertahankan informasi diskriminatif yang mampu membedakan pola warna, tekstur, bentuk, atau kerusakan lokal yang saling mirip.

Pada domain kopi, bukti fine-grained difficulty muncul dalam beberapa bentuk. Benchmark klasifikasi menunjukkan penurunan tajam ketika taxonomy diperluas dari kelas kasar menjadi 17 kategori cacat [COF-07]. Studi object detection dengan taxonomy 15–20 kelas menunjukkan ketimpangan AP antarkelas dan kegagalan pada beberapa defect yang visualnya menyerupai kelas lain atau normal bean [COF-02][COF-04][COF-05]. Studi lain secara eksplisit menyebut *subtle visual differences* sebagai tantangan representasi [COF-13].

Dengan demikian, penelitian ini memandang masalah utama sebagai kebutuhan meningkatkan kemampuan diskriminasi fine-grained. Namun literatur kopi tersebut **tidak membuktikan** bahwa penyebab kesulitan adalah kurangnya informasi domain frekuensi.

---

## 2.7 Preprocessing Citra untuk Object Detection

Preprocessing citra merupakan transformasi yang dilakukan sebelum citra diproses oleh model utama. Tujuannya dapat berupa normalisasi data, peningkatan kontras, pengurangan noise, penajaman detail, atau transformasi representasi agar informasi yang relevan lebih mudah dimanfaatkan oleh detector.

Pendekatan preprocessing dapat dibedakan secara sederhana menjadi dua kelompok. Pertama, preprocessing tetap (*fixed preprocessing*) seperti CLAHE, filtering, denoising, dan sharpening. Kedua, preprocessing yang parameter atau transformasinya dipelajari secara *task-driven* bersama detector.

Syauqi et al. menggunakan pipeline preprocessing berbasis CLAHE sebelum YOLOv8m pada white pepper dan membandingkannya dengan data tanpa preprocessing [PRE-04]. Chen et al. menggunakan kombinasi wavelet denoising dan image enhancement sebelum YOLOv8 pada maize seed crack detection [PRE-05]. IA-YOLO menunjukkan bahwa transformasi citra dapat dipelajari dengan tujuan meningkatkan downstream detection [PRE-01], sedangkan DENet menggunakan dekomposisi multiskala/frekuensi untuk menghasilkan enhancement yang diarahkan oleh kebutuhan deteksi [PRE-02].

Literatur ini mendukung penggunaan preprocessing sebagai ruang solusi yang sah, tetapi tidak membuktikan bahwa satu jenis preprocessing tertentu akan efektif pada cacat biji kopi.

---

## 2.8 Representasi Citra pada Domain Frekuensi

### 2.8.1 Discrete Fourier Transform dan Fast Fourier Transform

Transformasi Fourier merepresentasikan citra dari domain spasial ke domain frekuensi. Untuk citra diskrit dua dimensi \(f(x,y)\) berukuran \(M \times N\), *Discrete Fourier Transform* dapat dituliskan sebagai:

\[
F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)
\exp\left[-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
\]

Transformasi balik merekonstruksi citra spasial dari representasi frekuensi:

\[
f(x,y)=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)
\exp\left[j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
\]

Fast Fourier Transform (FFT) merupakan algoritma komputasional efisien untuk menghitung DFT dan menjadi dasar implementasi pemrosesan spektral pada penelitian ini.

### 2.8.2 Magnitudo/Amplitudo dan Fase Spektrum

Representasi kompleks Fourier dapat dinyatakan melalui magnitudo dan fase:

\[
A(u,v)=|F(u,v)|,
\]

\[
\phi(u,v)=\operatorname{atan2}(\Im(F(u,v)),\Re(F(u,v))).
\]

Magnitudo menggambarkan kekuatan komponen frekuensi, sedangkan fase berkaitan dengan susunan spasial informasi pada citra. Literatur Fourier-based enhancement seperti FE-YOLO memanfaatkan pemrosesan amplitude/phase sebelum rekonstruksi citra dan deteksi [PRE-03].

Penelitian ini tidak mengasumsikan bahwa magnitude atau phase merupakan bottleneck yang telah terbukti pada kopi.

### 2.8.3 Representasi Radial dan Angular pada Spektrum Fourier

Spektrum dua dimensi dapat dianalisis menggunakan koordinat polar berdasarkan radius frekuensi \(r\) dan sudut \(\theta\). Distribusi radial digunakan untuk merangkum energi berdasarkan skala/frekuensi, sedangkan distribusi angular digunakan untuk merangkum energi berdasarkan orientasi/directionality.

Secara konseptual, energi pada suatu rentang radial dapat dituliskan sebagai:

\[
E_r(r_1,r_2)=\sum_{r_1^2\le u^2+v^2\le r_2^2}|F(u,v)|^2,
\]

sementara energi angular dapat diringkas sebagai:

\[
E_\theta(\theta_1,\theta_2)=
\sum_{\theta_1\le \theta(u,v)\le \theta_2}|F(u,v)|^2.
\]

Literatur texture analysis menunjukkan bahwa radial spectrum dapat berkaitan dengan periodisitas/skala tekstur, sedangkan angular spectrum dapat menggambarkan directionality dan orientasi pola [FREQ-01][FREQ-02].

Inilah dasar teoritis penggunaan istilah **frekuensi-angular** pada penelitian, bukan bukti bahwa operator AF2 telah terbukti efektif pada kopi.

### 2.8.4 Pemrosesan Frekuensi untuk Object Detection

Pemrosesan frekuensi telah digunakan pada object detection dalam beberapa bentuk. FE-YOLO melakukan Fourier enhancement pada citra sebelum YOLO [PRE-03]. Pendekatan wavelet dan Fourier lain memasukkan informasi frekuensi ke feature space detector [AGR-01][AGR-02][FREQ-03][FREQ-04]. Pada fine-grained object detection, Xu et al. mengeksplorasi integrasi representasi frekuensi untuk membedakan kategori yang memiliki perbedaan visual subtil [FG-01].

Dari literatur tersebut dapat ditarik satu posisi yang aman: informasi frekuensi dan arah merupakan ruang representasi yang secara teknis dapat membawa informasi komplementer untuk detection dan texture discrimination. Efektivitas **preprocessing input parameter-free berbasis frekuensi-angular pada fine-grained coffee defect detection** masih harus dibuktikan melalui eksperimen penelitian ini.

---

## 2.9 Penelitian Terkait

Tabel 2.1 merangkum penelitian yang paling relevan terhadap tiga jalur utama tesis: (1) deteksi cacat kopi, (2) preprocessing sebelum object detector, dan (3) pemrosesan frekuensi untuk fine-grained/detection tasks.

### Tabel 2.1. Perbandingan studi relevan deteksi cacat kopi, preprocessing citra, dan pemrosesan frekuensi

| No | Penulis & Tahun | Indeks | Fokus Penelitian | Metode / Model | Kontribusi dan Pengisian Gap Penelitian |
|---:|---|---|---|---|---|
| 1 | Hong et al. (2026) [COF-01] | TBD - verify | Deteksi cacat biji kopi dan peningkatan fine-grained feature extraction | Improved YOLOv10 + DSConv + SPPF-Attention + PConv | Menunjukkan YOLO dapat bekerja sangat baik pada coffee detection, tetapi subtle/visually similar defects masih menjadi motivasi peningkatan representasi. Penelitian yang diusulkan tidak menambah modul internal serupa, melainkan menguji preprocessing input. |
| 2 | Bahy & Rifai (2026) [COF-02] | TBD - verify | Deteksi cacat kopi berbasis SNI dengan taxonomy besar | Lightweight YOLOv5s; 20 kelas | Memberikan bukti bahwa taxonomy 20 kelas menghasilkan heterogenitas performa antarkelas. Mendukung evaluasi class-wise dan lower-tail. |
| 3 | Jundullah et al. (2026) [COF-05] | TBD - verify | Multi-class coffee defect and contaminant detection | YOLOv8s; 20 kelas | Menunjukkan overall performance dapat menyembunyikan kelas yang sulit dan visually similar. Mendukung problem fine-grained discrimination. |
| 4 | Hebert & Alamsyah (2026) [COF-04] | TBD - verify | Deteksi cacat kopi berbasis kategori SCA | YOLOv12; 15 kelas | Menunjukkan beberapa subtle defects memiliki AP sangat rendah dibanding kelas yang visualnya khas. Menguatkan kebutuhan analisis difficult classes. |
| 5 | Kesiman et al. (2023) [COF-07] | TBD - verify | Benchmark klasifikasi cacat kopi berbasis SNI | MobileNet dan InceptionResNetV2; 3 vs 17 kelas | Menunjukkan peningkatan granularitas taxonomy secara drastis meningkatkan kesulitan diskriminasi. Digunakan sebagai diagnostic evidence, bukan bukti object detection. |
| 6 | Syauqi et al. (2025) [PRE-04] | IEEE conference - verify index | Defect detection pada white pepper dengan image enhancement sebelum YOLO | CLAHE-based composite preprocessing + YOLOv8m | Memberi analog komoditas berbentuk biji bahwa fixed preprocessing sebelum detector dapat meningkatkan downstream detection; hanya 2 kelas dan bukan coffee. |
| 7 | Chen et al. (2024) [PRE-05] | Computers and Electronics in Agriculture - verify quartile | Deteksi retak biji jagung berbasis soft X-ray | Image enhancement + optimized YOLOv8 | Memisahkan kontribusi image enhancement dari optimasi detector; mendukung preprocessing sebagai treatment eksperimental yang dapat diuji secara terkontrol. |
| 8 | Liu et al. (2022), IA-YOLO [PRE-01] | AAAI - verify index wording | Object detection pada adverse weather melalui task-driven image preprocessing | Differentiable image processing + YOLOv3 | Menunjukkan preprocessing dapat diarahkan untuk utility deteksi, bukan hanya kualitas visual. Learned/adaptive dan berbeda dari AF2 parameter-free. |
| 9 | Li et al. (2025), FE-YOLO [PRE-03] | Digital Signal Processing - verify quartile | Low-light detection dengan Fourier-domain enhancement | FFT + learned FENet/FPB + IFFT + YOLO | Pembanding metodologis terdekat untuk input-space Fourier enhancement. Berbeda karena learned, low-light oriented, dan tidak memakai angular spectral selection. |
| 10 | Xu et al. (2025) [FG-01] | TBD - verify | Fine-grained object detection dengan language dan frequency representations | LFDet + AFAB/AFAB-2 | Memberi bridge bahwa frequency representation dapat digunakan untuk fine-grained detection; domain aircraft sehingga transfer ke kopi belum terbukti. |
| 11 | **Penelitian yang Diusulkan** | - | Analisis preprocessing frekuensi-angular untuk deteksi fine-grained cacat biji kopi | **AF2 parameter-free preprocessing + YOLO26** | Mengisi gap dengan mengevaluasi input-space frequency-angular preprocessing pada coffee defect detection secara matched terhadap native YOLO26, disertai analisis aggregate, lower-tail, classification/localization diagnostics, dan efficiency trade-off. |

### Positioning summary

Pola penelitian terdahulu dapat disederhanakan menjadi:

```text
coffee defect detection
    ↓
YOLO / CNN / Transformer improvements inside the model
    ↓
finer taxonomy exposes class-wise discrimination difficulty
```

sementara literatur preprocessing dan frekuensi menunjukkan:

```text
image preprocessing can affect downstream detection
    +
frequency / directional representations can encode complementary visual information
```

Penelitian ini menguji titik pertemuan kedua jalur tersebut melalui:

```text
RGB image
    ↓
parameter-free frequency-angular preprocessing
    ↓
YOLO26
    ↓
fine-grained coffee-defect detection
```

The effectiveness of this transfer is an empirical research question and must not be assumed from prior non-coffee studies.
