# BAB II
# TINJAUAN PUSTAKA

## 2.1 Biji Kopi Hijau, Cacat Fisik, dan Benda Asing

Biji kopi hijau merupakan biji kopi yang belum melalui proses penyangraian dan masih dinilai berdasarkan karakteristik fisiknya. SNI 2907:2008 mengatur persyaratan mutu biji kopi Robusta dan Arabika, termasuk penggolongan mutu, jenis cacat fisik, cara pengujian, penandaan, dan pengemasan (Badan Standardisasi Nasional, 2008). Dalam penelitian berbasis citra, karakteristik fisik tersebut menjadi informasi visual untuk membedakan biji normal, biji cacat, dan benda asing.

SNI 2907:2008 mencakup kondisi seperti biji hitam, biji hitam sebagian, biji hitam pecah, biji coklat, biji pecah, biji muda, biji berlubang akibat serangga, kulit kopi, kulit tanduk, serta benda asing seperti ranting, tanah, atau batu. Standar tersebut menggunakan sistem nilai cacat untuk menentukan mutu kopi. Dalam penelitian *computer vision*, susunan kelas yang diprediksi tetap ditentukan oleh dataset dan protokol anotasi yang digunakan.

Kesiman et al. (2023) mengembangkan dataset klasifikasi cacat yang mengacu pada SNI 2907:2008 dengan subset akhir 17 kelas. Arwatchananukul et al. (2024) mengembangkan dataset Thai Arabica dengan 17 jenis cacat, sedangkan Bahy dan Rifai (2026) menggunakan 20 kategori fisik pada deteksi objek. Pada deteksi multiobjek, Bahy dan Rifai (2026) melaporkan 107 citra dengan 13.863 anotasi, sementara Tarekegn dan Debelee (2025) menggunakan 562 citra dengan 19.228 objek pada 13 kelas cacat dan satu kelas normal. Data tersebut menunjukkan bahwa skala dataset deteksi perlu dilihat dari jumlah citra sumber sekaligus jumlah objek per kelas.

## 2.2 Inspeksi Mutu Biji Kopi

Identifikasi cacat biji kopi secara konvensional dilakukan melalui inspeksi visual dengan memperhatikan bentuk, warna, tekstur, dan tanda cacat pada permukaan biji. Kesiman et al. (2023) menyebut kebutuhan waktu, tenaga, dan pekerja berpengalaman sebagai keterbatasan inspeksi manual, sedangkan Arwatchananukul et al. (2024) juga menyoroti ketergantungan pada tenaga manusia dalam proses pemilahan biji kopi hijau.

Sebelum berkembangnya *deep learning*, otomasi inspeksi kopi telah dilakukan menggunakan pengolahan citra dan fitur yang dirancang secara manual. De Oliveira et al. (2016) menggunakan kondisi pengambilan citra terkontrol, kalibrasi warna, dan ruang warna CIE L*a*b* untuk mengukur karakteristik warna biji kopi hijau dan melakukan klasifikasi. Pendekatan modern memungkinkan representasi visual dipelajari langsung dari data melalui jaringan konvolusional, Transformer, dan model deteksi objek.

## 2.3 Deteksi Objek

Deteksi objek (*object detection*) menentukan kategori sekaligus lokasi objek pada citra. Lokasi umumnya direpresentasikan dengan kotak pembatas (*bounding box*), sedangkan kategori dinyatakan melalui skor kelas. Secara umum, metode deteksi dapat dibedakan menjadi pendekatan dua tahap (*two-stage*) dan satu tahap (*one-stage*).

Faster R-CNN menggunakan *Region Proposal Network* untuk menghasilkan kandidat objek sebelum klasifikasi dan regresi kotak pembatas dilakukan (Ren et al., 2015). Sebaliknya, YOLO merumuskan deteksi sebagai prediksi langsung dari citra menuju lokasi objek dan probabilitas kelas dalam satu jaringan (Redmon et al., 2016).

Kesesuaian spasial antara kotak prediksi $B_p$ dan *ground truth* $B_g$ dapat diukur menggunakan *Intersection over Union* (IoU):

$$
IoU(B_p,B_g)=\frac{|B_p\cap B_g|}{|B_p\cup B_g|}.
$$

Klasifikasi dan lokalisasi tetap merupakan dua tugas yang berbeda di dalam sistem deteksi. Feng et al. (2021) membahas ketidakselarasan kedua tugas melalui TOOD, Wu et al. (2020) menunjukkan bahwa representasi yang sesuai untuk klasifikasi tidak selalu identik dengan representasi terbaik untuk regresi lokasi, dan Jiang et al. (2018) menunjukkan bahwa keyakinan klasifikasi tidak sama dengan kualitas lokalisasi.

## 2.4 You Only Look Once (YOLO)

YOLO diperkenalkan oleh Redmon et al. (2016) sebagai pendekatan deteksi yang memprediksi kotak pembatas dan probabilitas kelas secara langsung dari citra. Generasi berikutnya mengembangkan berbagai rancangan *backbone*, agregasi fitur, *detection head*, strategi penetapan target, fungsi kerugian, dan mekanisme inferensi.

Keluarga YOLO telah digunakan pada inspeksi biji kopi. Gope et al. (2024) membandingkan beberapa varian YOLO pada deteksi biji kopi hijau, Hong et al. (2026) menggunakan YOLOv10 untuk tujuh kategori cacat, dan Bahy dan Rifai (2026) menerapkan YOLOv5s pada 20 kategori fisik. Hasil-hasil tersebut menunjukkan kelayakan YOLO pada domain kopi sekaligus variasi kinerja ketika kategori yang dibedakan semakin rinci.

## 2.5 YOLO26 dan Pembanding Arsitektur

YOLO26 merupakan keluarga model *real-time vision* yang diperkenalkan oleh Jocher et al. (2026). Pada tugas deteksi, YOLO26 menggunakan rancangan *dual-head* yang mendukung jalur inferensi end-to-end tanpa *Non-Maximum Suppression* (NMS) sebagai jalur utama. Paper YOLO26 juga menjelaskan perubahan pada regresi kotak pembatas, penetapan target, dan optimisasi dibandingkan generasi sebelumnya.

Arsitektur deteksi YOLO26 tetap menggunakan alur *backbone*, *neck*, dan *detection head*, dengan prediksi pada beberapa tingkat fitur. Keluarga ini tersedia dalam beberapa skala model; varian YOLO26n digunakan sebagai model utama dalam penelitian ini.

Sebagai keluarga pembanding, Wang et al. (2025) mengembangkan RT-DETRv3 sebagai detektor Transformer end-to-end berbasis RT-DETR dengan tambahan supervisi positif yang lebih padat pada pelatihan. Varian R18 digunakan hanya untuk analisis lintas arsitektur setelah konfigurasi prapemrosesan utama ditetapkan.

## 2.6 Fine-Grained Object Detection

*Fine-grained recognition* membahas pengenalan kategori subordinat yang memiliki kemiripan visual tinggi. Perbedaan kelas dapat ditentukan oleh tekstur, warna, pola lokal, atau struktur yang relatif halus. Pada *fine-grained object detection*, sistem harus menentukan lokasi objek sekaligus membedakan subkategori yang berdekatan secara visual.

Xie et al. (2025) menekankan kebutuhan representasi yang cukup diskriminatif untuk membedakan kategori berdekatan pada tugas deteksi. Karakteristik serupa terlihat pada domain kopi. Kesiman et al. (2023) melaporkan penurunan akurasi ketika klasifikasi diperluas dari tiga menjadi 17 kelas cacat, sedangkan Jundullah et al. (2026) dan Hebert dan Alamsyah (2026) menunjukkan variasi kinerja antarkelas pada deteksi dengan susunan kategori yang lebih rinci.

Jumlah kelas bukan satu-satunya penentu sifat *fine-grained*; kedekatan karakteristik visual antarkelas merupakan aspek yang lebih utama. Karena itu, analisis *fine-grained* terutama relevan pada kelas yang memiliki ciri visual serupa atau menunjukkan kinerja lebih rendah.

## 2.7 Prapemrosesan Citra untuk Deteksi Objek

Prapemrosesan citra merupakan transformasi terhadap citra sebelum diterima model utama. Pada deteksi objek, prapemrosesan dapat digunakan untuk mengubah kontras, menekan *noise*, mempertahankan detail, atau mengubah representasi sinyal.

Syauqi et al. (2025) menerapkan CLAHE bersama *gamma correction*, *denoising*, dan *unsharp masking* sebelum YOLOv8m pada deteksi cacat *white pepper*. Chen et al. (2024) menggunakan kombinasi *wavelet-threshold denoising*, standardisasi citra, *bilateral filtering*, dan *Laplacian sharpening* sebelum YOLOv8 untuk mendeteksi keretakan biji jagung. Kedua penelitian tersebut menunjukkan penggunaan transformasi citra sebelum model deteksi, tetapi kontribusi tiap komponen tidak selalu dapat dipisahkan karena digunakan secara komposit.

Liu et al. (2022) melalui IA-YOLO menggunakan *differentiable image-processing filters* yang dioptimalkan berdasarkan *detection loss*. Qin et al. (2022) melalui DENet memisahkan citra menjadi komponen frekuensi rendah dan tinggi menggunakan *Laplacian pyramid* sebelum rekonstruksi dan deteksi. Li et al. (2025) melalui FE-YOLO memproses amplitudo dan fase pada domain Fourier sebelum citra direkonstruksi dan diberikan kepada YOLO. Literatur tersebut menunjukkan bahwa kualitas prapemrosesan dapat dinilai berdasarkan pengaruhnya terhadap tugas deteksi, bukan hanya perubahan visual citra.

## 2.8 Representasi Citra pada Domain Frekuensi

### 2.8.1 Discrete Fourier Transform dan Fast Fourier Transform

Citra digital dapat dipandang sebagai sinyal dua dimensi pada domain spasial. *Discrete Fourier Transform* (DFT) mengubah representasi tersebut ke domain frekuensi (Gonzalez & Woods, 2018). Bentuk DFT dan transformasi balik juga dituliskan oleh Xu et al. (2025, §3.1.1, Persamaan 1 dan 4). Untuk citra diskrit $f(x,y)$ berukuran $M\times N$:

$$
F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)
\exp\left[-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
$$

Transformasi balik dinyatakan sebagai:

$$
f(x,y)=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)
\exp\left[j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
$$

*Fast Fourier Transform* (FFT) merupakan algoritma untuk menghitung DFT secara lebih efisien. Pada penelitian terkait, Yang dan Soatto (2020) memanipulasi amplitudo Fourier untuk *domain adaptation*, Li et al. (2025) melakukan pemrosesan spektral sebelum YOLO, dan Xu et al. (2025) menggunakan DFT lokal pada patch.

### 2.8.2 Amplitudo dan Fase

Koefisien Fourier $F(u,v)$ merupakan bilangan kompleks:

$$
F(u,v)=R(u,v)+jI(u,v).
$$

Amplitudo dan fase dihitung sebagai:

$$
A(u,v)=\sqrt{R^2(u,v)+I^2(u,v)},
$$

$$
\phi(u,v)=\mathrm{atan2}(I(u,v),R(u,v)).
$$

Bentuk tersebut sejalan dengan Persamaan (2) dan (3) pada Xu et al. (2025, §3.1.1). Amplitudo menunjukkan besar respons spektral, sedangkan fase berkaitan dengan susunan spasial dalam representasi Fourier (Gonzalez & Woods, 2018). Xu et al. (2025) memodifikasi respons amplitudo dan mempertahankan fase asli saat rekonstruksi pada LFDet.

### 2.8.3 Representasi Radial dan Angular

Untuk pusat spektrum $(u_c,v_c)$, radius dan sudut koordinat frekuensi dapat dituliskan sebagai:

$$
r(u,v)=\sqrt{(u-u_c)^2+(v-v_c)^2},
$$

$$
\theta(u,v)=\mathrm{atan2}(v-v_c,u-u_c).
$$

Representasi radial mengelompokkan informasi berdasarkan jarak dari pusat spektrum, sedangkan representasi angular mengelompokkannya berdasarkan arah. Cao et al. (2019) menggunakan distribusi radial dan angular energi spektrum Fourier untuk analisis tekstur *remote sensing*, sementara Zhang dan Tan (2003) menunjukkan penggunaan distribusi orientasi spektral sebagai ciri diskriminatif pada klasifikasi tekstur.

Pada *fine-grained object detection*, Xu et al. (2025) mengembangkan *Adaptive Frequency Augmentation Branch* (AFAB). AFAB menggunakan DFT berbasis patch; AFAB-1 menerapkan penyaring lolos-tinggi adaptif, sedangkan AFAB-2 menyesuaikan amplitudo berdasarkan distribusi angular sebelum rekonstruksi dengan fase asli. Penelitian ini mengambil prinsip AFAB-2 sebagai konfigurasi referensi untuk pemrosesan frekuensi-angular pada ruang citra masukan.

Istilah *frekuensi-angular* dalam penelitian ini merujuk pada representasi Fourier lokal dan analisis amplitudo berdasarkan arah, bukan orientasi kotak pembatas atau *oriented object detection*.

### 2.8.4 Pemrosesan Frekuensi pada Computer Vision

Pemrosesan frekuensi dapat diterapkan pada citra masukan maupun fitur internal jaringan. Yang dan Soatto (2020), Li et al. (2025), dan Xu et al. (2025) menunjukkan pemrosesan pada atau sebelum tahap ekstraksi fitur utama. Sebaliknya, Chi et al. (2020), Li et al. (2024), dan Chen et al. (2025) menerapkan operasi frekuensi pada fitur internal jaringan.

Perbedaan posisi tersebut penting karena prapemrosesan pada ruang masukan dapat dibandingkan tanpa mengubah arsitektur utama model. Literatur yang ditinjau belum menetapkan *frequency signature* khusus untuk cacat biji kopi; pengaruh representasi frekuensi-angular pada domain ini karena itu perlu dievaluasi secara empiris.

## 2.9 Visualisasi Aktivasi Model

Visualisasi aktivasi digunakan untuk membantu menginterpretasikan respons internal jaringan terhadap suatu prediksi. Selvaraju et al. (2017) memperkenalkan *Gradient-weighted Class Activation Mapping* (Grad-CAM), yang menggunakan gradien target prediksi terhadap peta fitur untuk membentuk peta aktivasi *class-discriminative*.

Muhammad dan Yeasin (2020) memperkenalkan *Eigen-CAM*, yang menggunakan komponen utama dari representasi fitur dan tidak bergantung pada *backpropagation gradient* maupun *class relevance score*. Eigen-CAM menjadi kandidat utama untuk analisis visual, sedangkan Grad-CAM dapat digunakan apabila target prediksi dan aliran gradien pada YOLO26n dapat didefinisikan secara konsisten. Metode akhir ditentukan setelah kompatibilitas teknis diverifikasi.

## 2.10 Penelitian Terkait

Penelitian yang relevan mencakup deteksi cacat biji kopi, prapemrosesan sebelum model deteksi, dan pemrosesan frekuensi pada tugas *fine-grained* atau deteksi objek. Ringkasannya ditunjukkan pada Tabel 2.1.

### Tabel 2.1 Penelitian Terkait

| No. | Penulis dan Tahun | Sumber Publikasi/Venue | Fokus Penelitian | Metode/Model | Kontribusi terhadap Penelitian |
|---:|---|---|---|---|---|
| 1 | Hong et al. (2026) | *Current Research in Food Science* | Deteksi tujuh kategori cacat biji kopi | Improved YOLOv10 dengan modifikasi ekstraksi dan pemrosesan fitur | Menunjukkan keluarga YOLO dapat digunakan untuk deteksi cacat kopi dan bahwa kemiripan visual antarkategori tetap menjadi tantangan. |
| 2 | Gope et al. (2024) | *Scientific Reports* | Deteksi dan klasifikasi cacat biji kopi hijau | Perbandingan beberapa varian YOLO | Menunjukkan kelayakan keluarga YOLO pada domain biji kopi dengan jumlah kelas yang relatif terbatas. |
| 3 | Bahy dan Rifai (2026) | *International Journal on ICT* | Deteksi 20 kategori fisik berbasis SNI | Lightweight YOLOv5s | Menunjukkan adanya perbedaan kinerja antarkelas pada susunan kelas yang lebih rinci dan memberi contoh dataset deteksi multiobjek. |
| 4 | Tarekegn dan Debelee (2025) | *Journal on Artificial Intelligence* | Deteksi cacat biji kopi multiobjek | KN-YOLOv8 | Memberikan pembanding skala pengumpulan dataset primer dan jumlah objek pada citra deteksi biji kopi. |
| 5 | Jundullah et al. (2026) | *Brilliance: Research of Artificial Intelligence* | Deteksi multikelas cacat dan kontaminan | YOLOv8s | Menunjukkan bahwa kinerja agregat dapat disertai ketimpangan kinerja antarkelas dan kesulitan pada kelas yang mirip secara visual. |
| 6 | Hebert dan Alamsyah (2026) | *INOVTEK Polbeng - Seri Informatika* | Deteksi 15 kategori cacat biji kopi | YOLOv12 | Menunjukkan bahwa beberapa kategori dengan tanda cacat yang halus memiliki kinerja deteksi lebih rendah. |
| 7 | Kesiman et al. (2023) | ICITRI 2023 | Klasifikasi cacat berbasis SNI | MobileNet dan InceptionResNetV2 | Menunjukkan bahwa peningkatan granularitas dari tiga kelas menjadi 17 kelas meningkatkan kesulitan diskriminasi. |
| 8 | Arwatchananukul et al. (2024) | *Smart Agricultural Technology* | Klasifikasi 17 jenis cacat biji kopi Arabika | Transfer learning CNN | Menunjukkan pentingnya pengujian pada data yang tidak terlihat sebelumnya pada klasifikasi fine-grained. |
| 9 | Hu et al. (2025) | *LWT* | Pengenalan cacat kopi dengan perbedaan visual halus | Siamese network | Menunjukkan bahwa pembelajaran berbasis kemiripan dapat digunakan untuk meningkatkan diskriminasi antarkelas. |
| 10 | Liu et al. (2022) | AAAI 2022 | Prapemrosesan adaptif untuk deteksi objek pada cuaca buruk | IA-YOLO | Menunjukkan bahwa prapemrosesan dapat dioptimalkan berdasarkan kebutuhan tugas deteksi. |
| 11 | Syauqi et al. (2025) | IEEE ICONS-IoT 2025 | Deteksi cacat white pepper | Composite preprocessing + YOLOv8m | Memberikan contoh rangkaian prapemrosesan sebelum model deteksi; karena bersifat komposit, hasilnya tidak mengisolasi kontribusi CLAHE. |
| 12 | Chen et al. (2024) | *Computers and Electronics in Agriculture* | Deteksi keretakan biji jagung | Image enhancement + YOLOv8 | Menunjukkan wavelet dan peningkatan citra sebagai alternatif prapemrosesan multiskala, tetapi dengan ruang desain tambahan. |
| 13 | Li et al. (2025) | *Digital Signal Processing* | Deteksi objek pada pencahayaan rendah | Fourier enhancement + YOLO | Menunjukkan pemrosesan Fourier pada citra masukan sebelum model deteksi. |
| 14 | Xu et al. (2025) | *Neural Networks* | Fine-grained aircraft detection | LFDet dengan AFAB | Menunjukkan penggunaan pemrosesan frekuensi lokal dan distribusi angular pada tugas fine-grained detection. |
| 15 | Xie et al. (2025) | *IEEE Transactions on Circuits and Systems for Video Technology* | Fine-grained object detection | DRNet | Menunjukkan kebutuhan representasi diskriminatif pada deteksi kategori yang saling berdekatan. |
| 16 | Wang et al. (2025) | WACV 2025 | Real-time end-to-end object detection | RT-DETRv3 | Menjadi dasar pilihan RT-DETRv3-R18 sebagai evaluasi lintas arsitektur yang bersifat tambahan. |
| 17 | **Penelitian yang Diusulkan** | — | Deteksi fine-grained cacat biji kopi | Prapemrosesan frekuensi-angular + YOLO26n, dengan CLAHE sebagai pembanding | Menganalisis variasi desain prapemrosesan pada citra masukan menggunakan dataset primer, lalu mengevaluasi kinerja deteksi dan biaya komputasi. |

Tabel tersebut memperlihatkan dua posisi utama penelitian: tantangan diskriminasi antarkelas pada deteksi cacat kopi dan penggunaan prapemrosesan/frekuensi sebelum detektor pada domain lain. Penelitian ini menguji hubungan keduanya melalui prapemrosesan frekuensi-angular sebelum YOLO26n pada dataset primer cacat biji kopi.
