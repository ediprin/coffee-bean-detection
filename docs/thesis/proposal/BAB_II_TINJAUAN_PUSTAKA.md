# BAB II
# TINJAUAN PUSTAKA

## 2.1 Biji Kopi Hijau dan Cacat Fisik Biji Kopi

Biji kopi hijau merupakan biji kopi yang belum melalui proses penyangraian dan masih dinilai berdasarkan karakteristik fisiknya. SNI 2907:2008 mengatur persyaratan mutu biji kopi Robusta dan Arabika, termasuk penggolongan mutu, jenis cacat fisik, cara pengujian, penandaan, dan pengemasan (Badan Standardisasi Nasional, 2008). Dalam konteks penelitian berbasis citra, karakteristik fisik tersebut menjadi informasi visual yang dapat digunakan untuk membedakan biji normal, biji cacat, dan material asing.

SNI 2907:2008 mencakup berbagai kondisi fisik seperti biji hitam, biji hitam sebagian, biji hitam pecah, biji coklat, biji pecah, biji muda, biji berlubang akibat serangga, kulit kopi, kulit tanduk, serta material asing seperti ranting, tanah, atau batu. Standar tersebut menggunakan sistem nilai cacat untuk menentukan mutu kopi. Namun, implementasi pada penelitian *computer vision* tidak selalu menggunakan keseluruhan kategori dan prosedur penilaian SNI. Label yang dipelajari model bergantung pada *taxonomy* dan protokol anotasi dataset yang digunakan.

Kesiman et al. (2023) mengembangkan dataset klasifikasi cacat biji kopi yang mengacu pada SNI 2907:2008. Pada proses pengumpulan, sampel diidentifikasi berdasarkan jenis cacat yang terdapat pada standar, sedangkan subset akhir untuk klasifikasi terdiri atas 17 kelas. Arwatchananukul et al. (2024) secara terpisah mengembangkan dataset Thai Arabica dengan 17 jenis cacat, sedangkan Bahy dan Rifai (2026) menggunakan 20 kategori fisik pada penelitian *object detection*. Perbedaan tersebut menunjukkan bahwa standar mutu menyediakan konteks dan terminologi cacat, sedangkan kelas yang diprediksi sistem ditentukan oleh dataset penelitian.

## 2.2 Inspeksi Mutu Biji Kopi

Identifikasi cacat biji kopi secara konvensional masih dapat dilakukan melalui inspeksi visual oleh manusia. Proses tersebut membutuhkan kemampuan membedakan bentuk, warna, tekstur, dan tanda cacat yang muncul pada permukaan biji. Kesiman et al. (2023) menjelaskan bahwa identifikasi jenis cacat secara manual membutuhkan waktu dan tenaga serta bergantung pada ketersediaan pekerja yang berpengalaman. Arwatchananukul et al. (2024) juga menempatkan ketergantungan pada tenaga manusia sebagai salah satu kendala dalam proses pemilahan biji kopi hijau.

Sebelum berkembangnya *deep learning*, otomasi inspeksi kopi telah dilakukan menggunakan pengolahan citra dan fitur yang dirancang secara manual. De Oliveira et al. (2016) menggunakan kondisi pengambilan citra terkontrol, kalibrasi warna, dan ruang warna CIE L*a*b* untuk mengukur karakteristik warna biji kopi hijau dan menggunakannya pada proses klasifikasi. Pendekatan seperti ini menunjukkan bahwa karakteristik visual biji kopi dapat diproses secara komputasional, tetapi keberhasilannya bergantung pada kondisi akuisisi dan representasi fitur yang telah ditentukan sebelumnya.

Perkembangan jaringan saraf konvolusional, Transformer, dan *object detector* memungkinkan representasi visual dipelajari langsung dari data. Pada penelitian modern, proses inspeksi tidak lagi terbatas pada klasifikasi satu objek, tetapi juga mencakup pendeteksian beberapa objek dalam satu citra, pengenalan berbagai kategori cacat, serta implementasi pada perangkat dengan keterbatasan komputasi. Oleh karena itu, *computer vision* menjadi salah satu pendekatan yang relevan untuk mendukung otomasi inspeksi mutu biji kopi.

## 2.3 Object Detection

*Object detection* merupakan tugas *computer vision* yang menentukan kategori sekaligus lokasi suatu objek pada citra. Lokasi objek umumnya direpresentasikan dalam bentuk *bounding box*, sedangkan kategori ditentukan melalui probabilitas atau skor kelas. Secara historis, metode *object detection* dapat dikelompokkan menjadi pendekatan *two-stage* dan *one-stage*.

Faster R-CNN merupakan contoh pendekatan *two-stage* yang menggunakan *Region Proposal Network* untuk menghasilkan kandidat lokasi objek sebelum kandidat tersebut diproses pada tahap klasifikasi dan regresi *bounding box* berikutnya (Ren et al., 2015). Sebaliknya, YOLO merumuskan deteksi sebagai prediksi langsung dari citra menuju lokasi objek dan probabilitas kelas dalam satu jaringan (Redmon et al., 2016). Pendekatan ini menjadi dasar perkembangan berbagai detector *one-stage* yang menekankan keseimbangan antara akurasi dan kecepatan inferensi.

Kesesuaian spasial antara *bounding box* prediksi dan *ground truth* dapat diukur menggunakan *Intersection over Union* (IoU). Untuk *bounding box* prediksi \(B_p\) dan *ground truth* \(B_g\), IoU dirumuskan sebagai:

\[
IoU(B_p,B_g)=\frac{|B_p\cap B_g|}{|B_p\cup B_g|}.
\]

Nilai IoU yang semakin tinggi menunjukkan tumpang tindih spasial yang semakin besar antara prediksi dan *ground truth*. Dalam evaluasi *object detection*, prediksi kelas, skor kepercayaan, dan kualitas lokalisasi digunakan secara bersama untuk menentukan benar atau salahnya suatu deteksi.

Klasifikasi dan lokalisasi merupakan dua tugas yang berbeda meskipun dilatih dalam satu sistem deteksi. Feng et al. (2021) melalui TOOD membahas adanya *task misalignment* antara kedua tugas tersebut, sedangkan Wu et al. (2020) menunjukkan bahwa representasi yang sesuai untuk klasifikasi tidak selalu identik dengan representasi yang paling sesuai untuk regresi lokasi. Jiang et al. (2018) juga menunjukkan bahwa *classification confidence* tidak sama dengan kualitas lokalisasi. Perbedaan ini menjadi dasar bahwa perubahan kinerja deteksi tidak selalu dapat ditafsirkan sebagai perubahan lokalisasi saja.

## 2.4 You Only Look Once (YOLO)

YOLO diperkenalkan oleh Redmon et al. (2016) dengan merumuskan *object detection* sebagai satu permasalahan regresi dari citra penuh menuju *bounding box* dan probabilitas kelas. Seluruh prediksi dilakukan melalui satu jaringan sehingga proses deteksi dapat dijalankan secara langsung tanpa memisahkan tahap pembentukan proposal dan klasifikasi menjadi pipeline yang terpisah.

Seiring perkembangannya, keluarga YOLO mengalami berbagai perubahan pada *backbone*, agregasi fitur, *detection head*, strategi assignment, *loss function*, dan mekanisme inferensi. Meskipun setiap generasi memiliki desain yang berbeda, karakteristik utama yang tetap dipertahankan adalah orientasi pada deteksi yang efisien dan dapat digunakan pada kebutuhan *real-time*.

Keluarga YOLO telah banyak digunakan pada inspeksi biji kopi. Gope et al. (2024) membandingkan beberapa varian YOLO pada deteksi biji kopi hijau dengan jumlah kelas yang relatif terbatas. Hong et al. (2026) menggunakan YOLOv10 sebagai dasar pengembangan sistem deteksi tujuh kategori cacat biji kopi. Pada jumlah kelas yang lebih besar, Bahy dan Rifai (2026) menerapkan YOLOv5s pada 20 kategori cacat fisik. Penelitian tersebut menunjukkan bahwa YOLO merupakan keluarga detector yang relevan untuk domain biji kopi, tetapi kinerja setiap kelas dapat berbeda ketika kategori yang harus dibedakan semakin rinci.

## 2.5 YOLO26

YOLO26 merupakan keluarga model *real-time vision* yang diperkenalkan oleh Jocher et al. (2026). Pada tugas *object detection*, YOLO26 menggunakan rancangan *dual-head* yang mendukung jalur inferensi end-to-end tanpa *Non-Maximum Suppression* (NMS) sebagai jalur utama. Paper YOLO26 juga menjelaskan perubahan pada mekanisme regresi *bounding box*, strategi assignment, dan proses optimisasi dibandingkan generasi sebelumnya.

Secara umum, arsitektur deteksi YOLO26 tetap menggunakan alur *backbone*, *neck*, dan *detection head*. Fitur pada beberapa skala diproses dan digabungkan sebelum prediksi dilakukan pada tingkat P3, P4, dan P5. Penggunaan beberapa tingkat fitur memungkinkan model menangani objek pada ukuran yang berbeda. Keluarga YOLO26 tersedia dalam beberapa skala model, sedangkan penelitian ini menggunakan varian YOLO26n sebagai detector utama.

Pada penelitian ini, YOLO26 diposisikan sebagai detector yang arsitektur utamanya dipertahankan. Pengembangan metode difokuskan pada pengolahan citra masukan sebelum citra diteruskan ke YOLO26. Dengan demikian, perubahan yang dianalisis berasal dari *preprocessing* citra, bukan dari penambahan modul pada *backbone*, *neck*, atau *detection head*.

## 2.6 Fine-Grained Object Detection

*Fine-grained recognition* membahas pengenalan kategori yang berada pada tingkat subordinat dan memiliki kemiripan visual yang tinggi. Pada kondisi ini, objek secara umum mempunyai bentuk yang serupa, sedangkan perbedaan kelas ditentukan oleh karakteristik yang lebih halus seperti tekstur, warna, pola lokal, atau struktur tertentu. Ketika permasalahan tersebut digabungkan dengan *object detection*, sistem harus mampu menentukan lokasi objek sekaligus membedakan subkategori yang memiliki kemiripan visual.

Xie et al. (2025) membahas *fine-grained object detection* sebagai permasalahan yang tidak hanya berkaitan dengan lokalisasi, tetapi juga membutuhkan representasi yang cukup diskriminatif untuk membedakan kategori yang berdekatan. Penelitian tersebut menunjukkan bahwa tugas klasifikasi dan lokalisasi dapat mempunyai kebutuhan representasi yang berbeda, sehingga peningkatan kemampuan diskriminasi menjadi bagian penting pada *fine-grained object detection*.

Karakteristik serupa ditemukan pada penelitian biji kopi. Kesiman et al. (2023) menunjukkan bahwa model yang diuji mengalami penurunan akurasi yang besar ketika jumlah kategori diperluas dari tiga kelas menjadi 17 kelas cacat. Jundullah et al. (2026) dan Hebert dan Alamsyah (2026) juga menunjukkan adanya perbedaan kinerja yang besar antarkelas pada deteksi cacat dengan jumlah kategori yang lebih banyak.

Temuan tersebut menunjukkan bahwa banyaknya kelas bukan satu-satunya faktor yang menentukan sifat *fine-grained*. Aspek yang lebih penting adalah kedekatan karakteristik visual antar kelas yang harus dipisahkan oleh model. Oleh karena itu, penelitian ini menempatkan kemampuan diskriminasi visual antarkategori cacat sebagai salah satu aspek utama yang dianalisis.

## 2.7 Preprocessing Citra untuk Object Detection

*Preprocessing* citra merupakan transformasi yang dilakukan terhadap citra sebelum citra diterima oleh model utama. Pada *object detection*, *preprocessing* dapat digunakan untuk memperbaiki kontras, menekan *noise*, mempertahankan detail, atau mengubah representasi sinyal agar informasi tertentu lebih mudah dimanfaatkan oleh detector. Karena dilakukan pada ruang input, *preprocessing* dapat dievaluasi secara terpisah dari perubahan arsitektur model.

Syauqi et al. (2025) menerapkan pipeline *preprocessing* pada deteksi cacat *white pepper* sebelum YOLOv8m. Pipeline tersebut menggunakan CLAHE sebagai komponen utama dan juga melibatkan *gamma correction*, *denoising*, serta *unsharp masking*. Pada setup penelitian mereka, penggunaan citra hasil *preprocessing* memberikan kinerja deteksi yang lebih tinggi dibandingkan citra asli. Chen et al. (2024) menggunakan kombinasi *wavelet-threshold denoising*, standardisasi citra, *bilateral filtering*, dan *Laplacian sharpening* sebelum YOLOv8 pada deteksi keretakan biji jagung. Kedua penelitian tersebut menunjukkan bahwa pengolahan citra sebelum detector dapat diperlakukan sebagai bagian eksperimen yang terpisah dari modifikasi model.

Pendekatan lain menggunakan *preprocessing* yang dipelajari bersama detector. Liu et al. (2022) melalui IA-YOLO menggunakan *differentiable image-processing filters* dengan parameter yang diprediksi secara adaptif dan dioptimalkan berdasarkan *detection loss*. Qin et al. (2022) melalui DENet memisahkan citra menjadi komponen frekuensi rendah dan tinggi menggunakan *Laplacian pyramid*, kemudian melakukan proses peningkatan citra sebelum hasil rekonstruksi diberikan kepada YOLO. Pendekatan tersebut menunjukkan bahwa kualitas hasil *preprocessing* sebaiknya dinilai berdasarkan manfaatnya terhadap tugas deteksi, bukan hanya berdasarkan kualitas visual bagi manusia.

Li et al. (2025) melalui FE-YOLO menggunakan pemrosesan domain Fourier sebelum YOLO pada kondisi *low-light*. Komponen amplitudo dan fase diproses melalui jaringan enhancement, kemudian hasilnya direkonstruksi kembali ke domain spasial sebelum diberikan kepada detector. Penelitian ini menjadi salah satu contoh bahwa pemrosesan Fourier dapat ditempatkan langsung sebelum model deteksi.

Berdasarkan penelitian tersebut, *preprocessing* sebelum *object detector* dapat berupa transformasi tetap maupun transformasi yang dipelajari. Penelitian ini berfokus pada *preprocessing* berbasis frekuensi-angular yang bekerja pada citra masukan sebelum YOLO26 dan tidak menambahkan jaringan enhancement terpisah yang memiliki parameter trainable.

## 2.8 Representasi Citra pada Domain Frekuensi

### 2.8.1 Discrete Fourier Transform dan Fast Fourier Transform

Citra digital dapat dipandang sebagai sinyal dua dimensi pada domain spasial. *Discrete Fourier Transform* (DFT) mengubah representasi tersebut ke domain frekuensi sehingga citra dinyatakan sebagai kombinasi komponen spektral. Untuk citra diskrit \(f(x,y)\) berukuran \(M\times N\), DFT dua dimensi dapat dituliskan sebagai:

\[
F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)
\exp\left[-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
\]

Rekonstruksi ke domain spasial dilakukan menggunakan inverse DFT:

\[
f(x,y)=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)
\exp\left[j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
\]

*Fast Fourier Transform* (FFT) merupakan algoritma yang digunakan untuk menghitung DFT secara lebih efisien. Pemrosesan berbasis Fourier memungkinkan suatu transformasi dilakukan pada representasi spektral, kemudian citra dikembalikan kembali ke domain spasial melalui transformasi invers.

Yang dan Soatto (2020) menggunakan manipulasi amplitudo Fourier pada *Fourier Domain Adaptation* dan merekonstruksi kembali citra menggunakan inverse transform. Li et al. (2025) menggunakan pola transformasi, pemrosesan spektral, dan rekonstruksi sebelum citra diteruskan ke YOLO. Xu et al. (2025) menggunakan DFT lokal pada patch untuk mempertahankan variasi frekuensi pada bagian-bagian citra yang berbeda.

### 2.8.2 Amplitudo dan Fase

Koefisien Fourier \(F(u,v)\) merupakan bilangan kompleks yang dapat dinyatakan sebagai:

\[
F(u,v)=R(u,v)+jI(u,v).
\]

Amplitudo dan fase dapat dihitung sebagai:

\[
A(u,v)=\sqrt{R^2(u,v)+I^2(u,v)},
\]

\[
\phi(u,v)=\operatorname{atan2}(I(u,v),R(u,v)).
\]

Amplitudo menunjukkan besar respons spektral pada suatu koordinat frekuensi, sedangkan fase berkaitan dengan susunan spasial struktur citra. Yang dan Soatto (2020) menunjukkan bahwa manipulasi amplitudo dapat dilakukan dengan mempertahankan fase sumber pada konteks *domain adaptation*. Li et al. (2025) juga memproses amplitudo dan fase secara khusus pada FE-YOLO. Pada LFDet, Xu et al. (2025) mengubah respons amplitudo berdasarkan distribusi frekuensi dan menggunakan fase asli pada proses rekonstruksi.

### 2.8.3 Representasi Radial dan Angular

Koordinat spektrum dua dimensi dapat dianalisis dalam bentuk polar. Untuk pusat spektrum \((u_c,v_c)\), radius dan sudut suatu koordinat frekuensi dapat dituliskan sebagai:

\[
r(u,v)=\sqrt{(u-u_c)^2+(v-v_c)^2},
\]

\[
\theta(u,v)=\operatorname{atan2}(v-v_c,u-u_c).
\]

Representasi radial mengelompokkan informasi berdasarkan jarak dari pusat spektrum, sedangkan representasi angular mengelompokkan informasi berdasarkan arah. Cao et al. (2019) menggunakan distribusi radial dan angular dari energi spektrum Fourier untuk menganalisis tekstur pada citra *remote sensing*. Distribusi radial digunakan untuk mengamati perubahan frekuensi dan skala tekstur, sedangkan distribusi angular digunakan untuk menggambarkan *directionality* atau arah dominan pola tekstur.

Zhang dan Tan (2003) juga menunjukkan bahwa distribusi orientasi pada domain spektral dapat digunakan sebagai ciri diskriminatif pada klasifikasi tekstur. Dengan demikian, informasi frekuensi dan arah menyediakan dua sudut pandang yang berbeda terhadap struktur citra.

Pada *fine-grained object detection*, Xu et al. (2025) menggunakan *Angular Frequency-Aware Block* (AFAB). Salah satu bagian AFAB, yaitu AFAB-2, menghitung distribusi densitas angular dari amplitudo Fourier pada patch lokal, kemudian menggunakan informasi tersebut untuk menyesuaikan respons spektral sebelum citra direkonstruksi kembali ke domain spasial. Mekanisme tersebut menunjukkan bahwa analisis angular dapat diterapkan pada tugas *fine-grained detection*, tetapi hasil pada citra pesawat tidak dapat langsung dianggap berlaku pada cacat biji kopi.

Dalam penelitian ini, istilah *frekuensi-angular* merujuk pada pemrosesan citra melalui representasi Fourier lokal dan analisis distribusi amplitudo berdasarkan arah. Istilah *angular* tidak merujuk pada orientasi *bounding box* atau *oriented object detection*.

### 2.8.4 Pemrosesan Frekuensi pada Computer Vision

Pemrosesan frekuensi telah digunakan pada berbagai posisi dalam pipeline *computer vision*. Pada ruang input, Yang dan Soatto (2020) memodifikasi amplitudo Fourier untuk *domain adaptation*, Li et al. (2025) menggunakan *Fourier enhancement* sebelum detector, dan Xu et al. (2025) menggunakan pemrosesan frekuensi lokal sebelum ekstraksi fitur utama pada LFDet. Ketiga penelitian tersebut menunjukkan bahwa citra dapat ditransformasikan dan direkonstruksi kembali sebelum diproses oleh model utama.

Pada ruang fitur, Chi et al. (2020) melalui *Fast Fourier Convolution* menggabungkan pemrosesan lokal dan spektral di dalam jaringan. Li et al. (2024) menggunakan transformasi domain frekuensi pada deteksi cacat permukaan, sedangkan Chen et al. (2025) mengembangkan *Frequency Dynamic Convolution* untuk memodulasi respons konvolusi secara adaptif pada tugas *dense prediction*. Walaupun sama-sama menggunakan domain frekuensi, metode-metode tersebut berbeda dari *preprocessing* karena operasi dilakukan pada fitur internal jaringan.

Berdasarkan literatur tersebut, domain frekuensi menyediakan mekanisme untuk mengolah informasi citra dari perspektif yang berbeda dengan domain spasial. Namun, literatur yang ditinjau belum memberikan dasar untuk menyimpulkan bahwa cacat biji kopi secara khusus memiliki *frequency signature* tertentu. Oleh sebab itu, penerapan pemrosesan frekuensi-angular pada penelitian ini diposisikan sebagai pendekatan yang perlu diuji secara empiris pada deteksi cacat biji kopi.

## 2.9 Visualisasi Aktivasi Model

Visualisasi aktivasi digunakan untuk membantu menginterpretasikan bagian representasi internal jaringan yang memberikan respons kuat terhadap suatu prediksi. Pada penelitian ini, visualisasi tersebut diposisikan sebagai analisis pendukung untuk membandingkan respons model YOLO26 tanpa *preprocessing* dan YOLO26 dengan *preprocessing* frekuensi-angular. Visualisasi tidak menggantikan evaluasi kuantitatif dan tidak digunakan sebagai bukti kausal tunggal mengenai fitur yang digunakan model.

Selvaraju et al. (2017) memperkenalkan *Gradient-weighted Class Activation Mapping* (Grad-CAM), yaitu metode yang menggunakan gradient dari target prediksi terhadap feature map pada layer konvolusional untuk membentuk peta aktivasi yang bersifat *class-discriminative*. Metode tersebut dapat digunakan tanpa mengubah arsitektur atau melakukan pelatihan ulang, tetapi penerapannya memerlukan target dan layer yang dapat didefinisikan dengan benar.

Muhammad dan Yeasin (2020) memperkenalkan *Eigen-CAM*, yang menghasilkan *class activation map* menggunakan komponen utama dari representasi fitur pada layer konvolusional. Pada paper primer, Eigen-CAM tidak bergantung pada backpropagation gradient maupun *class relevance score*. Karakteristik tersebut menjadikan Eigen-CAM kandidat utama untuk visualisasi respons pada penelitian ini, sedangkan Grad-CAM menjadi alternatif apabila target prediksi, layer target, dan aliran gradient pada YOLO26 dapat didefinisikan secara konsisten. Varian CAM lain dapat dipertimbangkan apabila kompatibilitas teknisnya telah diverifikasi.

Metode visualisasi akhir akan ditentukan setelah kompatibilitas teknis dengan implementasi YOLO26 diverifikasi. Agar perbandingan dapat ditafsirkan secara adil, metode visualisasi, layer target, ukuran input, dan prosedur normalisasi yang dipilih akan diterapkan secara sama pada model pembanding.

## 2.10 Penelitian Terkait

Penelitian yang relevan dapat dikelompokkan menjadi tiga bagian, yaitu penelitian deteksi cacat biji kopi, penelitian *preprocessing* sebelum detector, dan penelitian yang menggunakan pemrosesan frekuensi pada tugas *fine-grained* atau *object detection*. Ringkasan penelitian yang menjadi dasar posisi penelitian ditunjukkan pada Tabel 2.1.

### Tabel 2.1 Penelitian Terkait

| No. | Penulis dan Tahun | Sumber Publikasi/Venue | Fokus Penelitian | Metode/Model | Kontribusi terhadap Penelitian |
|---:|---|---|---|---|---|
| 1 | Hong et al. (2026) | *Current Research in Food Science* | Deteksi tujuh kategori cacat biji kopi | Improved YOLOv10 dengan modifikasi ekstraksi dan pemrosesan fitur | Menunjukkan keluarga YOLO dapat digunakan untuk deteksi cacat kopi dan bahwa kemiripan visual antarkategori tetap menjadi tantangan. |
| 2 | Gope et al. (2024) | *Scientific Reports* | Deteksi dan klasifikasi cacat biji kopi hijau | Perbandingan beberapa varian YOLO | Menunjukkan kelayakan keluarga YOLO pada domain biji kopi dengan jumlah kelas yang relatif terbatas. |
| 3 | Bahy dan Rifai (2026) | *International Journal on ICT* | Deteksi 20 kategori fisik berbasis SNI | Lightweight YOLOv5s | Menunjukkan adanya perbedaan kinerja antarkelas pada taxonomy yang lebih rinci. |
| 4 | Jundullah et al. (2026) | *Brilliance: Research of Artificial Intelligence* | Deteksi 20 kelas cacat dan kontaminan | YOLOv8s | Menunjukkan bahwa kinerja agregat dapat disertai ketimpangan kinerja antarkelas dan kesulitan pada kelas yang mirip secara visual. |
| 5 | Hebert dan Alamsyah (2026) | *INOVTEK Polbeng - Seri Informatika* | Deteksi 15 kategori cacat biji kopi | YOLOv12 | Menunjukkan bahwa beberapa kategori dengan tanda cacat yang halus memiliki kinerja deteksi lebih rendah. |
| 6 | Kesiman et al. (2023) | ICITRI 2023 | Klasifikasi cacat berbasis SNI | MobileNet dan InceptionResNetV2 | Menunjukkan bahwa peningkatan granularitas dari tiga kelas menjadi 17 kelas meningkatkan kesulitan diskriminasi. |
| 7 | Arwatchananukul et al. (2024) | *Smart Agricultural Technology* | Klasifikasi 17 jenis cacat biji kopi Arabika | Transfer learning CNN | Menunjukkan pentingnya pengujian pada data yang tidak terlihat sebelumnya pada klasifikasi fine-grained. |
| 8 | Hu et al. (2025) | *LWT* | Pengenalan cacat kopi dengan perbedaan visual halus | Siamese network | Menunjukkan bahwa pembelajaran berbasis kemiripan dapat digunakan untuk meningkatkan diskriminasi antarkelas. |
| 9 | Liu et al. (2022) | AAAI 2022 | Preprocessing adaptif untuk object detection pada cuaca buruk | IA-YOLO | Menunjukkan bahwa preprocessing dapat dioptimalkan berdasarkan kebutuhan tugas deteksi. |
| 10 | Syauqi et al. (2025) | IEEE ICONS-IoT 2025 | Deteksi cacat white pepper | CLAHE-based composite preprocessing + YOLOv8m | Memberikan contoh penerapan fixed preprocessing sebelum detector pada objek pertanian berbentuk biji. |
| 11 | Chen et al. (2024) | *Computers and Electronics in Agriculture* | Deteksi keretakan biji jagung | Image enhancement + YOLOv8 | Menunjukkan bahwa kontribusi preprocessing dapat dievaluasi secara terpisah dari optimasi detector. |
| 12 | Li et al. (2025) | *Digital Signal Processing* | Low-light object detection | Fourier enhancement + YOLO | Menunjukkan pemrosesan Fourier pada citra masukan sebelum detector. |
| 13 | Xu et al. (2025) | *Neural Networks* | Fine-grained aircraft detection | LFDet dengan AFAB | Menunjukkan penggunaan pemrosesan frekuensi lokal dan distribusi angular pada tugas fine-grained detection. |
| 14 | Xie et al. (2025) | *IEEE Transactions on Circuits and Systems for Video Technology* | Fine-grained object detection | DRNet | Menunjukkan kebutuhan representasi diskriminatif pada deteksi kategori yang saling berdekatan. |
| 15 | **Penelitian yang Diusulkan** | — | Deteksi fine-grained cacat biji kopi | Preprocessing citra berbasis frekuensi-angular + YOLO26 | Menganalisis dan mengoptimasi preprocessing pada citra masukan, kemudian mengevaluasi pengaruhnya terhadap kinerja deteksi dan biaya komputasi. |

Berdasarkan penelitian terkait tersebut, terdapat dua kecenderungan yang relevan. Pertama, penelitian biji kopi menunjukkan bahwa keluarga YOLO dapat digunakan untuk proses deteksi, tetapi kategori yang semakin rinci memperlihatkan perbedaan kinerja dan kesulitan diskriminasi antarkelas. Kedua, penelitian di luar domain kopi menunjukkan bahwa preprocessing citra dan pengolahan domain frekuensi dapat memengaruhi informasi yang diterima oleh detector. Penelitian ini menghubungkan kedua arah tersebut dengan menguji preprocessing citra berbasis frekuensi-angular sebelum YOLO26 pada deteksi fine-grained cacat biji kopi.