# BAB II
# TINJAUAN PUSTAKA

## 2.1 Biji Kopi Hijau, Cacat Fisik, dan Benda Asing

Biji kopi hijau merupakan biji kopi yang belum melalui proses penyangraian dan masih dinilai berdasarkan karakteristik fisiknya. SNI 2907:2008 mengatur persyaratan mutu biji kopi Robusta dan Arabika, termasuk penggolongan mutu, jenis cacat fisik, cara pengujian, penandaan, dan pengemasan (Badan Standardisasi Nasional, 2008). Dalam konteks penelitian berbasis citra, karakteristik fisik tersebut menjadi informasi visual yang dapat digunakan untuk membedakan biji normal, biji cacat, dan benda asing.

SNI 2907:2008 mencakup berbagai kondisi fisik seperti biji hitam, biji hitam sebagian, biji hitam pecah, biji coklat, biji pecah, biji muda, biji berlubang akibat serangga, kulit kopi, kulit tanduk, serta benda asing seperti ranting, tanah, atau batu. Standar tersebut menggunakan sistem nilai cacat untuk menentukan mutu kopi. Namun, penerapan pada penelitian *computer vision* tidak selalu menggunakan seluruh kategori dan prosedur penilaian SNI. Label yang dipelajari model bergantung pada susunan kelas dan protokol anotasi dataset yang digunakan.

Kesiman et al. (2023) mengembangkan dataset klasifikasi cacat biji kopi yang mengacu pada SNI 2907:2008. Pada proses pengumpulan, sampel diidentifikasi berdasarkan jenis cacat yang terdapat pada standar, sedangkan subset akhir untuk klasifikasi terdiri atas 17 kelas. Arwatchananukul et al. (2024) secara terpisah mengembangkan dataset Thai Arabica dengan 17 jenis cacat, sedangkan Bahy dan Rifai (2026) menggunakan 20 kategori fisik pada penelitian deteksi objek. Perbedaan tersebut menunjukkan bahwa standar mutu menyediakan konteks dan terminologi cacat, sedangkan kelas yang diprediksi sistem ditentukan oleh dataset penelitian.

Pada tugas deteksi objek, jumlah citra tidak dapat dipisahkan dari jumlah objek yang dianotasi di dalam setiap citra. Bahy dan Rifai (2026) menggunakan 107 citra sumber dan menghasilkan 13.863 anotasi pada deteksi 20 kategori fisik, sedangkan Tarekegn dan Debelee (2025) membangun dataset deteksi dari 562 citra dengan 19.228 objek pada 13 kelas cacat dan satu kelas normal. Pola tersebut menjadi salah satu dasar bahwa pengumpulan dataset primer pada penelitian deteksi perlu direncanakan berdasarkan jumlah citra sumber sekaligus jumlah objek pada setiap kelas.

## 2.2 Inspeksi Mutu Biji Kopi

Identifikasi cacat biji kopi secara konvensional masih dapat dilakukan melalui inspeksi visual oleh manusia. Proses tersebut membutuhkan kemampuan membedakan bentuk, warna, tekstur, dan tanda cacat yang muncul pada permukaan biji. Kesiman et al. (2023) menjelaskan bahwa identifikasi jenis cacat secara manual membutuhkan waktu dan tenaga serta bergantung pada ketersediaan pekerja yang berpengalaman. Arwatchananukul et al. (2024) juga menempatkan ketergantungan pada tenaga manusia sebagai salah satu kendala dalam proses pemilahan biji kopi hijau.

Sebelum berkembangnya *deep learning*, otomasi inspeksi kopi telah dilakukan menggunakan pengolahan citra dan fitur yang dirancang secara manual. De Oliveira et al. (2016) menggunakan kondisi pengambilan citra terkontrol, kalibrasi warna, dan ruang warna CIE L*a*b* untuk mengukur karakteristik warna biji kopi hijau dan menggunakannya pada proses klasifikasi. Pendekatan seperti ini menunjukkan bahwa karakteristik visual biji kopi dapat diproses secara komputasional, tetapi keberhasilannya bergantung pada kondisi akuisisi dan representasi fitur yang telah ditentukan sebelumnya.

Perkembangan jaringan saraf konvolusional, Transformer, dan model deteksi objek memungkinkan representasi visual dipelajari langsung dari data. Pada penelitian modern, proses inspeksi tidak lagi terbatas pada klasifikasi satu objek, tetapi juga mencakup pendeteksian beberapa objek dalam satu citra, pengenalan berbagai kategori cacat, serta penerapan pada perangkat dengan keterbatasan komputasi. Oleh karena itu, *computer vision* menjadi salah satu pendekatan yang relevan untuk mendukung otomasi inspeksi mutu biji kopi.

## 2.3 Deteksi Objek

Deteksi objek (*object detection*) merupakan tugas *computer vision* yang menentukan kategori sekaligus lokasi suatu objek pada citra. Lokasi objek umumnya direpresentasikan dalam bentuk kotak pembatas (*bounding box*), sedangkan kategori ditentukan melalui probabilitas atau skor kelas. Secara historis, metode deteksi objek dapat dikelompokkan menjadi pendekatan dua tahap (*two-stage*) dan satu tahap (*one-stage*).

Faster R-CNN merupakan contoh pendekatan dua tahap yang menggunakan *Region Proposal Network* untuk menghasilkan kandidat lokasi objek sebelum kandidat tersebut diproses pada tahap klasifikasi dan regresi kotak pembatas berikutnya (Ren et al., 2015). Sebaliknya, YOLO merumuskan deteksi sebagai prediksi langsung dari citra menuju lokasi objek dan probabilitas kelas dalam satu jaringan (Redmon et al., 2016). Pendekatan ini menjadi dasar perkembangan berbagai model deteksi satu tahap yang menekankan keseimbangan antara akurasi dan kecepatan inferensi.

Kesesuaian spasial antara kotak pembatas prediksi dan *ground truth* dapat diukur menggunakan *Intersection over Union* (IoU). Untuk kotak pembatas prediksi $B_p$ dan *ground truth* $B_g$, IoU dirumuskan sebagai:

$$
IoU(B_p,B_g)=\frac{|B_p\cap B_g|}{|B_p\cup B_g|}.
$$

Nilai IoU yang semakin tinggi menunjukkan tumpang tindih spasial yang semakin besar antara prediksi dan *ground truth*. Dalam evaluasi deteksi objek, prediksi kelas, skor kepercayaan, dan kualitas lokalisasi digunakan secara bersama untuk menentukan benar atau salahnya suatu deteksi.

Klasifikasi dan lokalisasi merupakan dua tugas yang berbeda meskipun dilatih dalam satu sistem deteksi. Feng et al. (2021) melalui TOOD membahas adanya ketidakselarasan tugas (*task misalignment*) antara kedua tugas tersebut, sedangkan Wu et al. (2020) menunjukkan bahwa representasi yang sesuai untuk klasifikasi tidak selalu identik dengan representasi yang paling sesuai untuk regresi lokasi. Jiang et al. (2018) juga menunjukkan bahwa tingkat keyakinan klasifikasi tidak sama dengan kualitas lokalisasi. Perbedaan ini menjadi dasar bahwa perubahan kinerja deteksi tidak selalu dapat ditafsirkan sebagai perubahan lokalisasi saja.

## 2.4 You Only Look Once (YOLO)

YOLO diperkenalkan oleh Redmon et al. (2016) dengan merumuskan deteksi objek sebagai satu permasalahan regresi dari citra penuh menuju kotak pembatas dan probabilitas kelas. Seluruh prediksi dilakukan melalui satu jaringan sehingga proses deteksi dapat dijalankan secara langsung tanpa memisahkan tahap pembentukan proposal dan klasifikasi menjadi rangkaian yang terpisah.

Seiring perkembangannya, keluarga YOLO mengalami berbagai perubahan pada *backbone*, agregasi fitur, *detection head*, strategi penetapan target, fungsi kerugian (*loss function*), dan mekanisme inferensi. Meskipun setiap generasi memiliki desain yang berbeda, karakteristik utama yang tetap dipertahankan adalah orientasi pada deteksi yang efisien dan dapat digunakan pada kebutuhan waktu nyata (*real-time*).

Keluarga YOLO telah banyak digunakan pada inspeksi biji kopi. Gope et al. (2024) membandingkan beberapa varian YOLO pada deteksi biji kopi hijau dengan jumlah kelas yang relatif terbatas. Hong et al. (2026) menggunakan YOLOv10 sebagai dasar pengembangan sistem deteksi tujuh kategori cacat biji kopi. Pada jumlah kelas yang lebih besar, Bahy dan Rifai (2026) menerapkan YOLOv5s pada 20 kategori fisik. Penelitian tersebut menunjukkan bahwa YOLO merupakan keluarga model deteksi yang relevan untuk domain biji kopi, tetapi kinerja setiap kelas dapat berbeda ketika kategori yang harus dibedakan semakin rinci.

## 2.5 YOLO26 dan Pembanding Arsitektur

YOLO26 merupakan keluarga model *real-time vision* yang diperkenalkan oleh Jocher et al. (2026). Pada tugas deteksi objek, YOLO26 menggunakan rancangan *dual-head* yang mendukung jalur inferensi end-to-end tanpa *Non-Maximum Suppression* (NMS) sebagai jalur utama. Paper YOLO26 juga menjelaskan perubahan pada mekanisme regresi kotak pembatas, strategi penetapan target, dan proses optimisasi dibandingkan generasi sebelumnya.

Secara umum, arsitektur deteksi YOLO26 tetap menggunakan alur *backbone*, *neck*, dan *detection head*. Fitur pada beberapa skala diproses dan digabungkan sebelum prediksi dilakukan pada tingkat P3, P4, dan P5. Penggunaan beberapa tingkat fitur memungkinkan model menangani objek pada ukuran yang berbeda. Keluarga YOLO26 tersedia dalam beberapa skala model, sedangkan penelitian ini menggunakan varian YOLO26n sebagai model deteksi utama.

Pada penelitian ini, YOLO26 diposisikan sebagai model deteksi yang arsitektur utamanya dipertahankan. Pengembangan metode difokuskan pada pengolahan citra masukan sebelum citra diteruskan ke YOLO26. Dengan demikian, perubahan yang dianalisis berasal dari prapemrosesan citra, bukan dari penambahan modul pada *backbone*, *neck*, atau *detection head*.

Sebagai pembanding keluarga arsitektur yang berbeda, RT-DETRv3 dapat digunakan pada analisis tambahan. Wang et al. (2025) mengembangkan RT-DETRv3 sebagai model deteksi Transformer end-to-end berbasis RT-DETR dengan tambahan supervisi positif yang lebih padat pada tahap pelatihan. Pada penelitian ini, varian R18 hanya direncanakan sebagai evaluasi transfer setelah konfigurasi prapemrosesan utama ditetapkan. Perbandingan tersebut tidak digunakan untuk menentukan apakah YOLO26n atau RT-DETRv3 lebih unggul, tetapi untuk melihat apakah arah pengaruh prapemrosesan tetap muncul ketika arsitektur model deteksi diganti.

## 2.6 Fine-Grained Object Detection

*Fine-grained recognition* membahas pengenalan kategori yang berada pada tingkat subordinat dan memiliki kemiripan visual yang tinggi. Pada kondisi ini, objek secara umum mempunyai bentuk yang serupa, sedangkan perbedaan kelas ditentukan oleh karakteristik yang lebih halus seperti tekstur, warna, pola lokal, atau struktur tertentu. Ketika permasalahan tersebut digabungkan dengan deteksi objek, sistem harus mampu menentukan lokasi objek sekaligus membedakan subkategori yang memiliki kemiripan visual.

Xie et al. (2025) membahas *fine-grained object detection* sebagai permasalahan yang tidak hanya berkaitan dengan lokalisasi, tetapi juga membutuhkan representasi yang cukup diskriminatif untuk membedakan kategori yang berdekatan. Penelitian tersebut menunjukkan bahwa tugas klasifikasi dan lokalisasi dapat mempunyai kebutuhan representasi yang berbeda, sehingga kemampuan diskriminasi menjadi bagian penting pada deteksi objek *fine-grained*.

Karakteristik serupa ditemukan pada penelitian biji kopi. Kesiman et al. (2023) menunjukkan bahwa model yang diuji mengalami penurunan akurasi yang besar ketika jumlah kategori diperluas dari tiga kelas menjadi 17 kelas cacat. Jundullah et al. (2026) dan Hebert dan Alamsyah (2026) juga menunjukkan adanya perbedaan kinerja yang besar antarkelas pada deteksi cacat dengan jumlah kategori yang lebih banyak.

Temuan tersebut menunjukkan bahwa banyaknya kelas bukan satu-satunya faktor yang menentukan sifat *fine-grained*. Aspek yang lebih penting adalah kedekatan karakteristik visual antarkelas yang harus dipisahkan oleh model. Tidak semua kelas pada suatu taksonomi multikelas harus sama sulitnya; kelas dengan bentuk atau warna yang khas dapat lebih mudah dikenali dibandingkan kelas yang hanya berbeda melalui tanda permukaan yang halus. Oleh karena itu, penelitian ini tetap mengevaluasi seluruh kelas yang memenuhi kecukupan data, tetapi analisis *fine-grained* terutama diarahkan pada kelas yang menunjukkan kemiripan visual dan kinerja yang lebih rendah.

## 2.7 Prapemrosesan Citra untuk Deteksi Objek

Prapemrosesan citra merupakan transformasi yang dilakukan terhadap citra sebelum citra diterima oleh model utama. Pada deteksi objek, prapemrosesan dapat digunakan untuk memperbaiki kontras, menekan *noise*, mempertahankan detail, atau mengubah representasi sinyal agar informasi tertentu lebih mudah dimanfaatkan oleh model deteksi. Karena dilakukan pada ruang masukan, prapemrosesan dapat dievaluasi secara terpisah dari perubahan arsitektur model.

Syauqi et al. (2025) menerapkan rangkaian prapemrosesan pada deteksi cacat *white pepper* sebelum YOLOv8m. Rangkaian tersebut melibatkan CLAHE bersama *gamma correction*, *denoising*, dan *unsharp masking*. Karena beberapa proses digunakan secara bersamaan, hasil penelitian tersebut tidak dapat digunakan untuk mengisolasi pengaruh CLAHE saja. Atas dasar itu, penelitian ini menggunakan CLAHE sebagai pembanding tunggal dengan konfigurasi tetap untuk menilai apakah peningkatan kontras lokal konvensional dapat menghasilkan perubahan kinerja tanpa analisis frekuensi-angular.

Chen et al. (2024) menggunakan kombinasi *wavelet-threshold denoising*, standardisasi citra, *bilateral filtering*, dan *Laplacian sharpening* sebelum YOLOv8 pada deteksi keretakan biji jagung. Penelitian tersebut menunjukkan bahwa pendekatan wavelet relevan sebagai alternatif transformasi multiskala. Namun, penerapannya memerlukan keputusan tambahan mengenai keluarga wavelet, tingkat dekomposisi, subband, ambang, dan rekonstruksi. Oleh karena itu, wavelet dibahas sebagai pendekatan pembanding konseptual tetapi tidak dijadikan pembanding utama yang ikut dioptimasi pada penelitian ini.

Pendekatan lain menggunakan prapemrosesan yang dipelajari bersama model deteksi. Liu et al. (2022) melalui IA-YOLO menggunakan *differentiable image-processing filters* dengan parameter yang diprediksi secara adaptif dan dioptimalkan berdasarkan *detection loss*. Qin et al. (2022) melalui DENet memisahkan citra menjadi komponen frekuensi rendah dan tinggi menggunakan *Laplacian pyramid*, kemudian melakukan proses peningkatan citra sebelum hasil rekonstruksi diberikan kepada YOLO. Pendekatan tersebut menunjukkan bahwa kualitas hasil prapemrosesan sebaiknya dinilai berdasarkan manfaatnya terhadap tugas deteksi, bukan hanya berdasarkan kualitas visual bagi manusia.

Li et al. (2025) melalui FE-YOLO menggunakan pemrosesan domain Fourier sebelum YOLO pada kondisi pencahayaan rendah. Komponen amplitudo dan fase diproses melalui jaringan peningkatan citra, kemudian hasilnya direkonstruksi kembali ke domain spasial sebelum diberikan kepada model deteksi. Penelitian ini menjadi salah satu contoh bahwa pemrosesan Fourier dapat ditempatkan langsung sebelum model deteksi.

Berdasarkan penelitian tersebut, prapemrosesan sebelum model deteksi dapat berupa transformasi tetap maupun transformasi yang dipelajari. Penelitian ini berfokus pada prapemrosesan frekuensi-angular tanpa parameter yang dilatih, menggunakan CLAHE sebagai pembanding peningkatan kontras, dan mempertahankan YOLO26n sebagai arsitektur utama yang tetap.

## 2.8 Representasi Citra pada Domain Frekuensi

### 2.8.1 Discrete Fourier Transform dan Fast Fourier Transform

Citra digital dapat dipandang sebagai sinyal dua dimensi pada domain spasial. *Discrete Fourier Transform* (DFT) mengubah representasi tersebut ke domain frekuensi sehingga citra dinyatakan sebagai kombinasi komponen spektral (Gonzalez & Woods, 2018). Bentuk DFT dan transformasi balik yang digunakan sebagai dasar penelitian juga dituliskan secara eksplisit oleh Xu et al. (2025, §3.1.1, Persamaan 1 dan 4). Untuk citra diskrit $f(x,y)$ berukuran $M\times N$, DFT dua dimensi dapat dituliskan sebagai:

$$
F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)
\exp\left[-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
$$

Rekonstruksi ke domain spasial dilakukan menggunakan inverse DFT:

$$
f(x,y)=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)
\exp\left[j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
$$

*Fast Fourier Transform* (FFT) merupakan algoritma yang digunakan untuk menghitung DFT secara lebih efisien. Pemrosesan berbasis Fourier memungkinkan suatu transformasi dilakukan pada representasi spektral, kemudian citra dikembalikan kembali ke domain spasial melalui transformasi invers (Gonzalez & Woods, 2018; Xu et al., 2025).

Yang dan Soatto (2020) menggunakan manipulasi amplitudo Fourier pada *Fourier Domain Adaptation* dan merekonstruksi kembali citra menggunakan transformasi invers. Li et al. (2025) menggunakan pola transformasi, pemrosesan spektral, dan rekonstruksi sebelum citra diteruskan ke YOLO. Xu et al. (2025) menggunakan DFT lokal pada patch untuk mempertahankan variasi frekuensi pada bagian-bagian citra yang berbeda.

### 2.8.2 Amplitudo dan Fase

Koefisien Fourier $F(u,v)$ merupakan bilangan kompleks yang dapat dinyatakan sebagai:

$$
F(u,v)=R(u,v)+jI(u,v).
$$

Amplitudo dan fase dapat dihitung sebagai:

$$
A(u,v)=\sqrt{R^2(u,v)+I^2(u,v)},
$$

$$
\phi(u,v)=\operatorname{atan2}(I(u,v),R(u,v)).
$$

Bentuk amplitudo dan fase tersebut sejalan dengan Persamaan (2) dan (3) pada Xu et al. (2025, §3.1.1). Secara umum, amplitudo menunjukkan besar respons spektral pada suatu koordinat frekuensi, sedangkan fase berkaitan dengan susunan spasial dalam representasi Fourier (Gonzalez & Woods, 2018). Yang dan Soatto (2020) menunjukkan bahwa manipulasi amplitudo dapat dilakukan dengan mempertahankan fase sumber pada konteks *domain adaptation*. Li et al. (2025) juga memproses amplitudo dan fase secara khusus pada FE-YOLO. Pada LFDet, Xu et al. (2025) mengubah respons amplitudo berdasarkan distribusi frekuensi dan menggunakan fase asli pada proses rekonstruksi.

### 2.8.3 Representasi Radial dan Angular

Koordinat spektrum dua dimensi dapat dianalisis dalam bentuk polar. Untuk pusat spektrum $(u_c,v_c)$, radius dan sudut suatu koordinat frekuensi dapat dituliskan sebagai:

$$
r(u,v)=\sqrt{(u-u_c)^2+(v-v_c)^2},
$$

$$
\theta(u,v)=\operatorname{atan2}(v-v_c,u-u_c).
$$

Representasi radial mengelompokkan informasi berdasarkan jarak dari pusat spektrum, sedangkan representasi angular mengelompokkan informasi berdasarkan arah. Cao et al. (2019) menggunakan distribusi radial dan angular dari energi spektrum Fourier untuk menganalisis tekstur pada citra *remote sensing*. Distribusi radial digunakan untuk mengamati perubahan frekuensi dan skala tekstur, sedangkan distribusi angular digunakan untuk menggambarkan arah dominan pola tekstur.

Zhang dan Tan (2003) juga menunjukkan bahwa distribusi orientasi pada domain spektral dapat digunakan sebagai ciri diskriminatif pada klasifikasi tekstur. Dengan demikian, informasi frekuensi dan arah menyediakan dua sudut pandang yang berbeda terhadap struktur citra.

Pada *fine-grained object detection*, Xu et al. (2025) mengembangkan *Adaptive Frequency Augmentation Branch* (AFAB). AFAB terdiri atas DFT berbasis patch, AFAB-1 berupa penyaring lolos-tinggi adaptif per patch, dan AFAB-2 berupa penekan amplitudo berdasarkan distribusi angular. Xu et al. (2025) juga menganalisis AFAB-1 dan AFAB-2 sebagai subkomponen yang dapat diuji secara terpisah. Pada AFAB-2, distribusi densitas angular dibentuk dari amplitudo Fourier pada patch lokal, kemudian digunakan untuk menyesuaikan amplitudo sebelum citra direkonstruksi kembali menggunakan fase asli.

Penelitian ini tidak mengadopsi keseluruhan LFDet maupun seluruh AFAB. Prinsip AFAB-2 dipilih sebagai konfigurasi referensi karena pertanyaan penelitian difokuskan pada pemrosesan distribusi frekuensi berdasarkan arah pada ruang citra masukan. Dengan demikian, AFAB-1, pemrosesan frekuensi pada ruang fitur melalui CGFI, dan interaksi teks-citra melalui FTIF tidak menjadi bagian dari metode utama penelitian. Pemindahan prinsip AFAB-2 dari citra pesawat ke citra cacat biji kopi tetap diperlakukan sebagai hipotesis yang harus diuji, bukan sebagai efektivitas yang telah terbukti pada domain kopi.

Dalam penelitian ini, istilah *frekuensi-angular* merujuk pada pemrosesan citra melalui representasi Fourier lokal dan analisis distribusi amplitudo berdasarkan arah. Istilah *angular* tidak merujuk pada orientasi kotak pembatas atau *oriented object detection*.

### 2.8.4 Pemrosesan Frekuensi pada Computer Vision

Pemrosesan frekuensi telah digunakan pada berbagai posisi dalam alur *computer vision*. Pada ruang masukan, Yang dan Soatto (2020) memodifikasi amplitudo Fourier untuk *domain adaptation*, Li et al. (2025) menggunakan *Fourier enhancement* sebelum model deteksi, dan Xu et al. (2025) menggunakan pemrosesan frekuensi lokal sebelum ekstraksi fitur utama pada LFDet. Ketiga penelitian tersebut menunjukkan bahwa citra dapat ditransformasikan dan direkonstruksi kembali sebelum diproses oleh model utama.

Pada ruang fitur, Chi et al. (2020) melalui *Fast Fourier Convolution* menggabungkan pemrosesan lokal dan spektral di dalam jaringan. Li et al. (2024) menggunakan transformasi domain frekuensi pada deteksi cacat permukaan, sedangkan Chen et al. (2025) mengembangkan *Frequency Dynamic Convolution* untuk memodulasi respons konvolusi secara adaptif pada tugas *dense prediction*. Walaupun sama-sama menggunakan domain frekuensi, metode-metode tersebut berbeda dari prapemrosesan karena operasi dilakukan pada fitur internal jaringan.

Berdasarkan literatur tersebut, domain frekuensi menyediakan mekanisme untuk mengolah informasi citra dari perspektif yang berbeda dengan domain spasial. Namun, literatur yang ditinjau belum memberikan dasar untuk menyimpulkan bahwa cacat biji kopi secara khusus memiliki *frequency signature* tertentu. Oleh sebab itu, penerapan pemrosesan frekuensi-angular pada penelitian ini diposisikan sebagai pendekatan yang perlu diuji secara empiris pada deteksi cacat biji kopi.

## 2.9 Visualisasi Aktivasi Model

Visualisasi aktivasi digunakan untuk membantu menginterpretasikan bagian representasi internal jaringan yang memberikan respons kuat terhadap suatu prediksi. Pada penelitian ini, visualisasi tersebut diposisikan sebagai analisis pendukung untuk membandingkan respons model YOLO26 tanpa prapemrosesan dan YOLO26 dengan prapemrosesan frekuensi-angular. Visualisasi tidak menggantikan evaluasi kuantitatif dan tidak digunakan sebagai bukti kausal tunggal mengenai fitur yang digunakan model.

Selvaraju et al. (2017) memperkenalkan *Gradient-weighted Class Activation Mapping* (Grad-CAM), yaitu metode yang menggunakan gradien dari target prediksi terhadap peta fitur pada lapisan konvolusional untuk membentuk peta aktivasi yang bersifat *class-discriminative*. Metode tersebut dapat digunakan tanpa mengubah arsitektur atau melakukan pelatihan ulang, tetapi penerapannya memerlukan target dan lapisan yang dapat didefinisikan dengan benar.

Muhammad dan Yeasin (2020) memperkenalkan *Eigen-CAM*, yang menghasilkan peta aktivasi menggunakan komponen utama dari representasi fitur pada lapisan konvolusional. Pada paper primer, Eigen-CAM tidak bergantung pada *backpropagation gradient* maupun *class relevance score*. Karakteristik tersebut menjadikan Eigen-CAM kandidat utama untuk visualisasi respons pada penelitian ini, sedangkan Grad-CAM menjadi alternatif apabila target prediksi, lapisan target, dan aliran gradien pada YOLO26 dapat didefinisikan secara konsisten. Varian CAM lain dapat dipertimbangkan apabila kompatibilitas teknisnya telah diverifikasi.

Metode visualisasi akhir akan ditentukan setelah kompatibilitas teknis dengan implementasi YOLO26 diverifikasi. Agar perbandingan dapat ditafsirkan secara adil, metode visualisasi, lapisan target, ukuran masukan, dan prosedur normalisasi yang dipilih akan diterapkan secara sama pada model pembanding.

## 2.10 Penelitian Terkait

Penelitian yang relevan dapat dikelompokkan menjadi tiga bagian, yaitu penelitian deteksi cacat biji kopi, penelitian prapemrosesan sebelum model deteksi, dan penelitian yang menggunakan pemrosesan frekuensi pada tugas *fine-grained* atau deteksi objek. Ringkasan penelitian yang menjadi dasar posisi penelitian ditunjukkan pada Tabel 2.1.

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

Berdasarkan penelitian terkait tersebut, terdapat dua kecenderungan yang relevan. Pertama, penelitian biji kopi menunjukkan bahwa keluarga YOLO dapat digunakan untuk proses deteksi, tetapi kategori yang semakin rinci memperlihatkan perbedaan kinerja dan kesulitan diskriminasi antarkelas. Kedua, penelitian di luar domain kopi menunjukkan bahwa prapemrosesan citra dan pengolahan domain frekuensi dapat memengaruhi informasi yang diterima oleh model deteksi. Penelitian ini menghubungkan kedua arah tersebut dengan menguji prapemrosesan citra berbasis frekuensi-angular sebelum YOLO26n pada dataset primer cacat biji kopi, menggunakan CLAHE sebagai pembanding peningkatan kontras konvensional dan mempertahankan evaluasi lintas arsitektur sebagai analisis tambahan.
