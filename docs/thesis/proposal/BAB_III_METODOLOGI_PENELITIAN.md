# BAB III
# METODOLOGI PENELITIAN

## 3.1 Arsitektur Umum Penelitian

Penelitian ini menggunakan pendekatan eksperimental komparatif untuk menganalisis pengaruh *preprocessing* citra berbasis frekuensi-angular terhadap kinerja YOLO26 pada deteksi *fine-grained* cacat biji kopi. Perbandingan dilakukan antara YOLO26n yang menerima citra asli dan YOLO26n yang menerima citra setelah melalui *preprocessing* frekuensi-angular. Arsitektur utama YOLO26 dipertahankan sehingga perubahan kinerja yang diamati dapat dikaitkan dengan perlakuan pada citra masukan.

Secara umum, tahapan penelitian terdiri atas persiapan dataset, pembentukan baseline YOLO26n, penerapan *preprocessing* frekuensi-angular, analisis dan optimasi rancangan *preprocessing*, pelatihan model, evaluasi kinerja deteksi, analisis visual, analisis kinerja per kelas dan kesalahan, serta evaluasi efisiensi komputasi.

Alur utama penelitian dirangkum sebagai berikut:

```text
Dataset biji kopi
        ↓
Persiapan dan pembagian data
        ↓
Baseline YOLO26n
        ↓
Preprocessing frekuensi-angular
        ↓
Analisis dan optimasi preprocessing
        ↓
Pelatihan YOLO26n
        ↓
Evaluasi kinerja deteksi
        ↓
Analisis visual dan analisis per kelas
        ↓
Evaluasi efisiensi komputasi
        ↓
Kesimpulan
```

Perbandingan utama penelitian dapat dinyatakan secara konseptual sebagai:

\[
\hat{Y}_{N}=\operatorname{YOLO26n}(I),
\]

untuk model tanpa *preprocessing*, dan:

\[
I'=\mathcal{P}_{FA}(I),
\]

\[
\hat{Y}_{P}=\operatorname{YOLO26n}(I'),
\]

dengan \(I\) merupakan citra asli, \(\mathcal{P}_{FA}\) merupakan fungsi *preprocessing* frekuensi-angular, \(I'\) merupakan citra hasil *preprocessing*, dan \(\hat{Y}\) merupakan hasil prediksi deteksi.

## 3.2 Dataset Penelitian

### 3.2.1 Sumber dan Karakteristik Dataset

Penelitian menggunakan dataset *object detection* biji kopi hijau dengan total 21 kelas. Setiap objek pada citra memiliki anotasi *bounding box* dan label kelas. Dataset digunakan untuk melatih dan mengevaluasi kemampuan model dalam menentukan lokasi sekaligus kategori objek pada citra.

Data pengembangan yang digunakan terdiri atas 1.665 citra pada bagian *training* dengan 2.986 anotasi dan 294 citra pada bagian *validation* dengan 526 anotasi. Seluruh kelas target terdapat pada kedua bagian data tersebut. Ringkasan dataset ditunjukkan pada Tabel 3.1.

### Tabel 3.1 Ringkasan Dataset Penelitian

| Bagian Data | Jumlah Citra | Jumlah Anotasi | Jumlah Kelas |
|---|---:|---:|---:|
| Training | 1.665 | 2.986 | 21 |
| Validation | 294 | 526 | 21 |

Konteks SNI 01-2907-2008 digunakan untuk menjelaskan relevansi cacat fisik biji kopi, sedangkan kelas yang dipelajari model mengikuti label yang tersedia pada dataset penelitian. Sistem yang dikembangkan tidak dimaksudkan untuk merekonstruksi keseluruhan prosedur penentuan mutu berdasarkan nilai cacat SNI.

### 3.2.2 Pembagian Dataset dan Pencegahan Kebocoran Data

Pembagian dataset dilakukan secara *grouped split* untuk mengurangi kemungkinan citra yang berasal dari sumber atau induk yang sama tersebar pada bagian *training* dan *validation*. Pemeriksaan juga dilakukan terhadap duplikasi citra yang identik agar data yang sama tidak muncul pada kedua bagian.

Pembagian data dipertahankan sama pada seluruh konfigurasi yang dibandingkan. Dengan demikian, perbedaan kinerja tidak berasal dari perubahan komposisi data antara baseline dan metode yang menggunakan *preprocessing*.

### 3.2.3 Augmentasi Data

Augmentasi yang digunakan merupakan bagian dari pipeline pelatihan YOLO26 dan diterapkan dengan konfigurasi yang sama pada seluruh model yang dibandingkan. *Preprocessing* frekuensi-angular tidak diperlakukan sebagai augmentasi karena tidak menghasilkan label baru dan tidak mengubah posisi objek melalui translasi, rotasi geometris, atau *warping* koordinat *bounding box*.

## 3.3 Model Dasar YOLO26n

YOLO26n digunakan sebagai model dasar pada penelitian ini. YOLO26 merupakan keluarga *real-time object detector* yang diperkenalkan oleh Jocher et al. (2026). Varian nano dipilih sebagai detector utama agar eksperimen tetap menggunakan model dengan kompleksitas komputasi relatif rendah sekaligus mempertahankan mekanisme deteksi multi-skala pada P3, P4, dan P5.

Model menggunakan bobot *pretrained* resmi sebagai inisialisasi awal. Karena dataset penelitian mempunyai 21 kelas, bagian prediksi kelas disesuaikan dengan jumlah kelas target. Seluruh kondisi inisialisasi, konfigurasi pelatihan, dan pembagian data dibuat sama pada baseline dan model dengan *preprocessing* sehingga perbedaan utama berada pada citra masukan.

Pada penelitian ini tidak dilakukan modifikasi terhadap *backbone*, *neck*, maupun *detection head* YOLO26n. Hal ini dilakukan untuk mengisolasi pengaruh *preprocessing* frekuensi-angular terhadap kinerja detector.

## 3.4 Preprocessing Citra Berbasis Frekuensi-Angular

*Preprocessing* yang digunakan mengadaptasi prinsip pemrosesan frekuensi lokal dan analisis distribusi angular pada AFAB-2 yang diperkenalkan Xu et al. (2025) untuk *fine-grained object detection*. Mekanisme tersebut tidak diterapkan sebagai salinan keseluruhan arsitektur LFDet, tetapi diadaptasi menjadi *preprocessing* pada citra masukan sebelum YOLO26.

Tahapan utama *preprocessing* meliputi pembentukan patch lokal, transformasi Fourier, pembentukan distribusi angular, penentuan ambang adaptif, pembobotan respons spektral, inverse Fourier transform, rekonstruksi patch, dan penggabungan residual dengan citra asli.

### 3.4.1 Pembentukan Patch Lokal

Untuk citra masukan \(I\), citra dibagi menjadi patch lokal berukuran \(m\times m\). Konfigurasi awal menggunakan ukuran patch:

\[
m=32.
\]

Patch dibentuk dengan overlap 50%, sehingga untuk ukuran patch 32 piksel digunakan *stride* 16 piksel. Penggunaan patch lokal bertujuan agar analisis frekuensi tidak hanya menggambarkan karakteristik global citra, tetapi juga mempertahankan variasi lokal pada bagian-bagian citra yang berbeda.

### 3.4.2 Transformasi Fourier

Setiap patch \(P_i\) ditransformasikan ke domain frekuensi menggunakan transformasi Fourier dua dimensi:

\[
F_i(u,v)=\mathcal{F}\{P_i\}(u,v).
\]

Koefisien Fourier kemudian dipisahkan menjadi amplitudo dan fase:

\[
A_i(u,v)=|F_i(u,v)|,
\]

\[
\phi_i(u,v)=\arg F_i(u,v).
\]

Amplitudo digunakan untuk menganalisis besar respons pada setiap koordinat frekuensi, sedangkan fase dipertahankan pada proses rekonstruksi.

### 3.4.3 Distribusi Angular

Setiap koordinat frekuensi dipetakan ke sudut berdasarkan posisi relatifnya terhadap pusat spektrum. Sudut dapat dinyatakan sebagai:

\[
\theta(u,v)=\operatorname{atan2}(v-v_c,u-u_c),
\]

dengan \((u_c,v_c)\) merupakan pusat spektrum.

Pada konfigurasi awal, domain angular dibagi menjadi 360 bin. Amplitudo pada koordinat yang termasuk pada bin arah yang sama dijumlahkan sehingga diperoleh densitas angular:

\[
D_i^c(k)=\sum_{(u,v):b(u,v)=k}A_i^c(u,v),
\]

dengan \(c\) menunjukkan kanal warna dan \(k\) menunjukkan indeks bin angular.

Densitas kemudian dinormalisasi menjadi distribusi probabilitas:

\[
p_i^c(k)=\frac{D_i^c(k)}{\sum_jD_i^c(j)+\varepsilon}.
\]

### 3.4.4 Ambang Adaptif Berdasarkan Entropi

Entropi distribusi angular dihitung untuk menggambarkan penyebaran respons pada setiap patch:

\[
H_i^c=-\sum_k p_i^c(k)\log\left(p_i^c(k)+\varepsilon\right).
\]

Nilai entropi digunakan untuk membentuk ambang adaptif:

\[
\tau_i^c=\frac{\gamma}{1+\exp(-H_i^c)},
\]

dengan \(\gamma\) merupakan parameter pengatur ambang. Konfigurasi awal menggunakan \(\gamma=0{,}10\). Karena nilai \(H_i^c\) dihitung dari patch yang sedang diproses, ambang berubah mengikuti karakteristik spektral citra meskipun tidak memiliki parameter trainable.

### 3.4.5 Pembobotan Respons Spektral

Densitas angular dinormalisasi terhadap respons maksimum:

\[
q_i^c(k)=\frac{D_i^c(k)}{\max_jD_i^c(j)+\varepsilon}.
\]

Pada konfigurasi awal digunakan pembobotan dengan ambang keras:

\[
w_i^c(k)=
\begin{cases}
0, & q_i^c(k)\le\tau_i^c,\\
q_i^c(k), & q_i^c(k)>\tau_i^c.
\end{cases}
\]

Bobot tersebut dipetakan kembali ke koordinat Fourier sehingga diperoleh spektrum yang telah disesuaikan:

\[
\widetilde F_i^c(u,v)=F_i^c(u,v)\,w_i^c(b(u,v)).
\]

### 3.4.6 Inverse Fourier Transform dan Rekonstruksi Citra

Spektrum yang telah dibobotkan dikembalikan ke domain spasial menggunakan inverse Fourier transform:

\[
\widetilde P_i=\Re\left\{\mathcal{F}^{-1}(\widetilde F_i)\right\}.
\]

Patch yang saling overlap kemudian digabungkan melalui perataan pada area yang bertumpang tindih sehingga diperoleh respons spasial \(R_{FA}(I)\). Respons tersebut dinormalisasi menggunakan *min-max normalization*:

\[
G(I)=\operatorname{MinMax}\left(R_{FA}(I)\right).
\]

Citra hasil *preprocessing* dibentuk melalui residual enhancement:

\[
I'=I+I\odot G(I).
\]

Operasi tersebut mempertahankan ukuran spasial citra sehingga anotasi *bounding box* tidak perlu diubah akibat proses *preprocessing*.

## 3.5 Analisis dan Optimasi Preprocessing

Kata "optimasi" pada penelitian ini merujuk pada analisis sistematis terhadap faktor rancangan *preprocessing*, bukan pada penambahan banyak modul secara bertumpuk. Setiap variasi dibandingkan terhadap konfigurasi referensi dengan mengubah satu faktor utama pada satu waktu. Pendekatan ini digunakan agar pengaruh setiap keputusan desain dapat dianalisis secara lebih jelas.

Faktor yang dianalisis ditunjukkan pada Tabel 3.2.

### Tabel 3.2 Faktor Rancangan Preprocessing yang Dianalisis

| Faktor | Konfigurasi Referensi | Variasi yang Dianalisis | Tujuan Analisis |
|---|---|---|---|
| Windowing patch | Rectangular window | Square-root Hann dengan normalized overlap-add | Menganalisis pengaruh spectral leakage pada batas patch |
| Representasi arah | 360 bin angular | 16 orientasi modulo \(\pi\) | Menganalisis kebutuhan resolusi dan redundansi arah |
| Struktur spektral | Angular | 3 band radial × 16 orientasi | Menganalisis kontribusi informasi radial dan angular secara bersama |
| Fungsi ambang | Hard threshold | Soft threshold | Menganalisis pengaruh transisi ambang terhadap respons lemah |
| Pemrosesan warna | Setiap kanal RGB | Gate berbasis luminance Rec.709 yang dibagi antar kanal | Menganalisis kebutuhan informasi spektral spesifik tiap kanal warna |

Setelah faktor struktural dianalisis, dilakukan *sensitivity analysis* terbatas terhadap parameter utama yang paling relevan, terutama ukuran patch dan koefisien ambang \(\gamma\). Nilai kandidat pada analisis sensitivitas ditetapkan sebelum hasil eksperimen digunakan untuk mengambil keputusan agar pemilihan parameter tidak dilakukan secara retrospektif.

Konfigurasi yang dipilih selanjutnya digunakan pada perbandingan utama dengan YOLO26n tanpa *preprocessing*.

## 3.6 Rancangan Eksperimen

Eksperimen disusun dalam dua tahap. Tahap pertama digunakan untuk menganalisis faktor rancangan *preprocessing* dan memilih konfigurasi yang digunakan. Tahap kedua digunakan untuk membandingkan konfigurasi terpilih dengan YOLO26n tanpa *preprocessing*.

Pada seluruh perbandingan, komponen berikut dijaga sama:

1. dataset dan pembagian data;
2. bobot *pretrained* YOLO26n;
3. ukuran input;
4. augmentasi data;
5. jumlah epoch;
6. batch size;
7. optimizer dan konfigurasi pelatihan;
8. kondisi perangkat keras dan perangkat lunak.

Perbandingan utama dilakukan pada tiga seed, yaitu 42, 123, dan 2026, agar kesimpulan tidak hanya bergantung pada satu kondisi acak. Untuk setiap seed, baseline dan model dengan *preprocessing* menggunakan kondisi pelatihan yang sama.

## 3.7 Konfigurasi Pelatihan

Konfigurasi awal pelatihan ditunjukkan pada Tabel 3.3.

### Tabel 3.3 Konfigurasi Pelatihan YOLO26n

| Parameter | Nilai |
|---|---:|
| Model | YOLO26n |
| Inisialisasi | Pretrained |
| Ukuran input | 640 × 640 piksel |
| Epoch maksimum | 50 |
| Batch size | 16 |
| Workers | 2 |
| Patience | 15 |
| Optimizer | Auto |
| Cache | False |
| Close mosaic | 10 |
| Maximum detection | 500 |
| Seed utama | 42, 123, 2026 |

Konfigurasi tersebut diterapkan secara konsisten pada baseline dan model dengan *preprocessing*. Perubahan hanya dilakukan pada faktor yang memang menjadi objek analisis penelitian.

## 3.8 Evaluasi Kinerja Deteksi

Evaluasi dilakukan menggunakan metrik yang umum digunakan pada *object detection*. *Precision* menyatakan proporsi prediksi positif yang benar, sedangkan *recall* menyatakan proporsi objek *ground truth* yang berhasil dideteksi. Kedua metrik dirumuskan sebagai:

\[
Precision=\frac{TP}{TP+FP},
\]

\[
Recall=\frac{TP}{TP+FN}.
\]

*Average Precision* (AP) menghitung luas di bawah kurva *precision-recall* pada suatu kelas. Evaluasi utama menggunakan mAP50 dan mAP50–95 mengikuti prinsip evaluasi *object detection* COCO (Lin et al., 2014). mAP50 dihitung pada IoU 0,50, sedangkan mAP50–95 merupakan rata-rata AP pada beberapa threshold IoU dari 0,50 sampai 0,95.

Selain nilai agregat, penelitian menggunakan rata-rata AP50–95 per kelas sebagai indikator utama agar setiap kelas memperoleh bobot yang setara. Analisis juga dilakukan terhadap AP setiap kelas untuk mengidentifikasi kategori dengan kinerja tinggi maupun rendah.

Untuk menggambarkan bagian bawah distribusi performa, digunakan dua ringkasan tambahan, yaitu rata-rata tiga kelas dengan AP50–95 terendah dan nilai AP50–95 kelas terendah. Kedua ukuran ini digunakan sebagai analisis tambahan untuk mengetahui apakah perubahan metode hanya meningkatkan nilai agregat atau juga memengaruhi kelas-kelas yang paling sulit.

## 3.9 Analisis Visual

Analisis visual dilakukan sebagai pendukung evaluasi kuantitatif untuk membantu menginterpretasikan perubahan yang terjadi pada citra, respons spektral, dan prediksi model setelah *preprocessing*. Pola ini mengadaptasi penggunaan visualisasi sebagai analisis pendukung pada penelitian Hong et al. (2026), tetapi visualisasi dalam penelitian ini tidak diperlakukan sebagai bukti kausal tunggal mengenai alasan peningkatan atau penurunan kinerja model.

### 3.9.1 Visualisasi Tahapan Preprocessing

Visualisasi pertama berfokus pada transformasi citra sebelum masuk ke YOLO26. Untuk contoh citra yang dipilih, panel visual akan menampilkan secara berurutan:

1. citra asli;
2. patch lokal yang dianalisis;
3. magnitude spektrum Fourier;
4. distribusi angular;
5. ambang adaptif dan respons angular yang dipertahankan;
6. respons hasil inverse Fourier transform;
7. citra hasil rekonstruksi dan residual enhancement.

Visualisasi ini digunakan untuk menunjukkan bagaimana operasi frekuensi-angular mengubah representasi citra secara transparan. Perubahan kontras, tekstur, atau respons spektral yang terlihat tidak langsung dianggap sebagai bukti bahwa citra menjadi lebih baik bagi detector; interpretasinya tetap harus dikaitkan dengan hasil evaluasi kuantitatif.

### 3.9.2 Visualisasi Respons Model

Apabila kompatibel secara teknis dengan implementasi YOLO26 yang digunakan, penelitian akan menerapkan metode visualisasi berbasis aktivasi untuk membandingkan wilayah citra yang memberikan respons kuat pada model tanpa *preprocessing* dan model dengan *preprocessing*. Metode visualisasi yang digunakan akan ditetapkan setelah kompatibilitasnya dengan arsitektur YOLO26 diverifikasi dan diterapkan secara sama pada kedua model.

Visualisasi respons model digunakan sebagai alat interpretasi untuk melihat apakah terdapat perubahan pola perhatian atau aktivasi pada area objek dan karakteristik cacat. Hasil visualisasi tidak digunakan sebagai pengganti metrik deteksi dan tidak ditafsirkan sebagai bukti kausal bahwa model menggunakan fitur tertentu secara eksklusif.

### 3.9.3 Visualisasi Prediksi Deteksi

Hasil prediksi YOLO26 tanpa *preprocessing* dan YOLO26 dengan *preprocessing* dibandingkan pada citra yang sama. Visualisasi mencakup *bounding box*, label kelas, dan skor kepercayaan sehingga perubahan prediksi dapat diamati secara langsung.

Contoh visual dipilih berdasarkan kriteria yang telah ditentukan, misalnya kelas dengan kinerja tinggi, kelas dengan kinerja rendah, kasus ketika kedua model benar, kasus ketika kedua model salah, serta kasus ketika hasil prediksi kedua model berbeda. Pemilihan tersebut dilakukan untuk mengurangi kecenderungan hanya menampilkan contoh yang mendukung metode yang diusulkan.

## 3.10 Analisis Kesalahan dan Kinerja Per Kelas

Analisis kesalahan dilakukan menggunakan hasil prediksi per kelas, *confusion matrix*, *false positive*, dan *false negative*. Perbandingan antara baseline dan model dengan *preprocessing* digunakan untuk melihat kelas yang mengalami peningkatan, kelas yang relatif tetap, dan kelas yang mengalami penurunan.

Hasil analisis kesalahan kemudian dihubungkan dengan visualisasi pada Subbab 3.9 untuk menelaah kasus-kasus yang berubah setelah *preprocessing*. Analisis ini bersifat deskriptif dan digunakan untuk melengkapi hasil metrik agregat maupun per kelas.

## 3.11 Evaluasi Efisiensi Komputasi

Meskipun *preprocessing* yang digunakan tidak menambahkan parameter trainable, operasi patch, FFT, analisis angular, inverse FFT, dan rekonstruksi tetap menambah biaya komputasi. Oleh karena itu, evaluasi tidak hanya dilakukan terhadap akurasi deteksi.

Efisiensi diukur menggunakan:

1. jumlah parameter model;
2. latency inferensi;
3. throughput dalam citra per detik; dan
4. penggunaan memori GPU.

Pengukuran baseline dan model dengan *preprocessing* dilakukan pada perangkat, ukuran input, batch size, dan presisi komputasi yang sama agar hasil dapat dibandingkan secara adil.

## 3.12 Lingkungan Implementasi

Implementasi penelitian menggunakan Python dan framework PyTorch melalui Ultralytics YOLO. Informasi versi library, perangkat GPU, CUDA, sistem operasi, dan konfigurasi perangkat keras dicatat pada saat eksperimen dilakukan. Pencatatan lingkungan implementasi dilakukan untuk menjaga keterulangan eksperimen dan memudahkan verifikasi hasil penelitian.