# BAB III
# METODOLOGI PENELITIAN

## 3.1 Rancangan Umum Penelitian

Penelitian ini menggunakan pendekatan eksperimen komparatif untuk menganalisis pengaruh prapemrosesan citra berbasis frekuensi-angular terhadap kinerja YOLO26n dalam mendeteksi cacat biji kopi yang memiliki perbedaan visual yang halus (*fine-grained*). Arsitektur utama YOLO26n dipertahankan tanpa perubahan pada perbandingan utama sehingga perbedaan antar kondisi eksperimen terutama berasal dari perlakuan terhadap citra masukan.

Secara umum, penelitian meliputi pengumpulan dataset primer, anotasi dan pemeriksaan kecukupan data per kelas, pembagian data pelatihan, validasi, dan pengujian, pembentukan model acuan YOLO26n, penerapan konfigurasi referensi prapemrosesan frekuensi-angular, pengujian variasi desain prapemrosesan beserta analisis sensitivitas terbatas, pemilihan konfigurasi \(C^*\), pengujian ulang beberapa *seed* terhadap YOLO26n tanpa prapemrosesan, CLAHE, konfigurasi referensi, dan konfigurasi terpilih, evaluasi akhir pada data uji, serta analisis per kelas, kesalahan, visual, dan efisiensi komputasi. Evaluasi pada arsitektur deteksi lain ditempatkan sebagai analisis tambahan apabila sumber daya penelitian memungkinkan.

Alur penelitian dirangkum pada Gambar 3.1.

![Alur penelitian](docs/thesis/proposal/assets/alur_penelitian.svg){width=12.5cm}

Gambar 3.1 Alur Penelitian

Perbandingan utama dapat dinyatakan secara konseptual sebagai:

\[
\hat{Y}_{N}=\operatorname{YOLO26n}(I),
\]

untuk YOLO26n tanpa prapemrosesan tambahan, dan:

\[
I'=\mathcal{P}_{FA}(I),
\]

\[
\hat{Y}_{P}=\operatorname{YOLO26n}(I'),
\]

dengan \(I\) merupakan citra masukan, \(\mathcal{P}_{FA}\) merupakan fungsi prapemrosesan frekuensi-angular, \(I'\) merupakan citra hasil prapemrosesan, dan \(\hat{Y}\) merupakan hasil prediksi deteksi. Persamaan tersebut hanya menunjukkan perbedaan pada jalur masukan dan tidak mengasumsikan bahwa prapemrosesan selalu meningkatkan kinerja.

## 3.2 Dataset Penelitian

### 3.2.1 Sumber, Target Jumlah, dan Karakteristik Dataset Primer

Penelitian direncanakan menggunakan **dataset primer** yang dikumpulkan secara langsung untuk tugas deteksi objek multikelas pada biji kopi hijau. Daftar kelas awal menargetkan 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008 ditambah satu kelas biji normal, sehingga jumlah kelas target awal adalah:

\[
C_{target}=21.
\]

Jumlah kelas akhir akan ditetapkan setelah kecukupan data pada setiap kelas diperiksa dan sebelum pembagian data serta pelatihan model dilakukan. Dengan demikian, penelitian tidak mengasumsikan bahwa seluruh kelas yang langka pasti dapat diperoleh dalam jumlah memadai.

Berbeda dengan dataset klasifikasi yang menggunakan satu biji pada satu citra, setiap citra pada penelitian ini direncanakan memuat banyak objek. Oleh karena itu, kecukupan dataset dinilai dari dua hal, yaitu jumlah citra sumber yang berbeda dan jumlah objek yang diberi anotasi pada setiap kelas. Target pengumpulan ditetapkan sekitar 180–220 citra sumber, dengan target nominal sekitar 200 citra asli. Citra hasil augmentasi tidak dihitung sebagai data primer.

Setiap citra direncanakan memuat sekitar 30–50 objek yang disusun dalam satu lapisan, dengan orientasi yang bervariasi dan tanpa tumpang tindih berat. Dengan rancangan tersebut, jumlah anotasi objek ditargetkan berada pada kisaran:

\[
N_{box}\approx 6.000-10.000.
\]

Untuk setiap kelas yang dipertahankan, ditargetkan tersedia sekurang-kurangnya sekitar 200 objek asli per kelas, dengan sasaran ideal sekitar 300–500 objek per kelas. Selain jumlah objek, setiap kelas diupayakan muncul pada sedikitnya sekitar 30 citra sumber yang berbeda. Target kemunculan pada beberapa citra sumber diperlukan agar objek dari suatu kelas tidak terkonsentrasi hanya pada sedikit foto dan agar kelas tersebut dapat diwakili pada data pelatihan, validasi, dan pengujian.

Angka tersebut merupakan target perencanaan pengumpulan, bukan jumlah data yang diklaim telah tersedia pada saat proposal disusun. Sebagai pembanding dari penelitian terdahulu, Bahy dan Rifai (2026) melaporkan 107 citra dengan 13.863 anotasi untuk deteksi 20 kelas SNI, sedangkan Tarekegn dan Debelee (2025) menggunakan 562 citra dengan 19.228 objek untuk 13 kelas cacat dan satu kelas normal. Hal ini menunjukkan bahwa pada tugas deteksi objek, jumlah citra perlu dipertimbangkan bersama jumlah objek yang terdapat di dalam setiap citra.

Pengambilan citra akan dilakukan secara tegak lurus dari atas menggunakan latar belakang polos dan tidak reflektif, posisi kamera tetap, jarak kamera tetap, serta pencahayaan yang dikendalikan. Biji kopi disusun dalam satu lapisan agar detail permukaan tetap terlihat. Orientasi biji tetap divariasikan agar data mencakup kemungkinan sisi dan arah biji yang berbeda.

Setiap sesi pengambilan citra akan memiliki identitas sesi dan identitas citra sumber. Untuk kelas yang definisinya bergantung pada ukuran fisik, khususnya benda asing berukuran kecil, sedang, dan besar, pengaturan kamera akan dilengkapi referensi skala sehingga ukuran objek dapat ditelusuri secara konsisten.

Setiap objek akan diberi kotak pembatas (*bounding box*) dan label kelas. Definisi operasional tiap kelas akan disusun sebelum anotasi dengan mengacu pada SNI dan referensi visual yang digunakan. Sampel yang secara visual meragukan tidak akan langsung dimasukkan ke kelas tertentu, tetapi akan ditandai untuk ditinjau kembali. Validasi label direncanakan melibatkan praktisi atau validator yang memahami penilaian fisik mutu kopi, terutama pada kelas yang memiliki kemiripan visual tinggi.

### 3.2.2 Pemeriksaan Kecukupan Data dan Penetapan Kelas

Sebelum data dibagi menjadi pelatihan, validasi, dan pengujian, jumlah objek dan jumlah citra sumber pada setiap kelas akan diperiksa. Langkah ini diperlukan karena penelitian terdahulu berbasis SNI menunjukkan bahwa beberapa kategori cacat dapat sangat langka atau sulit dibedakan secara konsisten hanya dari citra RGB.

Suatu kelas akan dipertahankan sebagai kelas evaluasi utama apabila jumlah datanya memadai. Batas awal yang direncanakan adalah sekitar 200 objek asli dan kemunculan pada sedikitnya sekitar 30 citra sumber yang berbeda. Kekurangan data pada suatu kelas tidak akan ditutupi hanya dengan memperbanyak hasil augmentasi dari sejumlah kecil citra asli. Jika suatu kelas belum memenuhi batas tersebut, pengumpulan data akan ditambah atau susunan kelas akan disesuaikan sebelum pelatihan utama dilakukan.

Setelah pemeriksaan selesai, jumlah kelas akhir dinotasikan sebagai:

\[
C\le C_{target},
\]

dengan target utama tetap \(C=21\) apabila seluruh kelas memenuhi jumlah data minimum yang telah ditetapkan.

### 3.2.3 Pembagian Data dan Pencegahan Kebocoran

Pembagian data dilakukan pada citra sumber asli **sebelum augmentasi**. Proporsi awal yang direncanakan adalah sekitar 70% untuk pelatihan, 15% untuk validasi, dan 15% untuk pengujian. Dengan target sekitar 200 citra sumber, proporsi tersebut setara secara kasar dengan sekitar 140 citra pelatihan, 30 citra validasi, dan 30 citra pengujian.

Pembagian dilakukan berdasarkan kelompok sumber atau sesi pengambilan citra. Citra yang berasal dari sumber, sesi, atau susunan objek yang sangat berkaitan akan ditempatkan pada bagian data yang sama agar tidak terjadi kebocoran informasi antara pelatihan, validasi, dan pengujian. Apabila biji atau spesimen fisik yang sama difoto lebih dari satu kali, seluruh citra yang memuat spesimen tersebut ditempatkan pada bagian data yang sama. Dengan demikian, model tidak dilatih menggunakan satu tampilan dari spesimen tertentu lalu diuji pada tampilan lain dari spesimen fisik yang sama.

Selain menjaga pemisahan kelompok, pembagian juga mempertimbangkan distribusi kelas. Pembagian tidak dilakukan hanya dengan mengacak kelompok sumber, tetapi mencari susunan kelompok yang tetap saling terpisah sekaligus mendekati proporsi keseluruhan dan memenuhi keterwakilan kelas. Untuk setiap kelas akhir, diupayakan tersedia sekurang-kurangnya lima citra sumber pada data validasi dan lima citra sumber pada data pengujian, sedangkan bagian lainnya digunakan pada data pelatihan. Proporsi 70:15:15 diperlakukan sebagai sasaran keseluruhan dan dapat bergeser sedikit apabila diperlukan untuk memenuhi keterwakilan kelas tanpa melanggar pemisahan kelompok sumber.

Secara umum, kelompok sumber pada ketiga bagian data harus saling terpisah:

\[
\mathcal{G}_{train}\cap\mathcal{G}_{val}
=\mathcal{G}_{train}\cap\mathcal{G}_{test}
=\mathcal{G}_{val}\cap\mathcal{G}_{test}
=\varnothing.
\]

Pemeriksaan citra identik juga akan dilakukan menggunakan nilai *hash*. Seluruh hasil augmentasi dari suatu citra sumber hanya boleh berada pada bagian pelatihan dan tidak boleh masuk ke data validasi atau pengujian.

Data validasi digunakan untuk penghentian dini, pengujian variasi prapemrosesan, dan pemilihan konfigurasi. Data uji disisihkan sejak awal dan tidak digunakan untuk memilih ukuran patch, nilai \(\gamma\), konfigurasi prapemrosesan, maupun keputusan metodologis lainnya. Data uji baru digunakan setelah konfigurasi dan prosedur evaluasi akhir ditetapkan.

### 3.2.4 Augmentasi Data

Augmentasi hanya diterapkan pada bagian pelatihan setelah pembagian citra sumber selesai. Data validasi dan pengujian tetap menggunakan citra asli tanpa augmentasi sintetis. Konfigurasi augmentasi akan dibuat sama untuk seluruh kondisi YOLO26n yang dibandingkan.

CLAHE dan prapemrosesan frekuensi-angular tidak diperlakukan sebagai augmentasi karena keduanya merupakan perlakuan terhadap citra masukan pada kondisi eksperimen tertentu. Pada pelatihan, augmentasi YOLO dilakukan terlebih dahulu. Setelah prapemrosesan dasar YOLO membentuk tensor RGB bertipe *floating point* dan menormalkan rentang masukan dasar ke sekitar \([0,1]\), perlakuan CLAHE atau prapemrosesan frekuensi-angular diterapkan pada antarmuka tepat sebelum citra diteruskan ke YOLO26n. Prapemrosesan frekuensi-angular menerima tensor ter-normalisasi tersebut secara langsung. Jika implementasi CLAHE memerlukan representasi luminansi integer, konversi tersebut dilakukan hanya di dalam *frontend* CLAHE dan hasilnya dikembalikan lagi ke citra RGB bertipe *floating point* pada skala masukan yang setara sebelum diteruskan ke model. Posisi konseptual kedua prapemrosesan dibuat sama agar perbandingan tidak dipengaruhi oleh perbedaan urutan terhadap augmentasi. Kedua prapemrosesan tidak mengubah geometri kotak pembatas.

## 3.3 Model Dasar YOLO26n

YOLO26n digunakan sebagai model dasar karena berukuran relatif ringan dan tetap mendukung deteksi pada beberapa skala. Model akan menggunakan bobot pralatih (*pretrained*) resmi `yolo26n.pt` sebagai sumber inisialisasi. Setelah jumlah kelas akhir ditetapkan, bagian keluaran model disesuaikan dengan jumlah kelas \(C\).

Pada setiap perbandingan, model akan menggunakan sumber bobot pralatih dan kondisi inisialisasi yang sama. Kondisi awal model akan diperiksa sebelum pelatihan agar perbedaan utama antarperlakuan berasal dari prapemrosesan citra, bukan dari perbedaan bobot awal.

Bagian utama arsitektur YOLO26n, yaitu *backbone*, *neck*, dan *detection head*, tidak dimodifikasi pada eksperimen utama. Prapemrosesan yang dianalisis juga tidak menambahkan parameter yang dilatih. Dengan demikian, penelitian difokuskan pada perubahan representasi citra masukan.

### 3.3.1 Model Acuan dan Pembanding

Empat kondisi utama direncanakan sebagai berikut:

| Kode | Kondisi | Peran dalam eksperimen |
|---|---|---|
| \(B_0\) | YOLO26n tanpa prapemrosesan tambahan | Model acuan |
| \(B_1\) | CLAHE + YOLO26n | Pembanding peningkatan kontras lokal |
| \(B_2\) | \(C_0\) + YOLO26n | Konfigurasi referensi frekuensi-angular |
| \(B_3\) | \(C^*\) + YOLO26n | Konfigurasi frekuensi-angular terpilih |

CLAHE (*Contrast Limited Adaptive Histogram Equalization*) digunakan sebagai pembanding peningkatan citra konvensional. Tujuannya adalah menilai apakah perubahan kinerja yang diperoleh dari prapemrosesan frekuensi-angular juga dapat dicapai hanya dengan peningkatan kontras lokal. CLAHE akan diterapkan pada kanal luminansi dan hasilnya dikembalikan ke citra RGB sehingga geometri citra tetap sama.

Agar CLAHE tidak menjadi metode kedua yang ikut dituning secara terpisah, hanya satu konfigurasi tetap yang digunakan, yaitu *clip limit* 2,0 dan ukuran kisi 8 × 8. Nilai tersebut ditetapkan sebelum eksperimen pembanding sebagai konfigurasi kontrol dan tidak dianggap sebagai nilai optimum untuk citra biji kopi. Konfigurasi CLAHE tidak akan diubah berdasarkan hasil data uji.

Wavelet tidak dimasukkan sebagai pembanding utama. Meskipun relevan sebagai pendekatan transformasi multiskala, penerapan wavelet memerlukan keputusan tambahan mengenai jenis wavelet, tingkat dekomposisi, subband, ambang, dan rekonstruksi. Penambahan seluruh keputusan tersebut dapat memperluas ruang penelitian di luar fokus utama. Jika diperlukan pada tahap evaluasi akademik, satu konfigurasi wavelet tetap dapat ditambahkan sebagai analisis tambahan tanpa pencarian parameter yang luas.

Sebagai analisis tambahan, konfigurasi \(C^*\) dapat diuji pada RT-DETRv3-R18 untuk melihat apakah pengaruh prapemrosesan juga muncul pada keluarga model deteksi yang berbeda. Analisis ini bersifat opsional dan tidak digunakan untuk memilih konfigurasi utama penelitian.

## 3.4 Prapemrosesan Citra Berbasis Frekuensi-Angular

Prapemrosesan yang digunakan mengadaptasi prinsip pemrosesan frekuensi lokal dan distribusi angular pada AFAB-2 yang diperkenalkan oleh Xu et al. (2025). Penelitian ini tidak mengadopsi keseluruhan LFDet ataupun AFAB-1, tetapi menggunakan mekanisme angular AFAB-2 sebagai konfigurasi referensi prapemrosesan sebelum citra diteruskan ke YOLO26n.

Batas adaptasi metode ditetapkan secara jelas. Dari Xu et al. (2025), penelitian mengacu pada pembentukan respons frekuensi lokal melalui DFT per patch, distribusi densitas angular, entropi untuk membentuk ambang adaptif, penekanan arah dengan densitas rendah, pembobotan amplitudo, rekonstruksi menggunakan fase asli, serta penggabungan antara ruang spasial asal dan hasil rekonstruksi melalui normalisasi, perkalian elemen, dan residual. Sementara itu, pemrosesan per kanal RGB, diskretisasi sudut menjadi 360 interval, tumpang tindih patch 50%, konstanta stabilitas numerik, aturan penambahan tepi, dan cara penggabungan patch yang bertumpang tindih merupakan keputusan adaptasi penelitian agar mekanisme tersebut dapat digunakan sebagai prapemrosesan mandiri sebelum YOLO26n. Variasi pada \(C_1\) sampai \(C_5\) merupakan rancangan eksperimen penelitian dan bukan bagian dari AFAB-2 asli.

Secara umum, proses terdiri atas pembentukan patch lokal, transformasi Fourier, pembentukan distribusi angular, perhitungan ambang berbasis entropi, pembobotan respons spektral, transformasi balik ke domain spasial, rekonstruksi patch, normalisasi respons, dan penggabungan residual dengan citra masukan.

Arsitektur integrasi metode ditunjukkan pada Gambar 3.2. Panel (a) memperlihatkan bahwa prapemrosesan frekuensi-angular ditempatkan sebagai *frontend* pada ruang masukan sebelum YOLO26n, sementara *backbone*, *neck*, dan *detection head* YOLO26n dipertahankan. Panel (b) merangkum alur internal *frontend* mulai dari analisis spektral lokal sampai rekonstruksi dan penggabungan residual. Gambar tersebut menggambarkan kerangka umum prapemrosesan; variasi \(C_1\) sampai \(C_5\) merupakan alternatif keputusan desain yang diuji secara bertahap dan tidak dimaksudkan untuk menyatakan bahwa seluruh variasi tersebut pasti menjadi konfigurasi akhir.

![Arsitektur integrasi prapemrosesan frekuensi-angular dengan YOLO26n](docs/thesis/proposal/assets/arsitektur_frekuensi_yolo26.svg){width=12.5cm}

Gambar 3.2 Arsitektur Integrasi Prapemrosesan Frekuensi-Angular dengan YOLO26n

### 3.4.1 Pembentukan Patch Lokal

Untuk citra RGB:

\[
I\in\mathbb{R}^{3\times H\times W},
\]

citra dibagi menjadi potongan lokal atau *patch*:

\[
P_i\in\mathbb{R}^{3\times m\times m}.
\]

Konfigurasi referensi menggunakan ukuran:

\[
m=32,
\]

dengan tumpang tindih 50%, sehingga jarak perpindahan antarpatch adalah:

\[
s=16.
\]

Ukuran patch 32 mengikuti konfigurasi yang digunakan Xu et al. (2025) sebagai titik awal. Paper tersebut menyatakan penggunaan tumpang tindih yang besar untuk mengurangi diskontinuitas pada batas patch, tetapi tidak menetapkan 50% sebagai satu-satunya nilai pada definisi metodenya. Oleh karena itu, tumpang tindih 50% pada penelitian ini diperlakukan sebagai keputusan implementasi awal, bukan sebagai nilai yang diasumsikan optimal untuk citra kopi.

Pembagian lokal digunakan agar informasi frekuensi tetap dapat dikaitkan dengan bagian tertentu dari citra, bukan hanya menggambarkan spektrum global. Pada konfigurasi referensi \(C_0\), apabila ukuran citra tidak tepat memenuhi susunan patch, nilai piksel tepi diulang menggunakan *replicate padding* pada sisi yang diperlukan hingga grid patch dapat dibentuk. Setelah rekonstruksi selesai, hasil dipotong kembali ke ukuran asli \(H\times W\). Aturan ini ditetapkan sebagai keputusan implementasi agar perlakuan pada batas citra dapat direproduksi secara konsisten.

### 3.4.2 Transformasi Fourier

Untuk patch ke-\(i\) dan kanal warna ke-\(c\), transformasi Fourier dua dimensi dihitung sebagai:

\[
F_i^c(u,v)=\mathcal{F}_2\{P_i^c\}(u,v).
\]

Implementasi menggunakan normalisasi Fourier ortonormal dan memusatkan spektrum dengan operasi *FFT shift* sebelum geometri frekuensi dihitung. Koefisien Fourier kemudian dinyatakan dalam amplitudo dan fase:

\[
A_i^c(u,v)=|F_i^c(u,v)|,
\]

\[
\phi_i^c(u,v)=\arg F_i^c(u,v).
\]

Spektrum dipusatkan sehingga distribusi frekuensi dapat dianalisis berdasarkan posisi relatif terhadap pusat spektrum. Amplitudo digunakan untuk membentuk distribusi spektral, sedangkan fase koefisien yang dipertahankan tidak diubah secara eksplisit. Pada rekonstruksi, pusat spektrum dikembalikan melalui *inverse FFT shift* sebelum transformasi Fourier balik dilakukan.

### 3.4.3 Distribusi Angular

Setiap koordinat frekuensi dipetakan ke sudut relatif terhadap pusat spektrum:

\[
\theta(u,v)=\operatorname{mod}\left(\operatorname{atan2}(v-v_c,u-u_c),2\pi\right).
\]

Xu et al. (2025) mendefinisikan distribusi angular pada rentang \([0,360^\circ)\). Pada konfigurasi referensi penelitian ini, domain tersebut didiskretkan menjadi 360 interval arah. Densitas angular untuk kanal \(c\) dihitung sebagai:

\[
D_i^c(k)=\sum_{(u,v):b(u,v)=k}A_i^c(u,v),
\]

dengan \(b(u,v)\) menunjukkan interval sudut ke-\(k\). Densitas tersebut kemudian dinormalisasi menjadi:

\[
p_i^c(k)=\frac{D_i^c(k)}{\sum_jD_i^c(j)+\varepsilon},
\]

dengan:

\[
\varepsilon=10^{-8}.
\]

Pada konfigurasi referensi, perhitungan dilakukan secara terpisah pada kanal R, G, dan B. Diskretisasi 360 interval, konstanta \(\varepsilon\), dan pemrosesan per kanal merupakan keputusan implementasi penelitian. Koordinat pusat spektrum, yaitu titik dengan radius nol, secara diskret dipetakan ke interval pertama untuk menjaga aturan indeks yang terdefinisi pada seluruh grid. Aturan tersebut merupakan keputusan diskretisasi dan tidak dimaksudkan sebagai interpretasi bahwa komponen DC memiliki arah fisik tertentu.

### 3.4.4 Ambang Adaptif Berdasarkan Entropi

Entropi distribusi angular dihitung sebagai:

\[
H_i^c=-\sum_k p_i^c(k)\log\left(\max(p_i^c(k),\varepsilon)\right).
\]

Nilai entropi digunakan untuk membentuk ambang pada setiap patch:

\[
\tau_i^c=\frac{\gamma}{1+\exp(-H_i^c)},
\]

dengan konfigurasi referensi:

\[
\gamma=0{,}10.
\]

Bentuk entropi dan ambang mengacu pada AFAB-2 Xu et al. (2025). Nilai \(\gamma=0{,}10\) digunakan sebagai konfigurasi referensi karena nilai tersebut digunakan pada eksperimen paper sumber setelah analisis sensitivitas pada dataset pesawat; penelitian ini tidak menganggap nilai tersebut otomatis optimal untuk biji kopi. Pengaruh \(\gamma\) tetap dapat diperiksa melalui analisis sensitivitas pada Subbab 3.5.6.

Karena \(H_i^c\ge0\), fungsi tersebut membatasi ambang pada rentang:

\[
\frac{\gamma}{2}\le\tau_i^c<\gamma.
\]

Dengan \(\gamma=0{,}10\), ambang berada pada rentang sekitar 0,05 sampai kurang dari 0,10. Dengan demikian, entropi menyesuaikan ambang di dalam rentang yang dibatasi oleh \(\gamma\), bukan menghasilkan ambang tanpa batas. Proses ini tetap tidak menggunakan parameter yang dilatih.

### 3.4.5 Pembobotan Respons Spektral

Densitas angular dinormalisasi terhadap respons maksimum:

\[
q_i^c(k)=\frac{D_i^c(k)}{\max_jD_i^c(j)+\varepsilon}.
\]

Pada konfigurasi referensi digunakan ambang keras:

\[
w_i^c(k)=
\begin{cases}
0, & q_i^c(k)\le\tau_i^c,\\
q_i^c(k), & q_i^c(k)>\tau_i^c.
\end{cases}
\]

Bobot tersebut diterapkan pada koefisien Fourier:

\[
\widetilde F_i^c(u,v)=F_i^c(u,v)\,w_i^c(b(u,v)).
\]

Bentuk seleksi ini mengadaptasi mekanisme AFAB-2 yang mempertahankan respons dengan densitas relatif di atas ambang dan menekan respons di bawah ambang. Karena nilai bobot berada pada rentang 0 sampai 1, tahap ini berfungsi memilih dan menekan respons spektral tertentu, bukan memperbesar koefisien Fourier melebihi nilai asalnya.

### 3.4.6 Rekonstruksi dan Penggabungan Residual

Spektrum yang telah dibobotkan dikembalikan ke domain spasial menggunakan transformasi Fourier balik dengan normalisasi yang sama seperti transformasi maju:

\[
\widetilde P_i^c=\Re\left\{\mathcal{F}_2^{-1}(\widetilde F_i^c)\right\}.
\]

Prinsip AFAB-2 mempertahankan fase asli saat amplitudo yang telah disesuaikan direkonstruksi ke domain spasial. Pada konfigurasi referensi penelitian ini, patch yang saling bertumpang tindih kemudian digabungkan kembali dengan merata-ratakan bagian yang bertumpang tindih sehingga diperoleh respons spasial \(R_{FA}\) dengan ukuran yang sama seperti citra masukan. Cara penggabungan patch tersebut merupakan keputusan implementasi penelitian.

Respons tersebut dinormalisasi secara min–maks pada setiap citra dan setiap kanal:

\[
G^c(x,y)=\frac{R_{FA}^c(x,y)-r_{min}^c}{\max(r_{max}^c-r_{min}^c,\varepsilon)},
\]

sehingga:

\[
G^c(x,y)\in[0,1].
\]

Citra keluaran dibentuk melalui penggabungan residual:

\[
\boxed{I'^c=I^c+I^c\odot G^c}.
\]

Normalisasi hasil rekonstruksi, perkalian dengan ruang spasial asal, dan operasi residual mengikuti prinsip penggabungan yang dijelaskan Xu et al. (2025), sedangkan formulasi per kanal pada persamaan di atas merupakan bentuk implementasi penelitian. Tidak dilakukan *clipping* tambahan setelah operasi residual. Jika tensor masukan dasar berada pada rentang \([0,1]\), maka karena \(G^c\in[0,1]\), keluaran secara teoritis dapat berada pada rentang \([0,2]\). Ukuran spasial citra tetap dipertahankan sehingga koordinat kotak pembatas tidak perlu diubah.

## 3.5 Analisis Variasi Desain Prapemrosesan

Pada penelitian ini, optimasi dilakukan dengan membandingkan beberapa variasi desain prapemrosesan yang telah ditentukan terlebih dahulu. Proses ini tidak dimaksudkan sebagai pencarian *global optimum*. Pengujian dilakukan secara bertahap dan kumulatif: setiap tahap menambahkan satu perubahan desain, sedangkan perubahan yang telah diterapkan pada tahap sebelumnya tetap dipertahankan.

Urutan konfigurasi dinyatakan sebagai:

\[
C_0\rightarrow C_1\rightarrow C_2\rightarrow C_3\rightarrow C_4\rightarrow C_5.
\]

Ringkasan perubahan pada setiap tahap ditunjukkan pada Tabel 3.2.

### Tabel 3.2 Variasi Desain Prapemrosesan

| Konfigurasi | Perubahan utama | Tujuan pengujian |
|---|---|---|
| \(C_0\) | Konfigurasi frekuensi-angular referensi | Menjadi acuan prapemrosesan |
| \(C_1\) | Fungsi jendela Hann | Menguji pengaruh batas patch |
| \(C_2\) | Orientasi tak bertanda dengan resolusi sudut tetap | Menguji arah dan orientasi tanpa mengubah resolusi angular |
| \(C_3\) | Tiga pita radial | Menguji informasi frekuensi radial rendah, menengah, dan tinggi secara operasional |
| \(C_4\) | Ambang lunak | Menguji pembobotan yang lebih bertahap di sekitar ambang |
| \(C_5\) | Panduan luminansi | Menguji kebutuhan pembobotan terpisah pada setiap kanal warna |

Perbedaan antara dua konfigurasi yang berurutan digunakan untuk melihat pengaruh tambahan dari perubahan yang baru diperkenalkan. Pendekatan ini tidak dimaksudkan sebagai eksperimen faktorial lengkap terhadap seluruh kombinasi faktor.

### 3.5.1 Variasi Fungsi Jendela

Pada \(C_1\), patch persegi biasa diganti dengan fungsi jendela Hann akar kuadrat periodik (*periodic square-root Hann window*). Fungsi satu dimensinya dinyatakan sebagai:

\[
h[n]=\sqrt{\frac{1}{2}-\frac{1}{2}\cos\left(\frac{2\pi n}{m}\right)}.
\]

Jendela dua dimensi dibentuk sebagai:

\[
W[p,q]=h[p]h[q].
\]

Pada tahap analisis, patch dikalikan dengan jendela sebelum FFT:

\[
P_{i,a}=P_i\odot W.
\]

Setelah transformasi Fourier balik, jendela yang sama diterapkan kembali pada patch hasil rekonstruksi. Patch kemudian digabungkan menggunakan *normalized overlap-add*, yaitu jumlah respons pada setiap posisi dibagi dengan jumlah bobot \(W^2\) yang menutup posisi tersebut. Dengan demikian, penggunaan jendela tidak dibiarkan mengubah amplitudo hanya karena suatu piksel berada pada daerah tumpang tindih yang berbeda.

Karena square-root Hann bernilai nol pada sampel tepi tertentu, pada \(C_1\) dan konfigurasi setelahnya digunakan konteks *replicate padding* di sekitar citra sebelum pembentukan patch agar piksel asli tidak hanya direkonstruksi dari titik jendela bernilai nol. Setelah *overlap-add*, bagian konteks tersebut dipotong kembali ke ukuran citra asli.

Secara sederhana, fungsi Hann mengurangi kontribusi pada tepi patch secara bertahap sehingga perubahan mendadak pada batas patch dapat dikurangi. Variasi ini akan diuji untuk mengetahui apakah pengurangan artefak batas tersebut berpengaruh terhadap hasil deteksi.

### 3.5.2 Variasi Representasi Orientasi

Pada \(C_2\), representasi arah pada konfigurasi sebelumnya diubah menjadi orientasi tak bertanda. Sudut \(\theta\) pada rentang \([0,2\pi)\) dipetakan ke rentang \([0,\pi)\) melalui:

\[
\theta_o=\theta\bmod\pi.
\]

Agar perubahan ini terutama menguji perbedaan antara arah dan orientasi, resolusi sudut dipertahankan sebesar \(1^\circ\) per interval. Oleh karena itu, rentang \([0,\pi)\) dibagi menjadi 180 interval orientasi:

\[
\Delta\theta=\frac{180^\circ}{180}=1^\circ.
\]

Dengan perubahan tersebut, dua arah yang berlawanan diperlakukan sebagai orientasi yang sama, tetapi ketelitian diskretisasi sudut tetap sama seperti konfigurasi 360 interval pada rentang \([0,2\pi)\). Dengan demikian, perbandingan \(C_1\) dan \(C_2\) terutama digunakan untuk menguji apakah representasi orientasi tak bertanda lebih sesuai daripada representasi arah bertanda, tanpa mencampurkan perubahan tersebut dengan pengurangan resolusi angular. Karena grid Fourier patch berukuran terbatas, sebagian interval dapat tidak terisi oleh koordinat frekuensi tertentu; tidak dilakukan penggabungan interval secara bergantung-data agar resolusi nominal tetap sama pada perbandingan ini.

### 3.5.3 Variasi Radial-Angular

Pada \(C_3\), informasi radial ditambahkan pada representasi orientasi. Radius pada grid Fourier dinyatakan sebagai:

\[
r(u,v)=\sqrt{(u-u_c)^2+(v-v_c)^2}.
\]

Radius kemudian dinormalisasi terhadap radius maksimum pada grid Fourier:

\[
\rho(u,v)=\frac{r(u,v)}{r_{max}},
\qquad 0\le\rho\le1.
\]

Untuk mempertahankan rancangan yang sederhana dan mudah ditafsirkan, spektrum dibagi menjadi tiga pita radial dengan lebar yang sama pada radius ternormalisasi:

\[
B_1: 0\le\rho\le\frac{1}{3},
\]

\[
B_2: \frac{1}{3}<\rho\le\frac{2}{3},
\]

\[
B_3: \frac{2}{3}<\rho\le1.
\]

Titik pusat spektrum \(\rho=0\) dengan demikian ditempatkan pada pita radial pertama agar seluruh koordinat Fourier memiliki definisi pita yang lengkap. Ketiga pita tersebut secara operasional mewakili rentang radial rendah, menengah, dan tinggi. Jumlah tiga pita merupakan keputusan desain penelitian untuk memperoleh pemisahan radial yang sederhana, bukan nilai yang dianggap optimal berdasarkan literatur.

Densitas dihitung untuk setiap kombinasi pita radial dan orientasi:

\[
D_i^c(b,k)=\sum_{(u,v)\in\Omega_{b,k}}A_i^c(u,v).
\]

Normalisasi probabilitas dilakukan secara terpisah pada setiap pita radial:

\[
p_i^c(b,k)=\frac{D_i^c(b,k)}{\sum_jD_i^c(b,j)+\varepsilon}.
\]

Entropi dan ambang adaptif kemudian dihitung pada masing-masing pita:

\[
H_i^c(b)=-\sum_k p_i^c(b,k)\log\left(\max(p_i^c(b,k),\varepsilon)\right),
\]

\[
\tau_i^c(b)=\frac{\gamma}{1+\exp(-H_i^c(b))}.
\]

Densitas relatif untuk pembobotan juga dinormalisasi di dalam pita yang sama:

\[
q_i^c(b,k)=\frac{D_i^c(b,k)}{\max_jD_i^c(b,j)+\varepsilon}.
\]

Pada \(C_3\), aturan ambang keras pada Subbab 3.4.5 diterapkan terhadap \(q_i^c(b,k)\) dan \(\tau_i^c(b)\) untuk setiap pita secara terpisah. Pembagian radial digunakan untuk menguji apakah pemisahan informasi berdasarkan jarak dari pusat spektrum memberikan informasi tambahan dibandingkan representasi yang hanya membedakan orientasi.

### 3.5.4 Variasi Ambang Lunak

Pada \(C_4\), ambang keras diganti dengan pembobotan yang berubah secara bertahap:

\[
w_{soft}(q,\tau)=q\,\sigma\left(\frac{q-\tau}{T}\right),
\]

dengan \(\sigma(\cdot)\) merupakan fungsi sigmoid dan \(T\) mengatur lebar transisi di sekitar nilai ambang. Nilai awal yang ditetapkan adalah:

\[
T=0{,}02.
\]

Nilai tersebut diperlakukan sebagai keputusan desain awal penelitian dan tidak dianggap sebagai nilai optimal yang berasal dari literatur. Secara matematis, nilai \(T\) yang lebih kecil membuat fungsi semakin mendekati ambang keras, sedangkan nilai yang lebih besar menghasilkan transisi yang lebih bertahap. Oleh karena itu, pengaruh \(T\) akan diperiksa secara terbatas sebagaimana dijelaskan pada Subbab 3.5.6.

Variasi ini digunakan untuk mengetahui apakah respons yang berada di sekitar nilai ambang lebih baik diperlakukan secara bertahap daripada langsung dipertahankan atau dihilangkan.

### 3.5.5 Variasi Panduan Luminansi

Pada \(C_5\), panduan spektral dibentuk dari sinyal luminansi menggunakan koefisien ITU-R BT.709-6 (International Telecommunication Union, 2015):

\[
Y=0{,}2126R+0{,}7152G+0{,}0722B.
\]

Bobot spektral dihitung dari panduan luminansi tersebut dan digunakan bersama pada ketiga kanal RGB. Citra keluaran tetap berupa RGB. Karena gate spasial yang sama digunakan pada ketiga kanal, operasi residual dapat dinyatakan sebagai \(R'=R(1+G_Y)\), \(G'=G(1+G_Y)\), dan \(B'=B(1+G_Y)\). Dengan demikian, rasio antar-kanal lokal dipertahankan selama penyebutnya tidak nol. Sifat ini menjadi alasan metodologis untuk menguji apakah satu panduan luminansi bersama dapat mengurangi perubahan warna relatif dibandingkan pembobotan yang dihitung secara independen pada setiap kanal.

Variasi ini digunakan untuk menguji apakah pembobotan spektral perlu dihitung secara terpisah pada setiap kanal warna atau cukup menggunakan satu panduan luminansi bersama.

### 3.5.6 Analisis Sensitivitas Terbatas

Setelah variasi utama \(C_0\) sampai \(C_5\) dianalisis, analisis sensitivitas terbatas dapat dilakukan terhadap parameter yang paling relevan pada konfigurasi struktur yang dipilih dari tahap tersebut. Analisis ini tidak dilakukan sebagai pencarian seluruh kombinasi parameter, tetapi satu parameter diuji pada satu waktu sementara parameter lain dipertahankan pada nilai referensinya. Kandidat yang direncanakan adalah:

\[
m\in\{16,32,64\},
\]

\[
\gamma\in\{0{,}05,0{,}10,0{,}15\},
\]

\[
T\in\{0{,}01,0{,}02,0{,}05\}.
\]

Ukuran patch \(m\) digunakan untuk melihat sensitivitas terhadap ukuran wilayah lokal. Ketika \(m\) diubah, overlap tetap 50% sehingga stride selalu ditetapkan sebagai:

\[
s=\frac{m}{2}.
\]

Parameter \(\gamma\) digunakan untuk melihat sensitivitas ambang adaptif, sedangkan \(T\) digunakan untuk melihat sensitivitas tingkat kelunakan pembobotan apabila konfigurasi struktur yang dianalisis telah memuat tahap ambang lunak. Nilai kandidat tersebut ditetapkan sebelum eksperimen sensitivitas dilakukan. Seluruh keputusan hanya menggunakan data pengembangan dan tidak menggunakan data uji.

Apabila suatu nilai nonreferensi memberikan kinerja validasi yang dipilih menurut aturan pada Subbab 3.6.2, hanya konfigurasi yang benar-benar telah dievaluasi yang dapat menjadi kandidat akhir. Nilai terbaik dari beberapa pengujian satu-parameter tidak boleh langsung digabungkan menjadi kombinasi baru yang belum diuji. Jika kombinasi baru memang ingin digunakan, kombinasi tersebut harus didefinisikan dan dievaluasi pada data pengembangan sebelum pengujian ulang beberapa *seed* dan sebelum data uji dibuka. Dengan demikian, analisis sensitivitas tetap berfungsi sebagai bagian pengembangan metode dan tidak digunakan untuk menyesuaikan metode terhadap data uji.

## 3.6 Rancangan Eksperimen

Eksperimen dibagi menjadi tiga tahap utama agar pembentukan model acuan, pemilihan konfigurasi, dan pengujian akhir memiliki fungsi yang jelas. Evaluasi pada arsitektur lain ditempatkan sebagai analisis tambahan.

### 3.6.1 Tahap I — Pembentukan Model Acuan

Tahap pertama membentuk model acuan YOLO26n menggunakan data pelatihan dan validasi yang telah ditetapkan. Model diinisialisasi langsung dari bobot pralatih resmi `yolo26n.pt` dan dilatih menggunakan konfigurasi pada Subbab 3.7. Model acuan digunakan untuk menetapkan kinerja dasar pada data validasi, menentukan kelompok tiga kelas sulit \(\mathcal{H}\), dan memeriksa bahwa pipeline data serta evaluasi dapat berjalan dengan benar. Checkpoint hasil pelatihan model acuan **tidak** digunakan sebagai bobot awal konfigurasi \(C_0\) sampai \(C_5\).

Sebelum pelatihan, dilakukan pemeriksaan terhadap format data, jumlah data pada setiap kelas, kesesuaian anotasi, hasil prapemrosesan, serta pemisahan data uji. Pemeriksaan ini bertujuan memastikan bahwa perbedaan hasil tidak berasal dari kesalahan data atau implementasi.

### 3.6.2 Tahap II — Pengujian Variasi Prapemrosesan

Tahap kedua digunakan untuk menganalisis konfigurasi \(C_0\) sampai \(C_5\). Setiap konfigurasi dibangun kembali langsung dari bobot pralatih resmi `yolo26n.pt` menggunakan *seed* pengembangan yang sama. Inisialisasi bagian keluaran untuk jumlah kelas \(C\) diperiksa agar setara antar konfigurasi. Dengan demikian, hubungan:

\[
C_0\rightarrow C_1\rightarrow\cdots\rightarrow C_5
\]

menunjukkan akumulasi perubahan desain prapemrosesan, bukan kelanjutan pelatihan atau pewarisan checkpoint dari konfigurasi sebelumnya.

Seluruh konfigurasi pada tahap ini menggunakan *seed* pengembangan 42. Data validasi digunakan untuk membandingkan perubahan antar tahap. Pertama, kandidat struktur dipilih dari \(C_0\) sampai \(C_5\) menggunakan metrik utama \(mAP_{50:95}\):

\[
C_{str}=\arg\max_{C_j,\,j\in\{0,\ldots,5\}} mAP_{50:95}^{val}(C_j).
\]

Dua nilai diperlakukan sebagai seri apabila selisih absolut \(mAP_{50:95}\) kurang dari 0,001 pada skala 0–1, yang setara dengan kurang dari 0,1 poin persentase. Jika seri terjadi, konfigurasi dengan rerata AP lebih tinggi pada kelompok tiga kelas sulit \(\mathcal{H}\) yang telah ditetapkan dari model acuan dipilih. Jika masih seri, konfigurasi dengan waktu pemrosesan total yang lebih rendah dipilih.

Jika analisis sensitivitas pada Subbab 3.5.6 dilakukan, pengujian tersebut diterapkan setelah \(C_{str}\) ditetapkan. Konfigurasi akhir \(C^*\) dipilih dari \(C_{str}\) dan seluruh varian sensitivitas yang benar-benar telah dievaluasi, menggunakan urutan kriteria yang sama: \(mAP_{50:95}\), kemudian \(AP_{\mathcal{H}}\) untuk kondisi seri, kemudian waktu pemrosesan total. Jika analisis sensitivitas tidak dilakukan, maka \(C^*=C_{str}\). Aturan tersebut ditetapkan sebelum evaluasi akhir agar pemilihan \(C^*\) tidak berubah setelah melihat data uji. Hasil tahap ini digunakan untuk memilih konfigurasi yang akan diuji lebih lanjut dan belum digunakan sebagai dasar kesimpulan akhir penelitian.

### 3.6.3 Tahap III — Pengujian Ulang dengan Beberapa Seed

Pada tahap ketiga, empat kondisi utama diuji kembali menggunakan beberapa *seed* untuk melihat kestabilan hasil:

| Kode | Kondisi |
|---|---|
| \(B_0\) | YOLO26n tanpa prapemrosesan |
| \(B_1\) | CLAHE + YOLO26n |
| \(B_2\) | \(C_0\) + YOLO26n |
| \(B_3\) | \(C^*\) + YOLO26n |

Nilai *seed* yang digunakan adalah:

\[
s\in\{42,123,2026\}.
\]

Pada setiap *seed*, keempat kondisi dibangun kembali langsung dari bobot pralatih resmi `yolo26n.pt`. Inisialisasi bagian keluaran untuk jumlah kelas \(C\) dilakukan dengan *seed* yang sama, kemudian kondisi awal parameter model diperiksa agar setara sebelum pelatihan. Dengan demikian, perbedaan utama yang dibandingkan pada setiap *seed* adalah perlakuan terhadap citra masukan. Konfigurasi \(C^*\) yang digunakan pada tahap ini telah ditetapkan dari Tahap II dan tidak diubah berdasarkan hasil pengujian ulang maupun data uji akhir.

Untuk suatu metrik \(M\), perubahan terhadap model acuan dihitung secara umum sebagai:

\[
\Delta_s=M_{perlakuan,s}-M_{acuan,s}.
\]

Hasil setiap *seed*, rata-rata, dan variasinya akan dilaporkan. Perbandingan langsung antara \(C^*\) dan CLAHE juga dilakukan untuk melihat apakah konfigurasi terpilih memberikan perubahan yang berbeda dari peningkatan kontras lokal biasa.

### 3.6.4 Evaluasi pada Arsitektur Lain — Opsional

Sebagai analisis tambahan, konfigurasi \(C^*\) dapat diterapkan pada RT-DETRv3-R18 setelah konfigurasi utama ditetapkan. Dua kondisi yang dibandingkan adalah RT-DETRv3-R18 tanpa prapemrosesan dan RT-DETRv3-R18 dengan \(C^*\). Keduanya menggunakan pembagian data dan aturan evaluasi yang sama.

Konfigurasi \(C^*\), termasuk parameter prapemrosesannya, tidak dituning ulang khusus untuk RT-DETRv3-R18. Analisis ini bertujuan melihat apakah arah pengaruh prapemrosesan yang telah ditetapkan pada YOLO26n juga muncul pada keluarga model deteksi yang berbeda. Hasilnya tidak digunakan untuk memilih \(C^*\) dan tidak dimaksudkan untuk menentukan arsitektur mana yang lebih unggul. Jika sumber daya komputasi tidak memadai, analisis ini tidak menjadi syarat bagi kesimpulan utama penelitian.

### 3.6.5 Evaluasi Akhir pada Data Uji

Data uji akhir disisihkan sejak awal dan tidak digunakan selama pengembangan metode maupun pemilihan parameter. Dengan target sekitar 200 citra primer dan proporsi sekitar 15% untuk pengujian, data uji direncanakan memuat sekitar 30 citra sumber.

Sebelum digunakan, data uji harus memenuhi beberapa syarat: seluruh kelas akhir tersedia, setiap kelas memiliki sedikitnya 10 objek dan muncul pada sedikitnya 5 citra sumber, tidak terdapat citra atau kelompok sumber yang sama dengan data pengembangan, serta seluruh anotasi telah lolos pemeriksaan. Keterwakilan tersebut diperiksa pada saat pembagian data dengan prosedur berbasis kelompok yang juga mempertimbangkan distribusi kelas, sehingga kekurangan dukungan suatu kelas pada data uji diketahui sebelum eksperimen utama dimulai dan tanpa menggunakan hasil prediksi model.

Jika syarat tersebut belum terpenuhi, pengumpulan data primer akan ditambah terlebih dahulu. Jika penambahan data tidak memungkinkan, evaluasi alternatif berupa validasi silang berbasis kelompok dapat digunakan dengan keterbatasannya dilaporkan secara jelas. Keputusan untuk menggunakan evaluasi alternatif harus dibuat sebelum hasil model pada data uji digunakan untuk keputusan metodologis.

Data uji baru digunakan setelah \(C^*\), aturan pemilihan model, metrik, dan prosedur evaluasi ditetapkan. Setelah Tahap III selesai, checkpoint final dari seluruh kondisi \(B_0\) sampai \(B_3\) pada seed 42, 123, dan 2026 dievaluasi pada data uji yang sama. Tidak dilakukan perubahan metode, parameter, atau pemilihan ulang konfigurasi berdasarkan hasil data uji.

## 3.7 Konfigurasi Pelatihan

Konfigurasi utama pelatihan YOLO26n ditunjukkan pada Tabel 3.3.

### Tabel 3.3 Konfigurasi Utama Pelatihan YOLO26n

| Parameter | Nilai |
|---|---|
| Model | YOLO26n |
| Bobot awal | `yolo26n.pt` pralatih resmi |
| Ukuran masukan | 640 × 640 piksel |
| Epoch maksimum | 50 |
| Ukuran batch | 16 |
| Penghentian dini | 15 epoch tanpa peningkatan |
| Optimizer | Auto |
| Seed pengembangan | 42 |
| Seed pengujian ulang | 42, 123, 2026 |

Seluruh kondisi yang dibandingkan pada tahap yang sama menggunakan dataset, augmentasi, ukuran masukan, jumlah epoch maksimum, ukuran batch, aturan penghentian dini, dan lingkungan komputasi yang sama. Model tidak harus berhenti pada epoch yang sama karena penghentian dini mengikuti kinerja validasi masing-masing kondisi.

Untuk setiap run, model yang digunakan pada evaluasi adalah checkpoint `best.pt` yang dipilih oleh mekanisme validasi Ultralytics pada versi perangkat lunak yang telah dikunci. Aturan pemilihan checkpoint dibuat sama untuk seluruh kondisi dan tidak diubah berdasarkan hasil data uji.

Versi Ultralytics akan ditetapkan dan tidak diubah selama eksperimen utama sehingga perilaku optimizer, augmentasi, dan pemilihan checkpoint tetap konsisten. Apabila `optimizer=Auto` memilih optimizer tertentu secara internal, optimizer aktual beserta parameter penting yang digunakan akan dicatat pada log eksperimen. Rincian implementasi lain, seperti jumlah proses pemuat data, penggunaan *cache*, penghentian *mosaic* menjelang akhir pelatihan, dan parameter prediksi, tetap ditetapkan serta dicatat pada konfigurasi eksperimen, tetapi tidak dijadikan faktor penelitian.

## 3.8 Evaluasi Kinerja Deteksi

Evaluasi utama menggunakan *Average Precision* (AP). Presisi dan *recall* tetap dilaporkan sebagai metrik tambahan:

\[
Precision=\frac{TP}{TP+FP},
\]

\[
Recall=\frac{TP}{TP+FN}.
\]

Metrik utama penelitian adalah:

\[
\boxed{mAP_{50:95}},
\]

yaitu rata-rata AP pada ambang IoU 0,50 sampai 0,95. Nilai \(mAP_{50}\) digunakan sebagai metrik sekunder. Jumlah maksimum prediksi yang dievaluasi pada setiap citra ditetapkan sebesar 500 untuk seluruh kondisi.

Presisi dan *recall* dilaporkan menggunakan prosedur evaluasi Ultralytics yang sama pada versi perangkat lunak yang telah dikunci. Keduanya merupakan metrik sekunder dan tidak digunakan untuk memilih \(C^*\); dengan demikian tidak dilakukan pemilihan *confidence threshold* khusus untuk menguntungkan salah satu kondisi.

Selain metrik rata-rata, AP50–95 setiap kelas juga akan dilaporkan:

\[
AP_{c,50:95},\qquad c=1,\ldots,C.
\]

Untuk mengamati kelas yang sulit secara konsisten, tiga kelas dengan AP50–95 terendah ditentukan satu kali dari model acuan pada data validasi:

\[
\mathcal{H}=\operatorname{Bottom3}(AP_{c,50:95}^{acuan}).
\]

Setelah ditetapkan, kelompok kelas \(\mathcal{H}\) dibekukan dan digunakan untuk seluruh perbandingan berikutnya, termasuk ketika hasil pada data uji akhir dilaporkan. Data uji tidak digunakan untuk memilih ulang kelas yang dianggap sulit. Rerata AP pada tiga kelas tersebut dihitung sebagai:

\[
AP_{\mathcal{H}}=\frac{1}{3}\sum_{c\in\mathcal{H}}AP_{c,50:95}.
\]

AP kelas terendah juga dilaporkan sebagai indikator tambahan:

\[
AP_{worst}=\min_c AP_{c,50:95}.
\]

Indikator ini digunakan untuk memastikan bahwa peningkatan nilai rata-rata tidak menutupi penurunan yang besar pada kelas tertentu dan tidak digunakan sebagai pengganti metrik utama. Pada pengujian ulang dengan beberapa *seed*, hasil dilaporkan untuk setiap *seed* beserta rata-rata dan simpangan baku.

Jika jumlah kelompok sumber memadai, ketidakpastian perbedaan antar kondisi pada evaluasi akhir akan dianalisis lebih lanjut menggunakan *paired bootstrap* berbasis kelompok sumber. Pada setiap pengulangan *bootstrap*, kelompok sumber yang sama diambil untuk seluruh kondisi yang dibandingkan, lalu selisih metrik dihitung dari pasangan hasil tersebut. Pendekatan ini digunakan agar interval ketidakpastian merefleksikan sifat perbandingan pada citra sumber yang sama.

## 3.9 Analisis Visual

Analisis visual digunakan sebagai pendukung hasil kuantitatif untuk membantu menjelaskan perubahan pada citra, respons spektral, aktivasi model, dan hasil deteksi. Visualisasi tidak digunakan sebagai satu-satunya bukti mengenai penyebab peningkatan atau penurunan kinerja.

### 3.9.1 Visualisasi Tahapan Prapemrosesan

Untuk contoh citra yang dipilih, visualisasi akan menampilkan:

1. citra masukan;
2. patch lokal;
3. amplitudo spektrum Fourier;
4. distribusi angular atau radial-angular;
5. ambang dan bobot spektral;
6. hasil transformasi Fourier balik;
7. respons hasil rekonstruksi; dan
8. citra setelah penggabungan residual.

Visualisasi ini digunakan untuk menunjukkan perubahan yang dilakukan oleh prapemrosesan. Perubahan kontras atau tekstur yang terlihat tidak langsung dianggap sebagai bukti bahwa citra tersebut lebih baik bagi model deteksi.

### 3.9.2 Visualisasi Respons Model

Visualisasi respons internal model direncanakan menggunakan metode CAM yang kompatibel dengan YOLO26. Eigen-CAM dipertimbangkan sebagai metode utama untuk menggambarkan struktur aktivasi fitur dominan. Metode lain hanya akan digunakan apabila lapisan target dan prosedur visualisasinya dapat diterapkan secara konsisten pada seluruh kondisi.

Metode, lapisan target, ukuran masukan, dan normalisasi visualisasi akan dibuat sama pada YOLO26n tanpa prapemrosesan, CLAHE, \(C_0\), dan \(C^*\). Untuk mencegah pemilihan seed berdasarkan tampilan visual, analisis utama menggunakan model dari seed 42 yang telah ditetapkan sebelumnya. Jika visualisasi dari seed lain ditampilkan sebagai tambahan, hasil tersebut diberi label secara eksplisit dan tidak digunakan untuk memilih contoh utama. Hasil CAM hanya digunakan sebagai analisis pendukung dan tidak menggantikan metrik deteksi.

### 3.9.3 Visualisasi Hasil Deteksi

Hasil prediksi YOLO26n tanpa prapemrosesan, CLAHE, \(C_0\), dan \(C^*\) akan dibandingkan pada citra yang sama. Visualisasi mencakup kotak pembatas, label kelas, dan skor kepercayaan. Parameter prediksi yang memengaruhi tampilan, termasuk *confidence threshold*, ambang IoU yang relevan, dan `max_det`, ditetapkan sama untuk seluruh kondisi sebelum contoh hasil dibuat. Visualisasi utama menggunakan model seed 42 agar pemilihan hasil tidak dilakukan berdasarkan seed dengan tampilan yang paling menguntungkan.

Contoh citra akan dipilih berdasarkan kriteria yang ditetapkan sebelumnya, misalnya kelas dengan kinerja tinggi, kelas dengan kinerja rendah, kasus ketika seluruh model benar, seluruh model salah, dan kasus ketika hasil antar model berbeda. Pendekatan ini digunakan untuk mengurangi kecenderungan memilih contoh yang hanya mendukung salah satu kondisi.

## 3.10 Analisis Kesalahan dan Kinerja Per Kelas

Analisis kesalahan dilakukan menggunakan AP per kelas, matriks kebingungan, prediksi positif palsu (*false positive*), dan objek yang terlewat (*false negative*). Perbandingan antar kondisi digunakan untuk mengetahui kelas yang mengalami peningkatan, relatif stabil, atau menurun.

Kelas juga dapat dikelompokkan berdasarkan jenis informasi visual yang diperlukan oleh definisi label, misalnya karakteristik permukaan dan warna, jumlah detail lokal, bentuk dan integritas, tingkat keutuhan objek, serta ukuran fisik. Pengelompokan ditetapkan berdasarkan definisi kelas sebelum hasil eksperimen diperiksa dan hanya digunakan untuk analisis deskriptif.

Sebagai analisis deskriptif tambahan, kesalahan salah kelas dan kesalahan lokalisasi dapat ditinjau secara terpisah menggunakan ambang IoU 0,50 sebagai acuan interpretasi. Prediksi yang mempunyai tumpang tindih spasial memadai dengan objek acuan pada IoU sedikitnya 0,50 tetapi memberikan label kelas yang salah dicatat sebagai contoh kesalahan diskriminasi kelas. Prediksi dengan kelas yang sesuai tetapi tidak mencapai IoU 0,50 dicatat sebagai contoh kesalahan lokalisasi, sedangkan objek acuan tanpa pasangan prediksi tetap termasuk *false negative* dan prediksi tanpa pasangan acuan termasuk *false positive*. Pemisahan ini digunakan untuk membantu interpretasi dan tidak dijadikan metrik utama baru di luar prosedur evaluasi deteksi yang telah ditetapkan.

## 3.11 Evaluasi Efisiensi Komputasi

Meskipun CLAHE dan prapemrosesan frekuensi-angular tidak menambahkan parameter yang dilatih, keduanya tetap memerlukan waktu komputasi. Oleh karena itu, efisiensi dievaluasi pada tingkat sistem, bukan hanya berdasarkan jumlah parameter model.

Pengukuran mencakup jumlah parameter model, waktu prapemrosesan, waktu inferensi model, waktu pemrosesan total, jumlah citra yang dapat diproses per detik, dan penggunaan memori GPU. Pengukuran utama latency dilakukan pada ukuran masukan 640 × 640 dengan *batch* 1 agar overhead *frontend* dapat terlihat secara langsung.

Untuk kondisi dengan prapemrosesan, waktu total dinyatakan sebagai:

\[
t_{total}=t_{pra}+t_{model}.
\]

Komponen \(t_{pra}\) hanya mencakup overhead prapemrosesan yang berbeda antar kondisi setelah representasi masukan umum telah disiapkan, sedangkan aktivitas I/O disk yang sama untuk seluruh kondisi tidak dimasukkan ke perbandingan latency utama. Untuk model tanpa prapemrosesan tambahan, \(t_{pra}\) diperlakukan sebagai nol pada definisi tersebut.

Pengukuran dilakukan menggunakan perangkat, ukuran masukan, ukuran batch, dan presisi komputasi yang sama. Jumlah *warm-up* dan pengulangan pengukuran ditetapkan sebelum benchmark dan dibuat sama pada seluruh kondisi. Pada GPU, sinkronisasi CUDA dilakukan sebelum dan sesudah bagian yang diukur agar waktu operasi asinkron tidak menghasilkan estimasi latency yang terlalu rendah. Hasil latency dilaporkan menggunakan ukuran statistik yang sama untuk semua kondisi, dan penggunaan memori dilaporkan sebagai penggunaan puncak (*peak allocated GPU memory*) pada prosedur yang sama.

## 3.12 Lingkungan Implementasi

Implementasi penelitian menggunakan Python, PyTorch, dan Ultralytics YOLO. Versi perangkat lunak akan ditetapkan sebelum eksperimen utama dan tidak diubah selama perbandingan berlangsung. Ultralytics 8.4.96 digunakan sebagai versi referensi. Informasi versi Python, PyTorch, CUDA, GPU, sistem operasi, dan perangkat keras akan dicatat.

Versi kode prapemrosesan juga akan ditetapkan melalui identitas *commit* agar konfigurasi yang digunakan pada setiap eksperimen dapat ditelusuri. Nilai *seed* serta konfigurasi reproducibility dan determinisme yang relevan pada PyTorch/CUDA akan dicatat bersama setiap run. Penelitian tidak mengasumsikan bahwa seluruh operasi GPU harus identik secara bitwise, tetapi seluruh kondisi dibandingkan menggunakan pengaturan reproducibility yang sama.

Jika evaluasi menggunakan RT-DETRv3-R18 dilakukan, versi kode dan bobot pralatih yang digunakan juga akan dicatat secara terpisah. Langkah ini dilakukan untuk menjaga reprodusibilitas eksperimen dan memudahkan verifikasi metodologi.