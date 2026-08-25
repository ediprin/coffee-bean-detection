# BAB III
# METODOLOGI PENELITIAN

## 3.1 Arsitektur Umum Penelitian

Penelitian ini menggunakan pendekatan eksperimental komparatif untuk menganalisis pengaruh *preprocessing* citra berbasis frekuensi-angular terhadap kinerja YOLO26n pada deteksi *fine-grained* cacat biji kopi. Arsitektur *backbone*, *neck*, dan *detection head* YOLO26n dipertahankan pada perbandingan utama sehingga perbedaan utama antar kondisi eksperimen berasal dari perlakuan terhadap tensor citra masukan.

Secara umum, penelitian akan terdiri atas pengumpulan dataset primer dan pencegahan kebocoran data, pembentukan baseline YOLO26n, pembentukan konfigurasi referensi *preprocessing* frekuensi-angular, analisis faktor desain *preprocessing*, konfirmasi berpasangan pada beberapa *seed*, evaluasi pada *locked test*, analisis kinerja per kelas dan kesalahan, analisis visual, serta evaluasi efisiensi komputasi.

Alur penelitian dirangkum sebagai berikut:

```text
Pengumpulan dataset primer biji kopi
        ↓
Anotasi, audit kelas, dan grouped split
        ↓
Baseline development YOLO26n
        ↓
Reference frequency-angular preprocessing
        ↓
Staged cumulative spectral factorization
        ↓
Pemilihan konfigurasi kandidat C*
        ↓
Paired multi-seed confirmation
        ↓
Locked final evaluation
        ↓
Analisis per kelas, kesalahan, visual, dan efisiensi
        ↓
Kesimpulan
```

Perbandingan utama dapat dinyatakan secara konseptual sebagai:

\[
\hat{Y}_{N}=\operatorname{YOLO26n}(I),
\]

untuk detector tanpa *preprocessing*, dan:

\[
I'=\mathcal{P}_{FA}(I),
\]

\[
\hat{Y}_{P}=\operatorname{YOLO26n}(I'),
\]

dengan \(I\) merupakan tensor citra masukan, \(\mathcal{P}_{FA}\) merupakan fungsi *preprocessing* frekuensi-angular, \(I'\) merupakan tensor hasil *preprocessing*, dan \(\hat{Y}\) merupakan hasil prediksi deteksi. Persamaan tersebut menyatakan perbedaan pada jalur masukan dan tidak mengasumsikan bahwa *preprocessing* akan selalu meningkatkan kinerja.

## 3.2 Dataset Penelitian

### 3.2.1 Sumber, Target Jumlah, dan Karakteristik Dataset Primer

Penelitian direncanakan menggunakan **dataset primer** yang dikumpulkan secara langsung untuk tugas *multi-class object detection* pada biji kopi hijau. Taksonomi awal menargetkan 20 kategori cacat fisik yang mengacu pada SNI 01-2907-2008 ditambah satu kelas biji normal, sehingga jumlah kelas target awal adalah:

\[
C_{target}=21.
\]

Jumlah kelas final akan dibekukan setelah tahap audit kelayakan kelas selesai dan sebelum pembagian data serta pelatihan model dilakukan. Dengan demikian, proposal tidak mengasumsikan bahwa seluruh kelas langka pasti dapat diperoleh dalam jumlah memadai hanya melalui augmentasi.

Berbeda dengan dataset klasifikasi satu-biji-satu-citra, satu citra pada penelitian ini akan memuat banyak objek. Oleh karena itu, kecukupan dataset dinilai menggunakan dua satuan sekaligus, yaitu jumlah **citra sumber independen** dan jumlah **instance/bounding box** per kelas. Target pengumpulan ditetapkan sekitar 180–220 citra sumber independen, dengan target nominal sekitar 200 citra asli. Augmentasi tidak dihitung sebagai citra primer.

Setiap citra direncanakan memuat sekitar 30–50 objek yang disusun dalam satu lapisan dengan orientasi bervariasi dan tanpa *severe overlap*. Dengan rancangan tersebut, target total anotasi berada pada kisaran:

\[
N_{box}\approx 6.000-10.000.
\]

Untuk setiap kelas yang dipertahankan pada taksonomi final, target dukungan ditetapkan sebagai berikut:

1. sekurang-kurangnya sekitar 200 instance asli per kelas;
2. target ideal sekitar 300–500 instance per kelas; dan
3. kelas muncul pada sekurang-kurangnya 15–20 citra sumber independen.

Angka tersebut merupakan **target perencanaan pengumpulan**, bukan jumlah data yang diklaim telah tersedia pada saat proposal disusun. Target ini dipilih agar tetap realistis untuk penelitian tesis tetapi tetap mempunyai dukungan multi-instance yang memadai. Sebagai konteks metodologis, Bahy dan Rifai (2026) melaporkan 107 citra sumber dengan 13.863 anotasi untuk deteksi 20 kelas SNI, sedangkan Tarekegn dan Debelee (2025) menggunakan 562 citra dengan 19.228 instance untuk 13 kelas cacat dan satu kelas normal. Dengan demikian, jumlah citra pada tugas deteksi tidak dapat dinilai terpisah dari jumlah objek yang teranotasi di dalam setiap citra.

Pengambilan citra akan dilakukan secara *top-down* menggunakan latar belakang polos dan tidak reflektif, posisi kamera tetap, jarak kamera tetap, serta pencahayaan yang dikendalikan. Objek disusun dalam satu lapisan agar detail permukaan tetap terlihat dan agar penelitian berfokus pada diskriminasi cacat *fine-grained*, bukan pada *severe occlusion*. Orientasi objek tetap divariasikan untuk merepresentasikan kemungkinan sisi dan arah biji yang berbeda.

Setiap sesi akuisisi akan memiliki identitas sesi/batch dan identitas citra sumber. Untuk kelas yang definisinya bergantung pada ukuran fisik, khususnya kategori benda asing berukuran kecil, sedang, dan besar, setup akan dikalibrasi menggunakan referensi skala pada awal sesi sehingga hubungan piksel terhadap ukuran fisik dapat ditelusuri secara konsisten.

Setiap objek akan diberi *bounding box* dan label kelas. Definisi operasional kelas akan disusun sebelum anotasi dengan mengacu pada SNI dan referensi visual yang digunakan. Sampel yang secara visual ambigu tidak akan dipaksakan masuk ke kelas tertentu; sampel tersebut akan ditandai untuk peninjauan ulang. Validasi label direncanakan melibatkan praktisi atau validator yang memahami penilaian fisik mutu kopi, terutama untuk kelas yang mempunyai kemiripan visual tinggi.

### 3.2.2 Audit Kelayakan Kelas dan Pembekuan Taksonomi

Sebelum *training-validation-test split* dibekukan, dataset primer akan diaudit untuk mengetahui jumlah instance dan jumlah citra sumber yang memuat setiap kelas. Audit ini penting karena penelitian terdahulu berbasis SNI menunjukkan bahwa beberapa kategori dapat sangat langka atau sulit dibedakan secara konsisten dari citra RGB.

Suatu kelas hanya akan dipertahankan sebagai kelas evaluasi utama apabila memenuhi batas dukungan minimum yang ditetapkan sebelum pelatihan, yaitu sekitar 200 instance asli dan muncul pada sekurang-kurangnya 15 citra sumber independen. Kekurangan data pada suatu kelas tidak akan ditutup dengan memperbanyak salinan augmentasi dari sejumlah kecil sumber asli. Apabila suatu kelas target tidak memenuhi kriteria tersebut, keputusan untuk menambah pengumpulan data atau menyesuaikan taksonomi akan dilakukan **sebelum** model utama dilatih dan akan didokumentasikan secara transparan.

Setelah audit selesai, jumlah kelas final dinotasikan sebagai:

\[
C\le C_{target},
\]

dengan target utama tetap \(C=21\) apabila seluruh kelas memenuhi kriteria dukungan yang telah ditentukan.

### 3.2.3 Pembagian Dataset dan Pencegahan Kebocoran Data

Pembagian data akan dilakukan **pada citra sumber asli sebelum augmentasi**. Target pembagian awal adalah sekitar 70% untuk *training*, 15% untuk *validation*, dan 15% untuk *locked test*. Untuk target nominal 200 citra sumber, pembagian tersebut setara secara kasar dengan sekitar 140 citra *training*, 30 citra *validation*, dan 30 citra *test*. Jumlah akhir dapat sedikit disesuaikan untuk menjaga dukungan setiap kelas, tetapi data *test* tidak boleh digunakan untuk memilih metode atau parameter.

Pembagian menggunakan *grouped split* pada tingkat sumber fisik atau sesi akuisisi. Jika beberapa citra berasal dari batch fisik, komposisi objek, atau sesi yang sangat berkaitan, seluruh citra tersebut akan ditempatkan pada bagian data yang sama. Dengan demikian, citra yang sangat berkaitan tidak tersebar antara *training*, *validation*, dan *test*.

Secara konseptual, apabila \(\mathcal{G}_{train}\), \(\mathcal{G}_{val}\), dan \(\mathcal{G}_{test}\) menyatakan himpunan *source/acquisition group*, maka:

\[
\mathcal{G}_{train}\cap\mathcal{G}_{val}=\varnothing,
\]

\[
\mathcal{G}_{train}\cap\mathcal{G}_{test}=\varnothing,
\]

\[
\mathcal{G}_{val}\cap\mathcal{G}_{test}=\varnothing.
\]

Pemeriksaan *exact hash* juga akan digunakan untuk mencegah citra identik muncul pada bagian data yang berbeda. Semua turunan augmentasi dari suatu citra sumber hanya boleh berada pada bagian *training* yang sama dengan sumbernya dan tidak boleh masuk ke *validation* atau *test*.

Data *validation* digunakan untuk *early stopping*, analisis faktor *preprocessing*, dan pemilihan konfigurasi kandidat. Data *test* tidak digunakan untuk pemilihan faktor *preprocessing*, *checkpoint*, ukuran patch, nilai \(\gamma\), maupun keputusan metodologis lainnya. *Locked test* hanya akan digunakan setelah konfigurasi final dan protokol evaluasi dibekukan sebagaimana dijelaskan pada Subbab 3.6.

### 3.2.4 Augmentasi Data

Augmentasi hanya diterapkan pada bagian *training* setelah pembagian citra sumber selesai. Data *validation* dan *test* mempertahankan citra asli tanpa augmentasi sintetis. Augmentasi merupakan bagian dari *runtime training pipeline* YOLO26 dan akan diterapkan dengan konfigurasi yang sama pada setiap kondisi yang dibandingkan dalam tahap eksperimen yang sama. Seluruh parameter augmentasi akan dikunci sebelum perbandingan dan direkam bersama konfigurasi *run*.

*Preprocessing* frekuensi-angular tidak diperlakukan sebagai augmentasi karena operator tersebut bersifat deterministik, tidak menghasilkan label baru, dan tidak melakukan translasi, rotasi geometris, *cropping*, atau *warping* koordinat *bounding box*. Pada implementasi penelitian, augmentasi dan pembentukan tensor masukan dilakukan terlebih dahulu oleh *pipeline* data YOLO26. Tensor tersebut kemudian diproses oleh *frontend* frekuensi-angular sebelum memasuki jalur *forward* native YOLO26n. Frontend yang sama akan digunakan pada *training*, *validation*, dan *inference* untuk kondisi yang menggunakan *preprocessing*.

## 3.3 Model Dasar YOLO26n

YOLO26n digunakan sebagai model dasar pada penelitian ini. YOLO26 merupakan keluarga *real-time object detector* yang diperkenalkan oleh Jocher et al. (2026). Varian nano dipilih agar eksperimen menggunakan detector dengan kompleksitas relatif rendah sekaligus mempertahankan deteksi multi-skala pada P3, P4, dan P5.

Model menggunakan bobot *pretrained* resmi `yolo26n.pt` sebagai sumber inisialisasi pada perbandingan utama. Setelah taksonomi dataset dibekukan, bagian prediksi kelas disesuaikan dengan jumlah kelas final \(C\), dengan target utama \(C=21\). Pada *final paired comparison*, pembentukan detector target akan menggunakan *seed* inisialisasi yang sama untuk kondisi yang dipasangkan. Setelah transfer bobot *pretrained*, kesetaraan *persistent detector state* antar kondisi akan diverifikasi sebelum pelatihan dimulai sehingga satu-satunya perbedaan awal yang diizinkan adalah keberadaan atau konfigurasi *frontend preprocessing*.

Pada penelitian ini tidak dilakukan modifikasi terhadap *backbone*, *neck*, maupun *detection head* YOLO26n. *Preprocessing* yang dianalisis juga tidak mempunyai parameter trainable. Dengan demikian, analisis difokuskan pada perubahan representasi masukan dan bukan pada perubahan kapasitas arsitektur detector.

## 3.4 Preprocessing Citra Berbasis Frekuensi-Angular

*Preprocessing* yang akan digunakan mengadaptasi mekanisme pemrosesan frekuensi lokal dan distribusi angular pada komponen AFAB-2 dari Xu et al. (2025). Pada penelitian asal, AFAB-2 merupakan bagian dari mekanisme AFAB dalam LFDet. Penelitian ini tidak mengadopsi keseluruhan LFDet maupun AFAB-1, tetapi menggunakan prinsip AFAB-2 sebagai konfigurasi referensi untuk membentuk *parameter-free input preprocessing frontend* sebelum jalur *forward* native YOLO26n.

Tahapan utama konfigurasi referensi meliputi pembentukan patch lokal, transformasi Fourier, pembentukan distribusi angular, perhitungan ambang berbasis entropi, pembobotan respons spektral, *inverse Fourier transform*, rekonstruksi patch yang saling overlap, normalisasi respons spasial, dan penggabungan residual dengan tensor masukan.

### 3.4.1 Pembentukan Patch Lokal

Untuk tensor citra RGB:

\[
I\in\mathbb{R}^{3\times H\times W},
\]

operator membentuk patch lokal:

\[
P_i\in\mathbb{R}^{3\times m\times m}.
\]

Konfigurasi referensi menggunakan:

\[
m=32,
\]

dengan overlap:

\[
o=0{,}50.
\]

Stride dihitung sebagai:

\[
s=m(1-o)=16.
\]

Patch lokal digunakan agar analisis frekuensi tetap mempertahankan variasi menurut lokasi dan tidak hanya menggambarkan spektrum global seluruh citra. Apabila dimensi tensor tidak tepat memenuhi grid patch, *replicate padding* akan diterapkan pada sisi yang diperlukan. Setelah rekonstruksi, bagian padding dibuang sehingga dimensi keluaran kembali menjadi \(H\times W\). Penggunaan overlap 50%, *replicate padding*, dan mekanisme rekonstruksi merupakan konvensi implementasi transfer yang dibekukan agar operator dapat direproduksi secara konsisten.

### 3.4.2 Transformasi Fourier

Untuk patch ke-\(i\) dan kanal warna ke-\(c\), transformasi Fourier dua dimensi dihitung sebagai:

\[
F_i^c(u,v)=\operatorname{fftshift}\left[\mathcal{F}_2\{P_i^c\}(u,v)\right].
\]

Transformasi menggunakan normalisasi ortonormal. Koefisien Fourier selanjutnya dinyatakan dalam amplitudo dan fase:

\[
A_i^c(u,v)=|F_i^c(u,v)|,
\]

\[
\phi_i^c(u,v)=\arg F_i^c(u,v).
\]

`fftshift` digunakan agar komponen frekuensi nol berada di sekitar pusat grid spektrum. Untuk ukuran patch \(m\times m\), pusat diskrit dinyatakan sebagai:

\[
(u_c,v_c)=\left(\left\lfloor\frac{m}{2}\right\rfloor,\left\lfloor\frac{m}{2}\right\rfloor\right).
\]

Amplitudo digunakan untuk membentuk distribusi spektral, sedangkan fase tidak dimodifikasi secara eksplisit pada koefisien yang dipertahankan karena spektrum kompleks hanya dikalikan dengan bobot real non-negatif.

### 3.4.3 Distribusi Angular

Setiap koordinat frekuensi dipetakan ke sudut relatif terhadap pusat spektrum:

\[
\theta(u,v)=\operatorname{mod}\left(\operatorname{atan2}(v-v_c,u-u_c),2\pi\right).
\]

Pada konfigurasi referensi, domain angular dibagi menjadi:

\[
K=360
\]

*directional bins*. Indeks bin dapat dinyatakan sebagai:

\[
b(u,v)=\left\lfloor\frac{\theta_{deg}(u,v)}{360^\circ/K}\right\rfloor.
\]

Dengan \(K=360\), resolusi nominal setiap bin adalah sekitar \(1^\circ\). Pada implementasi diskrit, koordinat pusat spektrum mengikuti konvensi binning operator dan dipetakan ke bin pertama.

Densitas angular untuk kanal \(c\) dihitung sebagai:

\[
D_i^c(k)=\sum_{(u,v):b(u,v)=k}A_i^c(u,v).
\]

Densitas tersebut kemudian dinormalisasi menjadi distribusi probabilitas:

\[
p_i^c(k)=\frac{D_i^c(k)}{\sum_jD_i^c(j)+\varepsilon},
\]

menggunakan:

\[
\varepsilon=10^{-8}.
\]

Konfigurasi referensi melakukan analisis tersebut secara independen pada kanal R, G, dan B.

### 3.4.4 Ambang Adaptif Berdasarkan Entropi

Entropi distribusi angular dihitung sebagai:

\[
H_i^c=-\sum_k p_i^c(k)\log\left(\max(p_i^c(k),\varepsilon\right).
\]

Nilai tersebut digunakan untuk membentuk ambang *patch-dependent*:

\[
\tau_i^c=\frac{\gamma}{1+\exp(-H_i^c)},
\]

dengan konfigurasi referensi:

\[
\gamma=0{,}10.
\]

Karena \(H_i^c\) dihitung secara terpisah dari distribusi spektral setiap patch, nilai \(\tau_i^c\) dapat berubah antar patch, walaupun rentang perubahannya tetap dibatasi oleh formulasi sigmoid dan nilai \(\gamma\). Operator ini tidak mempunyai parameter trainable.

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

Bobot dipetakan kembali ke koordinat Fourier:

\[
\widetilde F_i^c(u,v)=F_i^c(u,v)\,w_i^c(b(u,v)).
\]

Karena \(q_i^c(k)\in[0,1]\), tahap ini diperlakukan sebagai proses seleksi dan pembobotan spektral, bukan sebagai amplifikasi langsung koefisien Fourier di atas amplitudo asal.

### 3.4.6 Inverse Fourier Transform, Rekonstruksi, dan Residual Gate

Spektrum yang telah dibobotkan dikembalikan ke domain spasial menggunakan:

\[
\widetilde P_i^c=\Re\left\{\mathcal{F}_2^{-1}\left[\operatorname{ifftshift}(\widetilde F_i^c)\right]\right\}.
\]

Pada konfigurasi referensi, patch yang saling overlap direkonstruksi melalui *normalized overlap averaging*. Jika \(\Pi_i(\cdot)\) menyatakan penempatan patch ke-\(i\) ke koordinat citra, maka respons hasil rekonstruksi dapat dinyatakan secara konseptual sebagai:

\[
R_{FA}^c(x,y)=\frac{\sum_i\Pi_i(\widetilde P_i^c)(x,y)}{\sum_i\Pi_i(\mathbf{1})(x,y)+\varepsilon}.
\]

Implementasi menggunakan operasi *fold* dan pembagi jumlah kontribusi overlap. Respons spasial kemudian dinormalisasi secara terpisah untuk setiap citra dan kanal:

\[
G^c(x,y)=\frac{R_{FA}^c(x,y)-r_{min}^c}{\max(r_{max}^c-r_{min}^c,\varepsilon)},
\]

dengan:

\[
r_{min}^c=\min_{x,y}R_{FA}^c(x,y),
\qquad
r_{max}^c=\max_{x,y}R_{FA}^c(x,y).
\]

Dengan demikian:

\[
G^c(x,y)\in[0,1].
\]

Tensor keluaran dibentuk melalui residual gate:

\[
\boxed{I'^c=I^c+I^c\odot G^c}.
\]

Tidak diterapkan *clipping* tambahan setelah residual gate; tensor hasil transformasi diteruskan ke detector sesuai definisi operator. Operasi ini mempertahankan dimensi spasial tensor sehingga anotasi *bounding box* tidak perlu mengalami transformasi geometris. Sifat tersebut tidak mengasumsikan bahwa prediksi lokasi yang dihasilkan detector akan selalu identik antara kondisi baseline dan *preprocessing*.

## 3.5 Analisis dan Optimasi Desain Preprocessing

Istilah "optimasi" pada penelitian ini merujuk pada penyempurnaan empiris terhadap keputusan desain *preprocessing* pada ruang konfigurasi yang telah ditentukan, bukan pada pencarian *global optimum* dan bukan pada penambahan modul trainable ke YOLO26n. Analisis dilakukan melalui **staged cumulative spectral factorization**, yaitu setiap tahap memperkenalkan satu keputusan desain baru terhadap konfigurasi tahap sebelumnya, sedangkan keputusan yang telah diperkenalkan pada tahap sebelumnya dipertahankan.

Dengan demikian, jalur desain didefinisikan sebagai:

\[
C_0\rightarrow C_1\rightarrow C_2\rightarrow C_3\rightarrow C_4\rightarrow C_5.
\]

Konfigurasi dan perubahan pada setiap tahap ditunjukkan pada Tabel 3.2.

### Tabel 3.2 Jalur Staged Cumulative Spectral Factorization

| Konfigurasi | Perubahan terhadap tahap sebelumnya | Definisi utama | Pertanyaan analisis |
|---|---|---|---|
| \(C_0\) AF2-Ref | Konfigurasi referensi | Rectangular, 360 directional bins, angular-only, hard threshold, RGB independen | Membentuk reference transfer AFAB-2 pada input YOLO26n |
| \(C_1\) WIN | Windowing | Periodic square-root Hann + normalized overlap-add | Apakah tapering batas patch mengubah kualitas respons spektral? |
| \(C_2\) ORI | Representasi angular | 16 unsigned orientations modulo \(\pi\) | Bagaimana pengaruh representasi angular yang lebih ringkas? |
| \(C_3\) POL | Struktur radial-angular | 3 radial bands × 16 orientations | Apakah conditioning radial memberi informasi tambahan terhadap representation orientation? |
| \(C_4\) SOFT | Fungsi threshold | Sigmoid soft weighting | Bagaimana pengaruh transisi threshold yang kontinu terhadap respons dekat ambang? |
| \(C_5\) LUM | Panduan warna | Rec.709 luminance-derived shared spectral gate | Apakah keputusan spektral perlu bersifat spesifik kanal RGB? |

Perbedaan antara dua konfigurasi berurutan akan dianalisis sebagai kontribusi inkremental pada jalur desain tersebut. Pendekatan ini tidak dimaksudkan sebagai evaluasi faktorial lengkap terhadap seluruh interaksi antar faktor, sehingga konfigurasi akhir tidak akan disebut sebagai *global optimum*.

### 3.5.1 Variasi Windowing

Pada \(C_1\), rectangular patch diganti dengan *periodic square-root Hann window*. Fungsi satu dimensinya dinyatakan sebagai:

\[
h[n]=\sqrt{\frac{1}{2}-\frac{1}{2}\cos\left(\frac{2\pi n}{m}\right)},
\]

untuk \(n=0,\ldots,m-1\). Window dua dimensi dibentuk sebagai:

\[
W[p,q]=h[p]h[q].
\]

Window diterapkan pada tahap analisis sebelum FFT dan pada tahap sintesis setelah IFFT. Rekonstruksi overlap menggunakan denominator yang mempertimbangkan \(W^2\):

\[
R_{WIN}^c(x,y)=\frac{\sum_i\Pi_i\left(W\odot\widetilde P_i^c\right)(x,y)}{\sum_i\Pi_i(W^2)(x,y)+\varepsilon}.
\]

Variasi ini akan digunakan untuk mengevaluasi pengaruh tapering batas patch terhadap respons spektral tanpa mengubah detector.

### 3.5.2 Variasi Representasi Angular

Pada \(C_2\), representasi arah diubah dari 360 *directional bins* pada domain \([0,2\pi)\) menjadi 16 *unsigned orientation bins* pada domain \([0,\pi)\):

\[
\theta_o=\theta\bmod\pi.
\]

Jumlah bin adalah:

\[
K_o=16,
\]

dengan resolusi nominal:

\[
\Delta\theta=\frac{180^\circ}{16}=11{,}25^\circ.
\]

Variasi ini mengubah sekaligus representasi arah dan resolusi angular. Oleh karena itu, perbedaan kinerja tidak akan ditafsirkan sebagai akibat satu sifat matematis tunggal, tetapi sebagai pengaruh keseluruhan perubahan representasi angular tersebut.

### 3.5.3 Variasi Radial-Angular

Pada \(C_3\), informasi radial ditambahkan pada representasi orientation. Untuk grid Fourier patch, radius dinyatakan sebagai:

\[
r(u,v)=\sqrt{(u-u_c)^2+(v-v_c)^2}.
\]

Batas tiga radial band ditentukan dari kuantil geometri radius grid Fourier non-nol:

\[
\rho_1=Q_{1/3}(\{r(u,v):r(u,v)>0\}),
\]

\[
\rho_2=Q_{2/3}(\{r(u,v):r(u,v)>0\}).
\]

Dengan demikian, radial band dibentuk dari geometri grid FFT dan tidak dihitung dari statistik *training* atau *validation*. Densitas pada radial band ke-\(b\) dan orientation ke-\(k\) dihitung sebagai:

\[
D_i^c(b,k)=\sum_{(u,v)\in\Omega_{b,k}}A_i^c(u,v).
\]

Normalisasi probabilitas dan entropi kemudian dihitung secara terpisah dalam setiap radial band:

\[
p_i^c(b,k)=\frac{D_i^c(b,k)}{\sum_jD_i^c(b,j)+\varepsilon}.
\]

Dengan formulasi ini, \(C_3\) diperlakukan sebagai *radially conditioned angular representation*.

### 3.5.4 Variasi Soft Threshold

Pada \(C_4\), hard threshold diganti dengan pembobotan kontinu:

\[
w_{soft}(q,\tau)=q\,\sigma\left(\frac{q-\tau}{T}\right),
\]

dengan \(\sigma(\cdot)\) merupakan fungsi sigmoid dan parameter suhu dibekukan pada:

\[
T=0{,}02.
\]

Variasi ini akan digunakan untuk membandingkan transisi ambang diskrit pada konfigurasi hard threshold dengan transisi yang lebih halus di sekitar \(\tau\).

### 3.5.5 Variasi Luminance-Derived Shared Gate

Pada \(C_5\), spectral guide dibentuk dari luminance Rec.709:

\[
Y=0{,}2126R+0{,}7152G+0{,}0722B.
\]

Analisis spektral menghasilkan satu bobot dari kanal luminance, \(w_i^Y\), yang kemudian digunakan bersama pada ketiga kanal RGB:

\[
\widetilde F_i^c(u,v)=F_i^c(u,v)\,w_i^Y(u,v),
\qquad c\in\{R,G,B\}.
\]

Dengan demikian, tensor keluaran tetap mempunyai tiga kanal RGB; yang berubah adalah sumber keputusan gate dari kanal-spesifik menjadi *shared luminance-derived spectral gate*.

### 3.5.6 Sensitivity Analysis Terbatas

Setelah faktor struktural dianalisis, *sensitivity analysis* terbatas dapat dilakukan terhadap parameter referensi yang paling relevan, terutama ukuran patch \(m\) dan koefisien ambang \(\gamma\). Seluruh nilai kandidat akan ditetapkan sebelum *run* sensitivitas dilakukan dan hanya menggunakan data pengembangan. Hasil *locked test* tidak akan digunakan untuk memilih nilai parameter tersebut.

## 3.6 Rancangan Eksperimen

Eksperimen dirancang dalam tiga tahap agar fungsi baseline, pemilihan desain, dan konfirmasi akhir tidak tercampur.

### 3.6.1 Tahap I — Baseline Development

Tahap pertama akan membentuk baseline development YOLO26n pada bagian *training-validation* dari dataset primer yang telah dibekukan. Detector diinisialisasi dari bobot resmi `yolo26n.pt` dan dilatih menggunakan konfigurasi pada Subbab 3.7. Baseline development ini digunakan sebagai referensi pengembangan dan sebagai *common parent state* pada analisis faktor tahap berikutnya.

Sebelum pelatihan, operator dan dataset akan melalui pemeriksaan statis yang mencakup kesesuaian dimensi tensor, nilai *finite*, sifat deterministik operator, jumlah parameter trainable frontend, konsistensi konfigurasi, dukungan kelas pada setiap split, serta pencegahan akses terhadap *locked test*.

### 3.6.2 Tahap II — Spectral Design-Factor Screening

Tahap kedua digunakan untuk menganalisis jalur \(C_0\) sampai \(C_5\) pada data pengembangan. Seluruh konfigurasi screening akan dimulai secara independen dari *common baseline development checkpoint* yang sama, bukan dari checkpoint konfigurasi spektral sebelumnya. Dengan demikian, hubungan:

\[
C_0\rightarrow C_1\rightarrow\cdots\rightarrow C_5
\]

menunjukkan akumulasi keputusan desain operator, bukan pewarisan bobot model dari satu konfigurasi ke konfigurasi berikutnya.

Screening dilakukan menggunakan *seed* pengembangan yang sama dan data *validation* digunakan untuk analisis perbedaan antar tahap. Tahap ini berfungsi untuk memilih konfigurasi kandidat \(C^*\) yang akan dibawa ke konfirmasi berpasangan. Hasil tahap screening tidak diperlakukan sebagai estimasi final generalisasi metode.

### 3.6.3 Tahap III — Paired Multi-Seed Confirmation

Konfirmasi utama akan membandingkan tiga kondisi:

\[
N_s=\operatorname{Train}(\text{YOLO26n native},s),
\]

\[
A_s=\operatorname{Train}(C_0+\text{YOLO26n},s),
\]

\[
O_s=\operatorname{Train}(C^*+\text{YOLO26n},s),
\]

untuk:

\[
s\in\{42,123,2026\}.
\]

Pada setiap *seed*, ketiga kondisi dimulai langsung dari sumber *pretrained* resmi yang sama. Pembentukan head target \(C\) kelas menggunakan *seed* inisialisasi yang sama dan *persistent detector state* akan diverifikasi identik sebelum pelatihan. Karena frontend tidak mempunyai parameter trainable, satu-satunya perbedaan awal antar kondisi adalah operasi terhadap tensor masukan.

Untuk metrik \(M\), perbedaan berpasangan per *seed* akan dihitung sebagai:

\[
\Delta_s^{C_0}=M(A_s)-M(N_s),
\]

\[
\Delta_s^{C^*}=M(O_s)-M(N_s).
\]

Nilai per *seed*, rerata, dan variasi antar *seed* akan dilaporkan sehingga kesimpulan tidak hanya bergantung pada satu kondisi acak.

### 3.6.4 Locked Final Evaluation

Data *test* dipertahankan terkunci selama Tahap I dan Tahap II serta selama seluruh pemilihan parameter. *Locked test* dibentuk dari citra sumber primer yang tidak digunakan pada pengembangan dan seluruh sumber/kelompok akuisisinya harus independen dari *training-validation*.

Dengan target nominal sekitar 200 citra primer dan proporsi sekitar 15% untuk *test*, bagian *locked test* direncanakan memuat sekitar 30 citra sumber independen. Sebelum digunakan untuk evaluasi akhir, *test set* harus memenuhi seluruh kriteria berikut:

1. sekurang-kurangnya sekitar 30 citra sumber independen;
2. seluruh kelas pada taksonomi final \(C\) terdapat pada *test set*;
3. setiap kelas mempunyai sekurang-kurangnya 10 instance;
4. setiap kelas muncul pada sekurang-kurangnya 5 citra sumber independen;
5. tidak terdapat overlap *source/acquisition group* maupun *exact hash* dengan data pengembangan; dan
6. anotasi yang digunakan lolos pemeriksaan geometri dan konsistensi label.

Jika kriteria tersebut tidak terpenuhi, *test inference* tidak akan dipaksakan. Data primer tambahan akan dikumpulkan terlebih dahulu; apabila penambahan data tidak memungkinkan, alternatif yang telah ditetapkan adalah *grouped cross-validation* pada data pengembangan dengan keterbatasan tersebut dilaporkan secara eksplisit.

Setelah \(C^*\), *checkpoint selection rule*, metrik, dan prosedur evaluasi dibekukan, *locked test* akan dibuka satu kali untuk mengevaluasi checkpoint final dari kondisi native, \(C_0\), dan \(C^*\). Tidak dilakukan perubahan metode atau *hyperparameter* berdasarkan hasil *locked test*.

## 3.7 Konfigurasi Pelatihan

Konfigurasi pelatihan diringkas pada Tabel 3.3.

### Tabel 3.3 Konfigurasi Pelatihan YOLO26n

| Parameter | Factor Screening | Final Paired Confirmation |
|---|---:|---:|
| Detector | YOLO26n | YOLO26n |
| Parent weights | Common baseline development checkpoint | Official `yolo26n.pt`, matched per seed |
| Ukuran input | 640 × 640 piksel | 640 × 640 piksel |
| Epoch maksimum | 50 | 50 |
| Batch size | 16 | 16 |
| Workers | 2 | 2 |
| Patience | 15 | 15 |
| Optimizer | Auto | Auto |
| Cache | False | False |
| Close mosaic | 10 | 10 |
| Deterministic | Ya | Ya |
| Seed | 42 | 42, 123, 2026 |
| Evaluasi pengembangan | Validation | Validation |
| Akses locked test | Tidak | Setelah method freeze |

Seluruh konfigurasi yang dibandingkan pada tahap yang sama akan menggunakan dataset, augmentasi, ukuran input, maksimum epoch, aturan *early stopping*, batch size, optimizer, dan lingkungan perangkat yang sama. Nilai 50 merupakan maksimum epoch; model tidak diwajibkan berhenti pada epoch yang sama karena semua kondisi mengikuti aturan *early stopping* yang identik dengan `patience=15`.

Optimizer `Auto` akan digunakan pada versi Ultralytics yang dipin sehingga aturan pemilihannya tidak berubah antar kondisi. Parameter augmentasi mengikuti konfigurasi runtime pada versi yang sama dan akan direkam bersama setiap *run*. Pada *factor screening*, seluruh frontend dimulai dari parent detector development yang sama. Pada *final paired confirmation*, seluruh kondisi dimulai langsung dari sumber *pretrained* resmi dengan inisialisasi target yang dipasangkan.

Parameter `max_det=500` tidak diperlakukan sebagai konfigurasi pelatihan karena parameter tersebut digunakan pada tahap prediksi dan evaluasi sebagaimana dijelaskan pada Subbab 3.8.

## 3.8 Evaluasi Kinerja Deteksi

Evaluasi utama menggunakan metrik *object detection* berbasis *Average Precision*. *Precision* dan *recall* tetap dilaporkan sebagai metrik tambahan dan dirumuskan sebagai:

\[
Precision=\frac{TP}{TP+FP},
\]

\[
Recall=\frac{TP}{TP+FN}.
\]

*Average Precision* (AP) menghitung luas di bawah kurva *precision-recall* untuk suatu kelas. Metrik utama penelitian adalah:

\[
\boxed{mAP_{50:95}},
\]

yaitu rata-rata AP pada threshold IoU 0,50 sampai 0,95. Nilai \(mAP_{50}\) digunakan sebagai metrik sekunder. Seluruh evaluasi prediksi menggunakan `max_det=500` secara konsisten pada kondisi yang dibandingkan.

Selain metrik agregat, AP50–95 setiap kelas akan dilaporkan:

\[
AP_{c,50:95},\qquad c=1,\ldots,C.
\]

Untuk menganalisis kelas sulit secara konsisten, himpunan tiga kelas dengan AP50–95 terendah ditentukan satu kali dari baseline development pada *validation*:

\[
\mathcal{H}=\operatorname{Bottom3}(AP_{c,50:95}^{\text{baseline-dev}}).
\]

Setelah ditetapkan, \(\mathcal{H}\) dibekukan dan digunakan pada seluruh konfigurasi berikutnya. Rerata AP pada himpunan tersebut dihitung sebagai:

\[
AP_{\mathcal{H}}=\frac{1}{3}\sum_{c\in\mathcal{H}}AP_{c,50:95}.
\]

Selain itu, nilai kelas terendah dilaporkan sebagai *safety indicator*:

\[
AP_{worst}=\min_c AP_{c,50:95}.
\]

Metrik tail tidak digunakan sebagai pengganti \(mAP_{50:95}\), tetapi untuk menilai apakah perubahan agregat juga diikuti perubahan pada kelas yang sulit. Pada konfirmasi multi-seed, hasil akan dilaporkan per *seed* beserta rerata dan simpangan baku sampel. Apabila sumber daya memungkinkan, *paired grouped bootstrap* pada unit *source/acquisition group* akan digunakan sebagai analisis ketidakpastian tambahan pada evaluasi akhir.

## 3.9 Analisis Visual

Analisis visual digunakan sebagai pendukung evaluasi kuantitatif untuk membantu menginterpretasikan perubahan pada citra, respons spektral, aktivasi model, dan prediksi deteksi. Visualisasi tidak diperlakukan sebagai bukti kausal tunggal mengenai alasan peningkatan atau penurunan kinerja.

### 3.9.1 Visualisasi Tahapan Preprocessing

Untuk contoh citra yang dipilih, panel visual akan menampilkan:

1. tensor/citra masukan;
2. patch lokal yang dianalisis;
3. magnitude spektrum Fourier;
4. distribusi angular atau radial-angular;
5. ambang adaptif dan bobot spektral;
6. respons hasil *inverse Fourier transform*;
7. respons hasil rekonstruksi; dan
8. citra setelah residual gate.

Visualisasi ini digunakan untuk menunjukkan transformasi yang dilakukan operator. Perubahan kontras, tekstur, atau respons spektral yang terlihat tidak langsung dianggap sebagai bukti bahwa citra menjadi lebih baik bagi detector.

### 3.9.2 Visualisasi Respons Model

Visualisasi respons model direncanakan menggunakan metode *class activation mapping* yang kompatibel dengan YOLO26. Eigen-CAM dipertimbangkan karena pada formulasi aslinya tidak memerlukan *class-gradient backpropagation*. Metode CAM lain hanya akan digunakan apabila *target layer*, target deteksi, dan prosedur visualisasinya dapat didefinisikan secara konsisten pada seluruh kondisi.

Metode, *target layer*, ukuran input, dan prosedur normalisasi visualisasi yang dipilih akan diterapkan secara sama pada model native, \(C_0\), dan \(C^*\). Hasil CAM tidak digunakan sebagai pengganti metrik deteksi dan tidak ditafsirkan sebagai bukti bahwa model menggunakan fitur tertentu secara eksklusif.

### 3.9.3 Visualisasi Prediksi Deteksi

Hasil prediksi kondisi native, \(C_0\), dan \(C^*\) akan dibandingkan pada citra yang sama. Visualisasi mencakup *bounding box*, label kelas, dan skor kepercayaan.

Contoh visual akan dipilih berdasarkan kriteria yang ditetapkan sebelum inspeksi hasil, misalnya kelas dengan kinerja tinggi, kelas dengan kinerja rendah, kasus ketika seluruh model benar, seluruh model salah, dan kasus ketika model berbeda. Pendekatan ini digunakan untuk mengurangi kecenderungan hanya menampilkan contoh yang mendukung salah satu kondisi.

## 3.10 Analisis Kesalahan dan Kinerja Per Kelas

Analisis kesalahan dilakukan menggunakan AP per kelas, *confusion matrix*, *false positive*, dan *false negative*. Perbandingan antar kondisi digunakan untuk menilai kelas yang mengalami perubahan positif, relatif stabil, maupun negatif.

Selain analisis berbasis kinerja, kelas dapat dikelompokkan secara apriori berdasarkan jenis informasi visual yang diperlukan oleh definisi label, misalnya karakteristik permukaan/warna, jumlah detail lokal, bentuk/integritas, *relative completeness*, dan ukuran fisik. Pengelompokan tersebut ditetapkan dari definisi kelas sebelum hasil treatment diperiksa dan hanya digunakan sebagai analisis deskriptif mekanisme, bukan sebagai label tambahan pada proses training.

Apabila memungkinkan dari output evaluator, analisis juga membedakan kesalahan klasifikasi dan lokalisasi. Prediksi yang telah memiliki kecocokan spasial dengan ground truth tetapi menghasilkan kelas yang salah akan dianalisis terpisah dari kasus ketika objek tidak terlokalisasi dengan memadai. Analisis ini digunakan untuk menelaah apakah perubahan akibat *preprocessing* lebih banyak berkaitan dengan diskriminasi kelas atau dengan proses lokalisasi.

## 3.11 Evaluasi Efisiensi Komputasi

Meskipun *preprocessing* tidak menambahkan parameter trainable, operasi pembentukan patch, FFT, analisis angular/radial-angular, IFFT, dan rekonstruksi menambah biaya komputasi. Oleh karena itu, efisiensi akan dievaluasi pada tingkat sistem, bukan hanya berdasarkan jumlah parameter detector.

Pengukuran mencakup:

1. jumlah parameter trainable detector dan frontend;
2. latency *preprocessing* \(t_{FA}\);
3. latency detector \(t_{YOLO}\);
4. latency *end-to-end*;
5. throughput dalam citra per detik; dan
6. penggunaan memori GPU.

Latency *end-to-end* didefinisikan sebagai:

\[
t_{E2E}=t_{FA}+t_{YOLO}.
\]

Pengukuran baseline dan model dengan *preprocessing* dilakukan pada perangkat, ukuran input, batch size, dan presisi komputasi yang sama. Jumlah *warm-up*, pengulangan pengukuran, dan mekanisme sinkronisasi perangkat akan dibuat sama pada seluruh konfigurasi yang dibandingkan.

## 3.12 Lingkungan Implementasi

Implementasi penelitian menggunakan Python dan PyTorch melalui Ultralytics YOLO. Versi perangkat lunak akan dipin sebelum eksperimen utama; protokol penelitian menggunakan Ultralytics 8.4.96 sebagai versi referensi. Informasi versi Python, PyTorch, CUDA, perangkat GPU, sistem operasi, serta konfigurasi perangkat keras akan dicatat bersama setiap *run*.

Versi implementasi *preprocessing* juga akan dikunci melalui identitas *commit* kode sehingga konfigurasi operator yang digunakan pada setiap eksperimen dapat ditelusuri. Pencatatan lingkungan dan versi implementasi dilakukan untuk menjaga keterulangan eksperimen dan memudahkan verifikasi metodologi.