# BAB III
# METODOLOGI PENELITIAN

## 3.1 Rancangan Umum Penelitian

Penelitian ini menggunakan eksperimen komparatif untuk menganalisis pengaruh prapemrosesan citra berbasis frekuensi-angular terhadap kinerja YOLO26n pada deteksi *fine-grained* cacat biji kopi. Arsitektur YOLO26n dipertahankan pada perbandingan utama, sedangkan perlakuan eksperimen diberikan pada citra masukan.

Tahapan penelitian mencakup pengumpulan dan anotasi dataset primer, pembagian data berbasis kelompok sumber, pembentukan model acuan, pengujian variasi prapemrosesan, pemilihan konfigurasi $C^*$, pelatihan ulang dengan beberapa *seed* konfirmasi, evaluasi akhir, serta analisis per kelas, visual, kesalahan, dan efisiensi komputasi. Alur penelitian ditunjukkan pada Gambar 3.1.

![Alur penelitian](assets/alur_penelitian.svg){width=12.5cm}

Gambar 3.1 Alur Penelitian

## 3.2 Dataset Penelitian

### 3.2.1 Sumber dan Karakteristik Dataset Primer

Penelitian menggunakan **dataset primer** untuk deteksi objek multikelas pada biji kopi hijau. Daftar kelas awal menargetkan 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008 ditambah satu kelas biji normal:

$$
C_{target}=21.
$$

Jumlah kelas final ditetapkan setelah audit kecukupan data dan sebelum pembagian dataset serta pelatihan utama. Setiap citra direncanakan memuat banyak objek, sehingga skala dataset dinilai berdasarkan jumlah citra sumber, jumlah objek, distribusi per kelas, dan penyebaran pada kelompok sumber independen. Informasi asal fisik sampel, lot/batch, dan sumber pengadaan dicatat sesuai kondisi pengumpulan aktual.

### 3.2.2 Target Pengumpulan dan Pemeriksaan Kecukupan Data

Target pengumpulan adalah sekitar 180–220 citra sumber dengan sasaran nominal sekitar 200 citra asli. Setiap citra direncanakan memuat sekitar 30–50 objek dalam satu lapisan, sehingga pada sasaran nominal tersebut jumlah anotasi diperkirakan sekitar:

$$
N_{box}\approx 6.000-10.000.
$$

Sebagai target operasional, setiap kelas diupayakan memiliki sekurang-kurangnya sekitar 200 objek asli, dengan sasaran ideal sekitar 300–500 objek, serta muncul pada sekitar 30 citra sumber berbeda. Sebagai pembanding skala, Bahy dan Rifai (2026) melaporkan 107 citra dengan 13.863 anotasi untuk 20 kelas SNI, sedangkan Tarekegn dan Debelee (2025) menggunakan 562 citra dengan 19.228 objek untuk 13 kelas cacat dan satu kelas normal.

Sebelum pembagian data, distribusi setiap kelas diaudit berdasarkan:

$$
N_{obj,c},\qquad N_{img,c},\qquad N_{group,c},
$$

dengan $N_{obj,c}$ sebagai jumlah objek asli kelas $c$, $N_{img,c}$ jumlah citra sumber yang memuat kelas $c$, dan $N_{group,c}$ jumlah kelompok sumber independen yang mengandung kelas $c$.

Kelas yang belum memenuhi target pengumpulan diprioritaskan untuk penambahan data. Jika kecukupan tetap tidak terpenuhi, kelas tersebut tidak dimasukkan ke evaluasi utama kecuali terdapat dasar taksonomi yang membenarkan penggabungan. Jumlah kelas final ditetapkan sebelum pembagian dataset dan pelatihan:

$$
C\le C_{target},
$$

dengan target utama $C=21$ apabila seluruh kelas memiliki data yang memadai.

### 3.2.3 Akuisisi Citra dan Anotasi

Pengambilan citra direncanakan secara tegak lurus dari atas menggunakan latar belakang polos dan tidak reflektif, dengan posisi kamera, jarak, dan pencahayaan yang dikendalikan. Biji disusun dalam satu lapisan dengan orientasi bervariasi.

Setiap sesi pengambilan dan citra sumber memiliki identitas yang dapat ditelusuri. Citra dari sesi, susunan objek, atau kelompok spesimen fisik yang berkaitan diberi `group_id` yang sama. Satu `group_id` merepresentasikan unit sumber yang tidak boleh dipisah antar-*split*.

Kategori yang definisinya bergantung pada ukuran fisik dilengkapi referensi skala. Setiap objek diberi *bounding box* dan label kelas berdasarkan definisi operasional yang ditetapkan sebelum anotasi. Sampel yang meragukan ditinjau ulang, dan validasi label direncanakan melibatkan praktisi atau validator yang memahami penilaian fisik mutu kopi.

### 3.2.4 Pembagian Data dan Pencegahan Kebocoran

Dataset dibagi sebelum augmentasi dengan target sekitar 70% pelatihan, 15% validasi, dan 15% pengujian. Proporsi dapat bergeser sedikit untuk menjaga keterwakilan kelas tanpa melanggar pemisahan kelompok sumber.

Pembagian dilakukan berdasarkan `group_id`. Seluruh citra dari sumber atau spesimen yang berkaitan ditempatkan pada bagian data yang sama, sehingga:

$$
\mathcal{G}_{train}\cap\mathcal{G}_{val}
=\mathcal{G}_{train}\cap\mathcal{G}_{test}
=\mathcal{G}_{val}\cap\mathcal{G}_{test}
=\varnothing.
$$

Pada validasi, setiap kelas ditargetkan muncul pada sedikitnya sekitar lima citra sumber. Pada data uji, target operasional adalah sedikitnya 10 objek per kelas yang tersebar pada sedikitnya lima citra sumber. Pemeriksaan *hash* digunakan sebagai lapisan tambahan untuk mendeteksi file identik.

Data validasi digunakan untuk penghentian dini, pembandingan konfigurasi, analisis sensitivitas, dan pemilihan $C^*$. Data uji disisihkan sampai konfigurasi akhir dan prosedur evaluasi dibekukan.

### 3.2.5 Augmentasi Data

Augmentasi hanya diterapkan pada data pelatihan setelah pembagian sumber selesai. Validasi dan pengujian menggunakan citra asli. Konfigurasi augmentasi dibuat sama pada seluruh kondisi YOLO26n yang dibandingkan. CLAHE dan prapemrosesan frekuensi-angular diperlakukan sebagai perlakuan eksperimen pada citra masukan, bukan sebagai augmentasi dataset.

## 3.3 Model Dasar YOLO26n

YOLO26n digunakan sebagai model utama dengan bobot pralatih resmi `yolo26n.pt`. Setelah jumlah kelas final ditetapkan, bagian keluaran disesuaikan dengan $C$. *Backbone*, *neck*, dan *detection head* tidak dimodifikasi pada eksperimen utama, dan prapemrosesan tidak menambahkan parameter trainable.

### 3.3.1 Kondisi Eksperimen Utama dan Pembanding

Empat kondisi utama adalah:

| Kode | Kondisi | Peran dalam eksperimen |
|---|---|---|
| $B_0$ | YOLO26n tanpa prapemrosesan tambahan | Kondisi acuan |
| $B_1$ | CLAHE + YOLO26n | Kontrol peningkatan kontras lokal konvensional |
| $B_2$ | $C_0$ + YOLO26n | Konfigurasi referensi frekuensi-angular |
| $B_3$ | $C^*$ + YOLO26n | Konfigurasi frekuensi-angular terpilih |

CLAHE diterapkan pada kanal luminansi dengan *clip limit* 2,0 dan kisi 8 × 8 sebagai kontrol tetap. Perbandingan $B_2-B_0$ mengukur efek *frontend* frekuensi-angular referensi, $B_3-B_2$ efek optimasi desain, dan $B_3-B_1$ perbedaan terhadap peningkatan kontras lokal. Jika konfigurasi referensi menjadi yang terbaik, maka $C^*=C_0$.

## 3.4 Prapemrosesan Citra Berbasis Frekuensi-Angular

Prapemrosesan mengadaptasi mekanisme angular AFAB-2 dari Xu et al. (2025), yaitu analisis frekuensi lokal per patch, pembentukan distribusi angular, ambang adaptif berbasis entropi, pembobotan amplitudo, rekonstruksi dengan fase asli, dan penggabungan residual. Diskretisasi angular, pemrosesan per kanal RGB, overlap patch, padding, penggabungan patch, konvensi DC, dan variasi $C_1$ sampai $C_5$ merupakan keputusan adaptasi penelitian.

Pada pipeline YOLO, citra telah menjadi tensor RGB *floating point* pada rentang $[0,1]$ sebelum masuk model. *Frontend* frekuensi-angular ditempatkan setelah augmentasi umum dan sebelum detektor. Kontrak keluaran mengikuti Subbab 3.4.6.

![Arsitektur integrasi prapemrosesan frekuensi-angular dengan YOLO26n](assets/arsitektur_frekuensi_yolo26.svg){width=12.5cm}

Gambar 3.2 Arsitektur Integrasi Prapemrosesan Frekuensi-Angular dengan YOLO26n

### 3.4.1 Pembentukan Patch Lokal

Untuk citra RGB:

$$
I\in\mathbb{R}^{3\times H\times W},
$$

citra dibagi menjadi patch:

$$
P_i\in\mathbb{R}^{3\times m\times m}.
$$

Konfigurasi referensi menggunakan $m=32$ dan overlap 50%, sehingga:

$$
s=16.
$$

Ukuran patch mengikuti konfigurasi referensi Xu et al. (2025), sedangkan overlap 50% merupakan keputusan implementasi penelitian. *Replicate padding* digunakan bila diperlukan dan dipotong kembali setelah rekonstruksi sehingga keluaran tetap berukuran $H\times W$.

### 3.4.2 Transformasi Fourier

Untuk patch ke-$i$ dan kanal ke-$c$, transformasi Fourier ortonormal dihitung lalu pusat spektrum dipindahkan ke tengah dengan *FFT shift*:

$$
F_i^c(u,v)=\mathrm{fftshift}\!\left(\mathcal{F}_{2,\mathrm{ortho}}\{P_i^c\}\right)(u,v).
$$

Amplitudo dan fase pada spektrum terpusat dihitung sebagai:

$$
A_i^c(u,v)=|F_i^c(u,v)|,
$$

$$
\phi_i^c(u,v)=\arg F_i^c(u,v).
$$

Amplitudo digunakan dalam analisis spektral, sedangkan fase dipertahankan untuk rekonstruksi. Sebelum transformasi balik, spektrum terpusat dikembalikan ke susunan indeks FFT dengan *inverse FFT shift* sebagaimana dituliskan pada Subbab 3.4.6.

### 3.4.3 Distribusi Angular

Sudut koordinat frekuensi dihitung relatif terhadap pusat spektrum:

$$
\theta(u,v)=\mathrm{atan2}(v-v_c,u-u_c)\bmod 2\pi.
$$

Pada $C_0$, domain $[0,2\pi)$ dibagi menjadi 360 interval angular. Densitas kanal $c$ dihitung sebagai:

$$
D_i^c(k)=\sum_{(u,v):b(u,v)=k}A_i^c(u,v),
$$

kemudian dinormalisasi:

$$
p_i^c(k)=\frac{D_i^c(k)}{\sum_jD_i^c(j)+\varepsilon},
$$

$$
\varepsilon=10^{-8}.
$$

Perhitungan dilakukan per kanal RGB. Koordinat pusat spektrum dipetakan ke bin angular 0 sebagai konvensi diskret untuk komponen DC.

### 3.4.4 Ambang Adaptif Berdasarkan Entropi

Entropi distribusi angular dihitung sebagai:

$$
H_i^c=-\sum_k p_i^c(k)\log\left(\max(p_i^c(k),\varepsilon)\right).
$$

Ambang adaptif ditentukan dengan:

$$
\tau_i^c=\frac{\gamma}{1+\exp(-H_i^c)},
$$

dengan konfigurasi referensi:

$$
\gamma=0{,}10.
$$

Karena $H_i^c\ge0$:

$$
\frac{\gamma}{2}\le\tau_i^c<\gamma.
$$

Nilai $\gamma$ diuji lebih lanjut pada analisis sensitivitas.

### 3.4.5 Pembobotan Respons Spektral

Densitas angular dinormalisasi terhadap nilai maksimum:

$$
q_i^c(k)=\frac{D_i^c(k)}{\max_jD_i^c(j)+\varepsilon}.
$$

Pada $C_0$, digunakan ambang keras:

$$
w_i^c(k)=
\begin{cases}
0, & q_i^c(k)\le\tau_i^c,\\
q_i^c(k), & q_i^c(k)>\tau_i^c.
\end{cases}
$$

Bobot diterapkan pada koefisien Fourier berdasarkan bin angular:

$$
\widetilde F_i^c(u,v)=F_i^c(u,v)\,w_i^c(b(u,v)).
$$

Karena $0\le w_i^c(k)\le1$, tahap ini tidak memperbesar amplitudo Fourier di atas nilai asal. Komponen DC mengikuti bobot bin 0.

### 3.4.6 Rekonstruksi dan Penggabungan Residual

Spektrum berbobot dikembalikan ke susunan indeks FFT sebelum transformasi balik:

$$
\widetilde P_i^c=
\mathrm{Re}\left\{
\mathcal{F}_{2,\mathrm{ortho}}^{-1}
\left[\mathrm{ifftshift}(\widetilde F_i^c)\right]
\right\}.
$$

Pada $C_0$, kontribusi patch yang bertumpang tindih dirata-ratakan menjadi respons $R_{FA}$. Respons tersebut dinormalisasi per citra dan kanal:

$$
G^c(x,y)=\frac{R_{FA}^c(x,y)-r_{min}^c}{\max(r_{max}^c-r_{min}^c,\varepsilon)},
$$

sehingga:

$$
G^c(x,y)\in[0,1].
$$

Citra keluaran dibentuk melalui residual:

$$
\boxed{I'^c=I^c+I^c\odot G^c}.
$$

Setelah residual tidak dilakukan *clipping* atau renormalisasi tambahan. Untuk $I\in[0,1]$, rentang teoritis keluaran adalah $[0,2]$. Ukuran spasial tetap $H\times W$. Karena kontrak ini mengubah distribusi nilai masukan sekaligus respons spektralnya, perbandingan terhadap $B_0$ diinterpretasikan sebagai efek *frontend* frekuensi-angular secara keseluruhan.

## 3.5 Analisis Variasi Desain Prapemrosesan

Optimasi dilakukan secara kumulatif dari $C_0$ sampai $C_5$, dengan satu perubahan utama ditambahkan pada setiap tahap:

$$
C_0\rightarrow C_1\rightarrow C_2\rightarrow C_3\rightarrow C_4\rightarrow C_5.
$$

Seluruh konfigurasi menggunakan kontrak residual $C_0$ yang sama.

### Tabel 3.2 Variasi Desain Prapemrosesan

| Konfigurasi | Perubahan utama | Tujuan pengujian |
|---|---|---|
| $C_0$ | Konfigurasi frekuensi-angular referensi | Menjadi acuan prapemrosesan |
| $C_1$ | Fungsi jendela Hann | Menguji pengaruh batas patch |
| $C_2$ | Orientasi tak bertanda dengan resolusi sudut tetap | Menguji arah dan orientasi tanpa mengubah resolusi angular nominal |
| $C_3$ | Tiga pita radial | Menguji seleksi angular pada wilayah radial berbeda |
| $C_4$ | Ambang lunak | Menguji pembobotan bertahap di sekitar ambang |
| $C_5$ | Panduan luminansi | Menguji kebutuhan pembobotan terpisah pada setiap kanal warna |

### 3.5.1 Variasi Fungsi Jendela

Pada $C_1$, digunakan *periodic square-root Hann window*:

$$
h[n]=\sqrt{\frac{1}{2}-\frac{1}{2}\cos\left(\frac{2\pi n}{m}\right)},
$$

$$
W[p,q]=h[p]h[q].
$$

Patch analisis dibentuk sebagai:

$$
P_{i,a}=P_i\odot W.
$$

Setelah transformasi balik, jendela yang sama diterapkan kembali dan patch digabung dengan *normalized overlap-add*, yaitu membagi jumlah respons pada setiap posisi dengan jumlah bobot $W^2$ yang menutup posisi tersebut. $C_1$ dan konfigurasi berikutnya menggunakan *replicate padding* sebelum pembentukan patch.

### 3.5.2 Variasi Representasi Orientasi

Pada $C_2$, arah bertanda diubah menjadi orientasi tak bertanda:

$$
\theta_o=\theta\bmod\pi.
$$

Rentang $[0,\pi)$ dibagi menjadi 180 interval sehingga resolusi angular nominal tetap:

$$
\Delta\theta=\frac{180^\circ}{180}=1^\circ.
$$

Dua arah yang berbeda $180^\circ$ diperlakukan sebagai orientasi yang sama. Konvensi DC diwarisi dari $C_0$ dan ditempatkan pada bin orientasi 0.

### 3.5.3 Variasi Radial-Angular

Pada $C_3$, informasi radial ditambahkan pada representasi orientasi. Radius dan radius ternormalisasi dihitung sebagai:

$$
r(u,v)=\sqrt{(u-u_c)^2+(v-v_c)^2},
$$

$$
\rho(u,v)=\frac{r(u,v)}{r_{max}},\qquad 0\le\rho\le1.
$$

Spektrum dibagi menjadi tiga pita:

$$
\mathcal{R}_1: 0\le\rho\le\frac{1}{3},
$$

$$
\mathcal{R}_2: \frac{1}{3}<\rho\le\frac{2}{3},
$$

$$
\mathcal{R}_3: \frac{2}{3}<\rho\le1.
$$

Komponen DC ditempatkan pada $\mathcal R_1$ dan bin orientasi 0. Dengan $\ell\in\{1,2,3\}$ sebagai indeks pita, densitas dihitung sebagai:

$$
D_i^c(\ell,k)=\sum_{(u,v)\in\Omega_{\ell,k}}A_i^c(u,v).
$$

Normalisasi, entropi, ambang, dan densitas relatif dihitung terpisah pada setiap pita:

$$
p_i^c(\ell,k)=\frac{D_i^c(\ell,k)}{\sum_jD_i^c(\ell,j)+\varepsilon},
$$

$$
H_i^c(\ell)=-\sum_k p_i^c(\ell,k)\log\left(\max(p_i^c(\ell,k),\varepsilon)\right),
$$

$$
\tau_i^c(\ell)=\frac{\gamma}{1+\exp(-H_i^c(\ell))},
$$

$$
q_i^c(\ell,k)=\frac{D_i^c(\ell,k)}{\max_jD_i^c(\ell,j)+\varepsilon}.
$$

Dengan normalisasi per pita, seleksi angular dilakukan relatif terhadap respons di masing-masing wilayah radial.

### 3.5.4 Variasi Ambang Lunak

Pada $C_4$, ambang keras diganti dengan:

$$
w_{soft}(q,\tau)=q\,\sigma\left(\frac{q-\tau}{T}\right).
$$

Nilai awal parameter transisi adalah:

$$
T=0{,}02.
$$

Karena $0\le w_{soft}\le q\le1$, amplitudo Fourier tetap tidak diperbesar di atas nilai asal.

### 3.5.5 Variasi Panduan Luminansi

Pada $C_5$, bobot spektral dihitung dari luminansi berdasarkan ITU-R BT.709-6 (International Telecommunication Union, 2015):

$$
Y=0{,}2126R+0{,}7152G+0{,}0722B.
$$

Satu gate luminansi $M_Y$ diterapkan pada ketiga kanal:

$$
R'=R(1+M_Y),\qquad G'=G(1+M_Y),\qquad B'=B(1+M_Y).
$$

Dengan gate yang sama dan tanpa *clipping* pasca-residual, rasio antar-kanal lokal dipertahankan selama penyebut tidak nol.

### 3.5.6 Analisis Sensitivitas Terbatas

Setelah struktur $C_0$ sampai $C_5$ dibandingkan, analisis sensitivitas dilakukan satu parameter pada satu waktu terhadap kandidat:

$$
m\in\{16,32,64\},
$$

$$
\gamma\in\{0{,}05,0{,}10,0{,}15\},
$$

serta, apabila menggunakan ambang lunak:

$$
T\in\{0{,}01,0{,}02,0{,}05\}.
$$

Overlap tetap 50% sehingga:

$$
s=\frac{m}{2}.
$$

Seluruh kandidat ditetapkan sebelum eksperimen sensitivitas dan dievaluasi pada data pengembangan. Nilai terbaik dari parameter berbeda tidak digabungkan menjadi konfigurasi baru tanpa evaluasi konfigurasi tersebut secara langsung.

## 3.6 Rancangan Eksperimen

Eksperimen dibagi menjadi tahap pengembangan, pemilihan konfigurasi, pelatihan ulang dengan *seed* konfirmasi, dan evaluasi akhir. Seluruh run YOLO26n pada tahap utama dimulai dari `yolo26n.pt` dengan prosedur inisialisasi keluaran yang setara; tidak ada pewarisan checkpoint antarkondisi.

### 3.6.1 Tahap I — Pembentukan Model Acuan Pengembangan

Model acuan pengembangan $B_0^{dev}$ dilatih dengan:

$$
s_{dev}=42.
$$

Model ini digunakan untuk menetapkan kinerja dasar validasi, menentukan kelompok tiga kelas sulit $\mathcal H$, dan memeriksa pipeline data serta evaluasi.

### 3.6.2 Tahap II — Pengujian Variasi Prapemrosesan

Konfigurasi $C_0$ sampai $C_5$ dilatih dengan *seed* 42. Untuk setiap konfigurasi, didefinisikan:

$$
m_j=mAP_{50:95}^{val}(C_j),\qquad
m_{max}=\max_j m_j.
$$

Konfigurasi yang berada kurang dari 0,001 dari nilai validasi tertinggi dimasukkan ke himpunan kandidat:

$$
\mathcal{C}_{tie}=\{C_j\mid m_{max}-m_j<0{,}001\}.
$$

Jika $\mathcal{C}_{tie}$ hanya berisi satu konfigurasi, konfigurasi tersebut menjadi $C_{str}$. Jika terdapat lebih dari satu kandidat, dipilih konfigurasi dengan $AP_{\mathcal H}$ tertinggi; jika masih seri, digunakan median latency *end-to-end* yang lebih rendah berdasarkan Subbab 3.11.

Analisis sensitivitas pada Subbab 3.5.6 dilakukan setelah $C_{str}$ ditetapkan. Konfigurasi akhir $C^*$ dipilih dari konfigurasi yang benar-benar telah dievaluasi menggunakan urutan kriteria yang sama, kemudian dibekukan sebelum tahap konfirmasi dan pengujian.

### 3.6.3 Tahap III — Pelatihan Ulang dengan Beberapa Seed Konfirmasi

Kondisi utama pada Subbab 3.3.1 dilatih ulang menggunakan:

$$
S_{conf}=\{123,2026,31415\}.
$$

Jika $C^*=C_0$, run duplikat $B_2$ dan $B_3$ tidak dilakukan. Hasil setiap *seed* dilaporkan secara berpasangan terhadap baseline menggunakan prosedur pada Subbab 3.8. Seed 42 dilaporkan terpisah sebagai hasil pengembangan.

### 3.6.4 Evaluasi pada Arsitektur Lain — Opsional

Sebagai analisis tambahan, RT-DETRv3-R18 dapat dibandingkan tanpa prapemrosesan dan dengan $C^*$. Konfigurasi $C^*$ tidak dituning ulang, sedangkan konfigurasi pelatihan spesifik RT-DETR dibuat sama pada kedua kondisi. Analisis ini tidak digunakan untuk memilih ulang $C^*$.

### 3.6.5 Evaluasi Akhir pada Data Uji

Data uji mengikuti pembagian dan kriteria keterwakilan pada Subbab 3.2.4. Setelah $C^*$, seed konfirmasi, aturan checkpoint, metrik, dan prosedur evaluasi dibekukan, checkpoint dari setiap seed dalam $S_{conf}$ dievaluasi pada data uji yang sama. Tidak dilakukan perubahan metode atau pemilihan ulang konfigurasi berdasarkan hasil uji.

## 3.7 Konfigurasi Pelatihan

Konfigurasi utama YOLO26n ditunjukkan pada Tabel 3.3.

### Tabel 3.3 Konfigurasi Utama Pelatihan YOLO26n

| Parameter | Nilai |
|---|---|
| Model | YOLO26n |
| Bobot awal | `yolo26n.pt` pralatih resmi |
| Ukuran masukan | 640 × 640 piksel |
| Epoch maksimum | 50 pada preflight baseline; nilai final dibekukan sebelum eksperimen utama |
| Ukuran batch | 16 sebagai target; satu nilai final bersama dibekukan sebelum eksperimen utama |
| Penghentian dini | 15 epoch tanpa peningkatan $mAP_{50:95}^{val}$ |
| Optimizer | Auto pada Ultralytics 8.4.96; ter-resolve ke AdamW pada rancangan ini |
| Seed pengembangan | 42 |
| Seed konfirmasi | 123, 2026, 31415 |

Preflight dilakukan pada baseline pengembangan sebelum eksperimen utama. Jika batas 50 epoch masih memotong tren perbaikan validasi, satu batas epoch yang lebih tinggi ditetapkan dan kemudian digunakan pada seluruh kondisi. Jika batch 16 tidak dapat digunakan secara konsisten karena keterbatasan memori, satu ukuran batch yang layak dipilih dan dibekukan untuk seluruh kondisi utama. Setelah eksperimen utama dimulai, kedua nilai tersebut tidak diubah antarkondisi.

Eksperimen menggunakan Ultralytics 8.4.96. Pada versi ini, `best.pt` dan penghentian dini untuk deteksi mengikuti *fitness* yang sama dengan $mAP_{50:95}$. `optimizer=Auto` ter-resolve ke AdamW pada rancangan ini; optimizer, learning rate aktual, dan parameter implementasi penting dicatat pada setiap run.

## 3.8 Evaluasi Kinerja Deteksi

Metrik utama penelitian adalah:

$$
\boxed{mAP_{50:95}},
$$

yaitu rata-rata AP pada IoU 0,50:0,05:0,95. Nilai $mAP_{50}$ digunakan sebagai metrik sekunder. Precision dan *recall* juga dilaporkan:

$$
Precision=\frac{TP}{TP+FP},
$$

$$
Recall=\frac{TP}{TP+FN}.
$$

Seluruh kondisi dievaluasi dengan Ultralytics 8.4.96 dan `max_det=500`. Precision dan *recall* ringkasan mengikuti operating point evaluator dan diperlakukan sebagai metrik deskriptif sekunder.

AP50–95 setiap kelas dilaporkan sebagai:

$$
AP_{c,50:95},\qquad c=1,\ldots,C.
$$

Kelompok tiga kelas sulit ditetapkan dari baseline pengembangan pada data validasi:

$$
\mathcal{H}=\mathrm{Bottom3}\left(AP_{c,50:95}^{val}(B_0^{dev})\right),
$$

kemudian dibekukan. Rerata AP kelompok tersebut adalah:

$$
AP_{\mathcal H}=\frac{1}{3}\sum_{c\in\mathcal H}AP_{c,50:95}.
$$

AP kelas terendah dilaporkan sebagai:

$$
AP_{worst}=\min_c AP_{c,50:95}.
$$

Pada seed konfirmasi, setiap metrik dilaporkan per seed beserta rata-rata dan simpangan baku. Selisih terhadap baseline dihitung secara berpasangan:

$$
\Delta_s=M_{perlakuan,s}-M_{B_0,s},
$$

beserta $\overline{\Delta}$ dan $SD(\Delta)$ pada $S_{conf}$.

Jika jumlah kelompok independen pada data uji memungkinkan, ketidakpastian perbedaan antar kondisi dianalisis dengan *paired bootstrap* berbasis `group_id`. Kelompok yang sama di-*resample* untuk seluruh kondisi; selisih dihitung per seed dan dirata-ratakan pada replikasi yang sama.

## 3.9 Analisis Visual

Analisis visual digunakan sebagai pendukung interpretasi hasil kuantitatif, bukan sebagai bukti kausal tunggal.

### 3.9.1 Visualisasi Tahapan Prapemrosesan

Untuk citra sumber yang sama, visualisasi konfigurasi referensi dan konfigurasi terpilih dapat mencakup:

1. citra masukan;
2. patch lokal;
3. amplitudo spektrum Fourier;
4. distribusi angular atau radial-angular;
5. ambang dan bobot spektral;
6. hasil transformasi Fourier balik;
7. respons rekonstruksi; dan
8. citra setelah penggabungan residual.

### 3.9.2 Visualisasi Respons Model

Eigen-CAM menjadi kandidat utama apabila kompatibel dengan YOLO26 dan target layer yang dipilih; jika tidak, digunakan metode aktivasi lain yang dapat diterapkan konsisten pada seluruh kondisi. Metode, target layer, ukuran masukan, dan normalisasi heatmap dibuat sama. Seed visualisasi ditetapkan sebagai:

$$
s_{vis}=123.
$$

### 3.9.3 Visualisasi Hasil Deteksi

Hasil $B_0$, $B_1$, $B_2$, dan $B_3$ dibandingkan pada citra yang sama dengan parameter prediksi yang sama, terutama *confidence threshold* dan `max_det`. Jika $C^*=C_0$, kondisi identik tidak ditampilkan dua kali.

Contoh mencakup kasus seluruh model benar, seluruh model salah, perbedaan hasil antar kondisi, serta kelas dengan kinerja relatif tinggi atau rendah berdasarkan evaluasi yang dilaporkan.

## 3.10 Analisis Kesalahan dan Kinerja Per Kelas

Analisis per kelas menggunakan $AP_{c,50:95}$, matriks kebingungan, *false positive*, dan *false negative*. Perubahan AP terhadap baseline pada seed konfirmasi dihitung sebagai:

$$
\Delta AP_{c,s}=AP_{c,s}^{perlakuan}-AP_{c,s}^{B_0}.
$$

Rerata perubahan per kelas dihitung pada $S_{conf}$. Matriks kebingungan, FP, dan FN menggunakan prosedur evaluator dan konfigurasi prediksi yang sama pada seluruh kondisi.

Kelas dapat dikelompokkan secara deskriptif berdasarkan karakteristik visual utama, seperti permukaan/warna, detail lokal, bentuk/integritas, tingkat keutuhan, atau ukuran fisik. Pemetaan tersebut ditetapkan sebelum hasil eksperimen diperiksa dan tidak digunakan untuk memilih konfigurasi.

Contoh kesalahan ditinjau untuk mengidentifikasi kebingungan kelas, ketidaktepatan lokalisasi, objek terlewat, dan prediksi tambahan. Analisis ini bersifat deskriptif dan tidak membentuk metrik evaluasi baru.

## 3.11 Evaluasi Efisiensi Komputasi

Efisiensi diukur pada tingkat sistem dengan masukan $640\times640$, *batch* 1, perangkat, dan presisi komputasi yang sama. Dilaporkan waktu prapemrosesan $t_{pra}$, waktu inferensi model $t_{model}$, dan latency *end-to-end* $t_{total}$.

Nilai $t_{total}$ diukur langsung dari antarmuka masukan umum sampai prediksi selesai. Seluruh operasi tambahan yang khusus diperlukan oleh suatu frontend, termasuk konversi dtype/ruang warna dan perpindahan CPU–GPU, dimasukkan ke biaya metode. I/O disk yang identik tidak dimasukkan.

Benchmark menggunakan jumlah *warm-up* dan pengulangan yang sama. Operasi GPU disinkronkan pada batas pengukuran, sedangkan pipeline yang melibatkan CPU diukur dengan *wall-clock timing*. Median $t_{total}$ digunakan sebagai ukuran utama efisiensi dan *tie-break* pada Subbab 3.6.2; variasi latency serta throughput/FPS dari protokol yang sama juga dilaporkan.

Jumlah parameter model dan *peak allocated GPU memory* dilaporkan sebagai informasi tambahan. Jika RT-DETR dievaluasi, hasil efisiensinya dilaporkan terpisah.

## 3.12 Lingkungan Implementasi

Implementasi menggunakan Python, PyTorch, dan Ultralytics 8.4.96. Versi Python, PyTorch, CUDA/driver, sistem operasi, CPU, GPU, RAM, dan perangkat keras utama dicatat.

Setiap run ditautkan dengan *commit* kode eksperimen, konfigurasi run, dan manifest pembagian dataset berbasis `group_id`. Seed serta pengaturan reproducibility Python/PyTorch/CUDA diterapkan secara konsisten pada seluruh kondisi. Jika RT-DETRv3-R18 digunakan, versi kode, bobot pralatih, dan konfigurasi pelatihannya dicatat terpisah.
