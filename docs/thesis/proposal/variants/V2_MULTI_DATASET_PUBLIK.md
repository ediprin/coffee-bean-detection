# V2 — Multi-Dataset Publik

Status: **WORKING VARIANT — NOT FORMAL**

Dokumen ini merupakan alternatif terhadap rancangan dataset primer pada BAB III. Selama belum dipromosikan secara eksplisit, source formal proposal tetap menggunakan V1.

## 1. Prinsip Utama

V2 menggunakan beberapa dataset publik deteksi cacat biji kopi hijau untuk menguji apakah efek prapemrosesan frekuensi-angular konsisten pada lebih dari satu sumber data.

Dataset **tidak langsung digabung menjadi satu dataset besar** karena perbedaan taksonomi, asal citra, anotasi, versi, dan kemungkinan hubungan fork/derivatif. Setiap dataset diperlakukan sebagai benchmark independen dengan ruang kelasnya sendiri.

Dengan himpunan dataset publik yang lolos audit:

$$
\mathcal D=\{D_1,D_2,\ldots,D_K\},
$$

setiap dataset $D_d$ memiliki ruang kelas:

$$
\mathcal C_d=\{1,2,\ldots,C_d\}.
$$

Tidak disyaratkan bahwa:

$$
\mathcal C_1=\mathcal C_2=\cdots=\mathcal C_K.
$$

Dengan demikian, perbandingan utama dilakukan **di dalam dataset**, lalu efek metode dibandingkan lintas dataset melalui delta terhadap baseline.

---

## 2. Candidate Pool Dataset Publik

Candidate pool awal hanya berfungsi sebagai daftar audit. Dataset belum dianggap masuk eksperimen sampai provenance, lisensi, versi, duplikasi, dan anotasinya diperiksa.

| Kandidat | Task | Snapshot yang terlihat | Lisensi yang tercantum | Status awal |
|---|---|---:|---|---|
| Green Coffee Bean Defects — `jimmy-74920/green-coffee-bean-defects-pmxsf` | Object detection | 860 citra, 18 kelas | CC BY 4.0 | Kandidat kuat; perlu audit provenance dan versi |
| Coffee Bean Defect — `capstone-2-wwe5t/coffee-bean-defect-a0vno` | Object detection | overview 966 citra, 14 kelas; salah satu versi model mencantumkan 1.500 citra | CC BY 4.0 | Kandidat; jumlah citra/version lineage harus dikunci |
| green coffee bean defects — `lulus-vpibo/green-coffee-bean-defects` | Object detection | 1.002 citra, 6 kelas | CC BY 4.0 | Kandidat; kelas berorientasi SNI-like |
| coffee-bean-defects — `niacubilla/coffee-bean-defects` | Object detection | 1.800 citra, 9 kelas | CC BY 4.0 | Kandidat; perlu audit kemungkinan fork/derivatif |
| Coffee Green Bean Defect — `dwi-adityaa/coffee-green-bean-defect-xdmvu` | Object detection | 421 citra, 8 kelas | CC BY 4.0 | Kandidat sekunder; label generik `objects` harus diselesaikan sebelum dapat dipakai |

Dataset klasifikasi-only, termasuk dataset satu-biji 17 kelas, **tidak masuk benchmark utama deteksi** hanya karena kelasnya relevan. Dataset seperti itu hanya dapat dipakai sebagai analisis tambahan bila tersedia anotasi lokasi yang sah atau prosedur konversi yang dapat dipertanggungjawabkan dan dibekukan sebelum eksperimen.

### Gate minimum V2

V2 dianggap layak sebagai rancangan utama jika sekurang-kurangnya tiga lineage dataset deteksi yang benar-benar independen lolos audit:

$$
K_{final}\ge3.
$$

Satu dataset digunakan untuk pengembangan/pemilihan $C^*$ dan sedikitnya dua dataset lain digunakan sebagai konfirmasi lintas dataset.

Jika setelah audit hanya satu atau dua lineage independen yang layak, V2 tidak boleh dinarasikan sebagai bukti multi-dataset yang kuat tanpa menyatakan keterbatasannya.

---

## 3. Audit Dataset Sebelum Eksperimen

Setiap kandidat harus melewati audit berikut sebelum dibekukan.

### 3.1 Provenance dan Lisensi

Dicatat:

- nama dataset dan pemilik/workspace;
- URL dan versi dataset yang digunakan;
- tanggal akses;
- lisensi;
- jumlah citra asli yang benar-benar digunakan;
- jumlah kelas dan nama kelas;
- format anotasi;
- apakah versi tersebut mengandung augmentasi yang sudah dibuat platform;
- hubungan dengan paper, repository, atau dataset lain bila dapat ditelusuri.

Dataset tanpa lisensi penggunaan yang jelas atau tanpa akses data yang dapat direproduksi tidak digunakan sebagai dataset utama.

### 3.2 Audit Fork dan Duplikasi Antar-Dataset

Nama workspace berbeda tidak dianggap sebagai bukti independensi. Audit dilakukan melalui:

1. exact file hash;
2. perceptual hash atau embedding similarity untuk near-duplicate;
3. perbandingan resolusi dan metadata file;
4. inspeksi visual sampel;
5. kesamaan kelas, nama file, latar, dan pola anotasi;
6. riwayat fork/version jika tersedia.

Jika dua kandidat terbukti berasal dari lineage data yang sama, hanya satu lineage dipertahankan pada set utama. Dataset derivatif dapat dicatat tetapi tidak dihitung sebagai dataset independen.

### 3.3 Audit Anotasi

Untuk setiap dataset $D_d$, diaudit:

$$
N_{img,d},\qquad N_{box,d},\qquad C_d,
$$

serta untuk setiap kelas:

$$
N_{obj,d,c},\qquad N_{img,d,c}.
$$

Sampel anotasi diperiksa untuk memastikan bounding box dan label konsisten. Kelas ambigu, placeholder, atau label generik seperti `objects` tidak dipakai tanpa definisi yang jelas.

### 3.4 Audit Versi dan Augmentasi

Versi yang sudah mengandung gambar hasil augmentasi tidak boleh diperlakukan sebagai citra sumber independen. Prioritas diberikan kepada versi original/raw atau versi dengan hubungan original–augmented yang dapat dipulihkan.

Jika hanya versi augmented tersedia dan lineage ke citra asli tidak dapat ditentukan, dataset dikeluarkan dari benchmark utama karena risiko leakage tidak dapat dikontrol secara memadai.

---

## 4. Rancangan Split per Dataset

Setiap dataset diproses secara independen.

Jika dataset menyediakan split resmi yang dapat ditelusuri ke citra asli dan tidak menunjukkan leakage, split tersebut dapat dipertahankan. Jika tidak, dibuat split baru sekitar 70/15/15 dari citra asli.

Karena dataset publik umumnya tidak menyediakan `group_id` fisik, unit grouping dibangun dari provenance dan cluster near-duplicate. Misalkan $g(x)$ adalah cluster sumber/duplikasi untuk citra $x$. Maka:

$$
\mathcal G_{train,d}\cap\mathcal G_{val,d}
=\mathcal G_{train,d}\cap\mathcal G_{test,d}
=\mathcal G_{val,d}\cap\mathcal G_{test,d}
=\varnothing.
$$

Augmentasi lokal untuk eksperimen hanya diterapkan setelah split. Validation dan test menggunakan citra sumber yang tidak dibuat dari augmentasi train.

Dataset yang tidak memungkinkan pembentukan split leakage-safe dikeluarkan dari benchmark utama.

---

## 5. Pemilihan Dataset Pengembangan

Satu dataset dipilih sebagai dataset pengembangan:

$$
D_{dev}\in\mathcal D.
$$

Pemilihan $D_{dev}$ dilakukan **sebelum hasil model diperiksa**, berdasarkan kualitas data, bukan performa YOLO. Urutan kriteria:

1. provenance dan versi paling jelas;
2. kualitas anotasi paling baik;
3. tidak terindikasi sebagai fork dari kandidat lain;
4. jumlah citra asli memadai;
5. cakupan kelas cukup beragam untuk menguji fine-grained discrimination.

Dataset lainnya menjadi dataset konfirmasi eksternal:

$$
\mathcal D_{ext}=\mathcal D\setminus\{D_{dev}\}.
$$

---

## 6. Integrasi dengan B0–B3 dan C0–C5

Arsitektur dan frontend tetap sama dengan V1:

- $B_0$: YOLO26n tanpa prapemrosesan tambahan;
- $B_1$: CLAHE + YOLO26n;
- $B_2$: $C_0$ + YOLO26n;
- $B_3$: $C^*$ + YOLO26n.

Perbedaan V2 adalah **di mana $C^*$ dipilih dan di mana ia diuji**.

### Tahap I — Baseline pada Dataset Pengembangan

Baseline pengembangan dilatih pada $D_{dev}$ menggunakan:

$$
s_{dev}=42.
$$

Baseline ini digunakan untuk memeriksa pipeline dan menentukan kelas sulit pada $D_{dev}$.

### Tahap II — Pemilihan C* Hanya pada Dataset Pengembangan

$C_0$ sampai $C_5$ dan analisis sensitivitas hanya digunakan untuk memilih struktur pada validation set $D_{dev}$.

Kriteria pemilihan tetap mengikuti V1: $mAP_{50:95}^{val}$ sebagai kriteria utama, kemudian $AP_{\mathcal H}$ dan median latency end-to-end bila diperlukan.

Setelah dipilih:

$$
C^*=\operatorname{Select}(D_{dev}^{val}),
$$

maka $C^*$ dibekukan.

**Tidak dilakukan tuning ulang $C^*$ pada dataset publik lain.**

### Tahap III — Konfirmasi Multi-Dataset

Pada setiap dataset $D_d\in\mathcal D$, kondisi B0–B3 dilatih dari bobot awal resmi yang sama dengan ruang kelas dataset tersebut.

Seed konfirmasi:

$$
S_{conf}=\{123,2026,31415\}.
$$

Jika $C^*=C_0$, run B2/B3 yang identik tidak diduplikasi.

Untuk dataset eksternal, validation dapat digunakan untuk early stopping sesuai protokol yang dibekukan, tetapi **tidak boleh digunakan untuk mengubah struktur $C^*$**.

### Tahap IV — Final Test

Setelah konfigurasi, seed, checkpoint rule, dan metrik dibekukan, checkpoint setiap seed dievaluasi pada test set masing-masing dataset.

Tidak ada reseleksi metode berdasarkan test set dataset mana pun.

---

## 7. Penanganan Perbedaan Taksonomi

Analisis utama V2 tidak memaksa harmonisasi kelas lintas dataset. Untuk dataset $D_d$, output YOLO disesuaikan menjadi:

$$
C=C_d.
$$

Dengan demikian, `full_black`, `black`, `hitam`, atau istilah lain tidak otomatis digabung hanya berdasarkan kemiripan nama.

Pemetaan lintas dataset hanya dapat dibuat sebagai analisis tambahan bila definisi kelas dari kedua dataset mendukung ekuivalensi. Pemetaan tersebut harus dibuat sebelum hasil eksperimen diperiksa.

Keuntungan pendekatan ini adalah menghindari dua masalah:

1. label yang tampak sama tetapi memiliki definisi berbeda;
2. kelas yang tidak dianotasi pada salah satu dataset menjadi false negative palsu ketika dataset digabung.

---

## 8. Evaluasi Multi-Dataset

Metrik utama tetap:

$$
mAP_{50:95}.
$$

Metrik sekunder tetap $mAP_{50}$, precision, recall, AP per kelas, $AP_{worst}$, dan analisis kelas sulit.

Untuk setiap dataset $d$, seed $s$, dan metrik $M$, perubahan terhadap baseline dihitung:

$$
\Delta_{d,s}=M_{B_3,d,s}-M_{B_0,d,s}.
$$

Rerata per dataset:

$$
\overline{\Delta}_d
=\frac{1}{|S_{conf}|}\sum_{s\in S_{conf}}\Delta_{d,s}.
$$

Ringkasan lintas dataset memberi bobot yang sama pada setiap dataset:

$$
\overline{\Delta}_{multi}
=\frac{1}{K}\sum_{d=1}^{K}\overline{\Delta}_d.
$$

Selain nilai agregat, seluruh $\overline{\Delta}_d$ harus tetap dilaporkan agar peningkatan pada satu dataset tidak menutupi penurunan pada dataset lain.

Sebagai indikator konsistensi, dapat dilaporkan proporsi dataset dengan delta positif:

$$
R_{win}=\frac{1}{K}\sum_{d=1}^{K}\mathbf 1(\overline{\Delta}_d>0).
$$

$R_{win}$ bersifat ringkasan tambahan dan tidak menggantikan mAP atau laporan per dataset.

---

## 9. Hard-Class Analysis pada Multi-Dataset

Karena ruang kelas berbeda, kelompok kelas sulit ditentukan terpisah untuk setiap dataset dari baseline validation yang dibekukan:

$$
\mathcal H_d=\mathrm{Bottom3}\left(AP_{d,c,50:95}^{val}(B_0)\right),
$$

atau seluruh kelas bila $C_d<3$.

Kemudian:

$$
AP_{\mathcal H_d}
=\frac{1}{|\mathcal H_d|}\sum_{c\in\mathcal H_d}AP_{d,c,50:95}.
$$

Tidak dibuat satu daftar kelas sulit global dengan memaksa penyamaan nama kelas antar-dataset.

---

## 10. Bootstrap dan Ketidakpastian

Paired bootstrap dilakukan secara terpisah pada setiap dataset menggunakan unit independen terbaik yang dapat direkonstruksi dari sumber data, misalnya cluster near-duplicate/provenance.

Dataset yang tidak memiliki unit independen yang cukup tetap dilaporkan per seed, tetapi inferensi bootstrap tidak dipaksakan.

Ringkasan lintas dataset terutama didasarkan pada distribusi paired delta antar-seed dan konsistensi arah efek antar-dataset.

---

## 11. Efisiensi Komputasi

Benchmark efisiensi tidak perlu diulang pada seluruh dataset karena biaya frontend pada input 640×640 terutama ditentukan oleh pipeline, bukan taksonomi dataset.

Benchmark utama latency dapat dilakukan pada satu set citra standar dari $D_{dev}$ dengan protokol V1:

- batch 1;
- perangkat dan precision sama;
- $t_{pra}$;
- $t_{model}$;
- $t_{total}$ diukur langsung;
- median $t_{total}$ sebagai ringkasan utama;
- peak allocated GPU memory dan parameter model sebagai informasi tambahan.

Jika terdapat perbedaan distribusi ukuran/aspect ratio sebelum resize yang signifikan, benchmark tambahan per dataset dapat dilaporkan secara deskriptif.

---

## 12. Klaim yang Diizinkan

Jika V2 berhasil, klaim yang dapat dipertahankan adalah:

> konfigurasi prapemrosesan frekuensi-angular yang dipilih pada satu dataset pengembangan menunjukkan efek yang konsisten atau tidak konsisten ketika diterapkan tanpa tuning ulang pada beberapa dataset publik deteksi cacat biji kopi.

V2 **tidak otomatis membuktikan** generalisasi ke seluruh kondisi industri, varietas kopi, perangkat kamera, atau standar mutu karena dataset publik tetap mempunyai domain terbatas.

---

## 13. Perubahan terhadap BAB III Formal bila V2 Dipilih

Jika V2 dipromosikan menjadi metodologi utama, perubahan minimal yang wajib dilakukan adalah:

1. Subbab 3.1: `pengumpulan dan anotasi dataset primer` diganti menjadi `kurasi, audit, dan pembentukan benchmark multi-dataset publik`.
2. Subbab 3.2: seluruh rancangan pengumpulan primer diganti dengan provenance audit, dataset gate, split, dan anti-duplication protocol V2.
3. Subbab 3.3: output kelas dinyatakan dataset-specific, $C=C_d$.
4. Subbab 3.6: pemilihan $C^*$ hanya pada $D_{dev}$ dan konfirmasi dilakukan pada seluruh $\mathcal D$.
5. Subbab 3.8: evaluasi ditambah paired delta lintas dataset dan $\overline{\Delta}_{multi}$.
6. Subbab 3.9–3.10: visualisasi dan error analysis dilakukan per dataset; tidak memaksa penyamaan kelas.
7. Gambar 3.1: alur pengumpulan data primer diganti dengan kurasi → provenance audit → duplicate audit → split per dataset → pilih $D_{dev}$ → pilih $C^*$ → konfirmasi lintas dataset.
8. BAB I dan BAB II: narasi kebutuhan dataset primer harus diubah agar sinkron dengan penggunaan dataset publik.
9. Bibliography/citation gate: dataset publik final yang benar-benar digunakan harus mempunyai citation entry dan metadata yang dikunci.

---

## 14. Keputusan Desain

V2 sengaja menggunakan **multi-dataset independent benchmark**, bukan pooled training set. Alasannya:

- lebih aman terhadap konflik taksonomi;
- lebih aman terhadap missing annotation;
- menghindari klaim bahwa dataset dengan nama kelas mirip mempunyai definisi identik;
- efek metode dapat dinilai melalui paired delta yang comparable;
- memungkinkan pengujian transfer $C^*$ tanpa tuning ulang;
- lebih langsung menguji generalitas frontend dibanding hanya memperbesar jumlah data melalui penggabungan dataset.

Pooled multi-dataset training tetap dapat dibuat sebagai eksperimen tambahan setelah ontology mapping dan duplicate audit selesai, tetapi bukan desain utama V2.
