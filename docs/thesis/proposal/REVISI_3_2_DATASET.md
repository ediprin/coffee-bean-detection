# Catatan Revisi Subbab 3.2 — Dataset Penelitian

Dokumen ini mencatat keputusan revisi yang telah disepakati untuk Subbab 3.2 pada `BAB_III_METODOLOGI_PENELITIAN.md`. Catatan ini belum dimaksudkan untuk mengganti sumber BAB III secara otomatis; tujuannya adalah menjaga keputusan metodologis agar dapat diterapkan secara terkontrol pada revisi naskah berikutnya.

## Struktur yang Disepakati

Subbab 3.2 akan dirapikan menjadi:

1. **3.2.1 Sumber dan Karakteristik Dataset Primer**
2. **3.2.2 Target Pengumpulan dan Pemeriksaan Kecukupan Data**
3. **3.2.3 Akuisisi Citra dan Anotasi**
4. **3.2.4 Pembagian Data dan Pencegahan Kebocoran**
5. **3.2.5 Augmentasi Data**

Tujuan perubahan ini adalah memisahkan dengan jelas: apa dataset yang digunakan, berapa target datanya, bagaimana citra dan label dibuat, bagaimana split dilakukan, dan bagaimana augmentasi diterapkan.

---

## 3.2.1 Sumber dan Karakteristik Dataset Primer

Penelitian direncanakan menggunakan **dataset primer** yang dikumpulkan secara langsung untuk tugas deteksi objek multikelas pada biji kopi hijau. Daftar kelas awal menargetkan 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008 ditambah satu kelas biji normal, sehingga jumlah kelas target awal dinyatakan sebagai:

$$
C_{target}=21.
$$

Jumlah kelas tersebut merupakan target awal penelitian dan belum dianggap sebagai jumlah kelas final. Jumlah kelas akhir akan ditetapkan setelah dilakukan pemeriksaan kecukupan data pada setiap kelas sebelum pembagian dataset dan pelatihan utama dilakukan.

Berbeda dengan dataset klasifikasi yang menggunakan satu objek pada satu citra, setiap citra pada penelitian ini direncanakan memuat banyak biji kopi. Oleh karena itu, ukuran dataset tidak hanya dinilai dari jumlah citra sumber, tetapi juga dari jumlah objek yang dianotasi, distribusi objek per kelas, dan penyebarannya pada citra serta kelompok sumber yang berbeda.

### Catatan yang belum boleh diasumsikan

Asal fisik sampel kopi, lot/batch, pemasok, kebun, koperasi, atau sumber pengadaan belum boleh diisi dengan asumsi. Informasi tersebut baru ditambahkan setelah sumber aktual ditetapkan dan dapat didokumentasikan.

---

## 3.2.2 Target Pengumpulan dan Pemeriksaan Kecukupan Data

Target pengumpulan ditetapkan sekitar **180–220 citra sumber**, dengan sasaran nominal sekitar 200 citra asli. Citra hasil augmentasi tidak dihitung sebagai citra sumber maupun sebagai data primer.

Setiap citra direncanakan memuat sekitar 30–50 objek yang disusun dalam satu lapisan, dengan orientasi yang bervariasi dan tanpa tumpang tindih berat. Berdasarkan rancangan tersebut, jumlah anotasi objek secara keseluruhan diperkirakan berada pada kisaran:

$$
N_{box}\approx 6.000-10.000.
$$

Untuk keperluan perencanaan penelitian ini, digunakan **target operasional awal** sekurang-kurangnya sekitar 200 objek asli per kelas, dengan sasaran ideal sekitar 300–500 objek per kelas. Selain jumlah objek, setiap kelas diupayakan muncul pada sedikitnya sekitar 30 citra sumber yang berbeda. Angka tersebut merupakan target operasional penelitian dan **bukan batas universal kecukupan dataset deteksi objek**.

Sebagai pembanding, penelitian terdahulu yang digunakan dalam proposal menunjukkan bahwa jumlah anotasi objek dapat jauh lebih besar daripada jumlah citra pada tugas deteksi multiobjek. Oleh karena itu, kecukupan dataset dipertimbangkan berdasarkan jumlah citra sumber dan jumlah objek secara bersama-sama, bukan berdasarkan jumlah citra saja.

Sebelum pembagian data, distribusi setiap kelas perlu diperiksa berdasarkan sekurang-kurangnya tiga informasi:

$$
N_{obj,c},\qquad N_{img,c},\qquad N_{group,c},
$$

dengan:

- $N_{obj,c}$ = jumlah objek asli kelas $c$;
- $N_{img,c}$ = jumlah citra sumber berbeda yang memuat kelas $c$;
- $N_{group,c}$ = jumlah kelompok sumber independen yang mengandung kelas $c$.

Penambahan $N_{group,c}$ diperlukan karena pembagian dataset dilakukan berbasis kelompok sumber. Kelas yang memiliki banyak objek dan citra tetapi hanya berasal dari sedikit sesi independen tetap berisiko lemah untuk evaluasi terpisah train/validation/test.

Jika suatu kelas belum memenuhi target operasional pengumpulan, prioritas pertama adalah **menambah pengumpulan data kelas tersebut**. Apabila kecukupan data tetap tidak dapat dipenuhi, kelas tersebut tidak dipaksakan menjadi kelas evaluasi utama. Penggabungan kelas tidak dilakukan hanya karena kekurangan data kecuali terdapat dasar taksonomi/SNI yang membenarkannya.

Keputusan jumlah kelas final harus dilakukan **sebelum pembagian dataset dan sebelum pelatihan utama**, sehingga penetapan kelas tidak dipengaruhi oleh hasil performa model.

Jumlah kelas akhir dinyatakan sebagai:

$$
C\le C_{target},
$$

dengan target utama tetap $C=21$ apabila seluruh kelas memiliki data yang dinilai memadai.

Urutan keputusan harus mengikuti:

```text
pengumpulan data
→ audit kecukupan per kelas
→ tetapkan dan bekukan C final
→ pembagian train/validation/test
→ pelatihan utama
```

Tidak diperbolehkan melatih 21 kelas terlebih dahulu kemudian menghapus kelas yang memiliki AP rendah untuk memperbaiki hasil akhir.

---

## 3.2.3 Akuisisi Citra dan Anotasi

Pengambilan citra direncanakan dilakukan secara tegak lurus dari atas menggunakan latar belakang polos dan tidak reflektif, dengan posisi kamera, jarak pengambilan, dan pencahayaan yang dikendalikan. Biji kopi disusun dalam satu lapisan agar karakteristik permukaannya tetap terlihat, sedangkan orientasi objek tetap divariasikan untuk memperoleh variasi sisi dan arah biji.

Setiap sesi pengambilan citra dan setiap citra sumber akan diberikan identitas yang dapat ditelusuri. Citra yang berasal dari sesi, susunan objek, atau kelompok spesimen fisik yang berkaitan akan diberi **`group_id` yang sama**. Informasi tersebut digunakan pada tahap pembagian data untuk mencegah kebocoran antar train, validation, dan test.

Definisi operasional `group_id` harus dibuat sebelum pengumpulan utama. Secara prinsip, satu group merepresentasikan unit sumber yang tidak boleh dipecah antar split, misalnya satu sesi pengambilan atau satu kelompok spesimen fisik yang saling berkaitan.

Untuk kategori yang definisinya bergantung pada ukuran fisik, khususnya benda asing yang dibedakan berdasarkan ukuran, proses akuisisi dilengkapi dengan referensi skala sehingga ukuran fisik objek dapat ditelusuri secara konsisten.

Setiap objek diberikan kotak pembatas (*bounding box*) dan label kelas. Definisi operasional setiap kelas ditetapkan sebelum proses anotasi dengan mengacu pada SNI dan referensi visual yang digunakan. Sampel yang secara visual meragukan tidak langsung diberikan label final, tetapi ditandai untuk ditinjau kembali. Validasi label direncanakan melibatkan praktisi atau validator yang memahami penilaian fisik mutu kopi, terutama untuk kelas-kelas yang memiliki kemiripan visual tinggi.

---

## 3.2.4 Pembagian Data dan Pencegahan Kebocoran

Pembagian dataset dilakukan terhadap **citra sumber asli sebelum augmentasi**. Proporsi awal yang direncanakan adalah sekitar 70% untuk pelatihan, 15% untuk validasi, dan 15% untuk pengujian. Dengan target nominal sekitar 200 citra sumber, proporsi tersebut secara kasar setara dengan sekitar 140 citra pelatihan, 30 citra validasi, dan 30 citra pengujian.

Pembagian tidak dilakukan semata-mata melalui pengacakan citra individual, tetapi berdasarkan kelompok sumber atau `group_id`. Seluruh citra yang berasal dari sesi, susunan objek, atau spesimen fisik yang saling berkaitan harus ditempatkan pada bagian data yang sama. Jika suatu spesimen fisik difoto lebih dari satu kali, seluruh citra yang memuat spesimen tersebut harus berada pada split yang sama.

Kelompok sumber pada ketiga bagian dataset harus saling terpisah:

$$
\mathcal{G}_{train}\cap\mathcal{G}_{val}
=\mathcal{G}_{train}\cap\mathcal{G}_{test}
=\mathcal{G}_{val}\cap\mathcal{G}_{test}
=\varnothing.
$$

Selain menjaga pemisahan kelompok sumber, pembagian dataset juga mempertimbangkan distribusi kelas. Proses pembagian mencari komposisi kelompok yang tetap saling terpisah tetapi sedapat mungkin mempertahankan keterwakilan seluruh kelas.

Target sekitar lima citra sumber per kelas pada masing-masing validation dan test dapat digunakan sebagai **sasaran keterwakilan**, bukan sebagai batas universal, sepanjang dapat dipenuhi tanpa melanggar pemisahan kelompok sumber. Proporsi 70:15:15 juga diperlakukan sebagai sasaran keseluruhan dan dapat bergeser sedikit jika diperlukan untuk memperoleh keterwakilan kelas yang lebih baik tanpa melanggar prinsip group separation.

Pemeriksaan tambahan terhadap citra identik dilakukan menggunakan nilai *hash*. Pemeriksaan hash merupakan lapisan tambahan untuk mendeteksi duplikasi file; mekanisme utama pencegahan kebocoran tetap identitas sumber dan `group_id`.

Data validasi digunakan untuk penghentian dini, pembandingan konfigurasi prapemrosesan, analisis sensitivitas, dan pemilihan konfigurasi $C^*$. Data uji disisihkan sejak awal dan tidak digunakan untuk memilih ukuran patch, parameter $\gamma$, variasi prapemrosesan, maupun keputusan metodologis lainnya. Evaluasi data uji dilakukan setelah konfigurasi akhir dan prosedur evaluasi dibekukan.

---

## 3.2.5 Augmentasi Data

Augmentasi hanya diterapkan pada bagian pelatihan setelah pembagian citra sumber selesai. Data validasi dan pengujian tetap menggunakan citra asli tanpa augmentasi sintetis.

Konfigurasi augmentasi dibuat sama untuk seluruh kondisi YOLO26n yang dibandingkan sehingga perbedaan hasil antar kondisi eksperimen tidak berasal dari perbedaan strategi augmentasi.

### Detail yang dipindahkan keluar dari Subbab 3.2

Penjelasan berikut **tidak dihapus**, tetapi sebaiknya dipindahkan ke bagian prapemrosesan atau rancangan eksperimen (Subbab 3.4/3.6):

- CLAHE dan prapemrosesan frekuensi-angular bukan augmentasi;
- urutan augmentasi → prapemrosesan → YOLO26n;
- posisi CLAHE dan frekuensi-angular harus setara dalam pipeline;
- domain/normalisasi tensor masukan;
- konversi internal luminansi integer jika diperlukan oleh CLAHE;
- pengembalian ke representasi RGB *floating point* yang setara;
- CLAHE dan prapemrosesan frekuensi-angular tidak mengubah geometri bounding box.

Alasan pemindahan: informasi tersebut merupakan aturan **pipeline perlakuan eksperimen**, bukan aturan pembentukan dataset.

---

## Prinsip yang Harus Dipertahankan pada Revisi BAB III

1. Target awal tetap **21 kelas** (20 kategori cacat/benda asing SNI + 1 normal), tetapi $C$ final ditentukan berdasarkan kecukupan data sebelum training utama.
2. Target 200 objek/kelas, 300–500 objek ideal, dan sekitar 30 citra sumber diposisikan sebagai **target operasional penelitian**, bukan standar universal.
3. Audit kecukupan menggunakan tiga dimensi: objek, citra sumber, dan group independen.
4. Kekurangan data asli tidak dianggap terselesaikan hanya dengan augmentasi.
5. Prioritas pada kelas kurang data adalah menambah pengumpulan; kelas tidak digabung tanpa dasar taksonomi/SNI.
6. `group_id` didefinisikan sebelum split dan digunakan sebagai unit independen pembagian data.
7. Tidak boleh ada group yang silang antara train, validation, dan test.
8. Validation digunakan untuk pengembangan/pemilihan $C^*$; test tidak digunakan untuk keputusan metodologis.
9. Augmentasi hanya diterapkan pada train dan dibuat sama pada seluruh kondisi utama.
10. Detail teknis posisi CLAHE/frequency-angular dipindahkan dari subbab dataset ke bagian pipeline eksperimen.
