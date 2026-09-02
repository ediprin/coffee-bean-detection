# V2 — Multi-Dataset Publik

Status: **WORKING VARIANT — NOT FORMAL**

Dokumen ini merupakan alternatif rancangan dataset untuk proposal tesis. Selama belum dipromosikan secara eksplisit, source formal proposal tetap menggunakan V1.

## 1. Prinsip Utama

Penelitian menggunakan satu dataset utama untuk pengembangan metode dan tiga dataset publik tambahan untuk konfirmasi lintas dataset. Dataset tidak digabung karena memiliki sumber, jumlah kelas, dan karakteristik citra yang berbeda.

Konfigurasi prapemrosesan frekuensi-angular dipilih hanya pada dataset utama. Setelah konfigurasi final ditetapkan, konfigurasi yang sama digunakan tanpa penyesuaian ulang pada dataset konfirmasi.

---

## 2. Dataset Penelitian

### 2.1 Sumber dan Peran Dataset

Dataset utama penelitian adalah **robusta_SNI_Dataset**, sedangkan **Coffee Bean Defect (Capstone)**, **Green Coffee Bean Defects (Lulus)**, dan **Coffee Bean Defects (Niacubilla)** digunakan sebagai dataset konfirmasi. Setiap dataset digunakan secara terpisah sesuai dengan kelas yang tersedia.

| Dataset | Sumber | Peran |
|---|---|---|
| robusta_SNI_Dataset | Roboflow Universe | Dataset utama untuk pengembangan dan pemilihan konfigurasi prapemrosesan |
| Coffee Bean Defect (Capstone) | Roboflow Universe | Dataset konfirmasi I |
| Green Coffee Bean Defects (Lulus) | Roboflow Universe | Dataset konfirmasi II |
| Coffee Bean Defects (Niacubilla) | Roboflow Universe | Dataset konfirmasi III |

### 2.2 Dataset Utama robusta_SNI_Dataset

**robusta_SNI_Dataset** merupakan dataset biji kopi yang tersedia dalam format instance segmentation dan mencakup 21 kelas yang dapat dipetakan ke taksonomi SNI. Dalam penelitian ini anotasi objek digunakan sebagai bounding box untuk tugas deteksi.

Dataset yang digunakan telah disusun ulang dengan pembagian terkelompok berdasarkan sumber citra. Data pengembangan terdiri atas **1.665 citra pelatihan dengan 2.986 objek** dan **294 citra validasi dengan 526 objek**. Seluruh 21 kelas terdapat pada data pelatihan dan validasi. Test set dipisahkan dari tahap pengembangan dan digunakan setelah konfigurasi penelitian dibekukan.

Dataset ini menjadi satu-satunya dataset untuk memilih konfigurasi prapemrosesan frekuensi-angular.

### 2.3 Dataset Publik untuk Konfirmasi

Tiga dataset publik digunakan untuk melihat apakah pengaruh metode tetap konsisten pada sumber data yang berbeda. Dataset tersebut tidak digunakan untuk memilih ulang konfigurasi prapemrosesan.

| Dataset | Task | Jumlah kelas | Penggunaan |
|---|---|---:|---|
| Coffee Bean Defect (Capstone) | Object detection | 14 | Konfirmasi lintas dataset |
| Green Coffee Bean Defects (Lulus) | Object detection | 6 | Konfirmasi lintas dataset |
| Coffee Bean Defects (Niacubilla) | Object detection | 9 | Konfirmasi lintas dataset |

Jumlah citra dan pembagian train, validation, dan test mengikuti versi dataset yang dibekukan setelah pemeriksaan data. Setiap dataset mempertahankan taksonomi kelasnya sendiri dan tidak disatukan dengan kelas pada robusta_SNI_Dataset.

Informasi varietas kopi pada sebagian dataset publik tidak dinyatakan secara eksplisit oleh sumber dataset. Oleh karena itu, dataset publik digunakan untuk mengevaluasi konsistensi metode pada sumber data yang berbeda, bukan untuk membandingkan performa antarvarietas kopi.

### 2.4 Pembagian Data

Setiap dataset dibagi menjadi data pelatihan, validasi, dan pengujian. Data pelatihan digunakan untuk melatih model, data validasi digunakan untuk pemantauan pelatihan dan pemilihan checkpoint, sedangkan data pengujian digunakan untuk evaluasi akhir.

Pada robusta_SNI_Dataset, pembagian dilakukan secara terkelompok untuk mencegah citra yang berasal dari sumber yang sama berada pada split berbeda. Dataset publik menggunakan split yang telah diperiksa; jika split bawaan tidak memenuhi kebutuhan penelitian, pembagian disusun ulang sebelum eksperimen.

Konfigurasi prapemrosesan hanya dipilih menggunakan data pelatihan dan validasi robusta_SNI_Dataset. Validation pada dataset konfirmasi tidak digunakan untuk mengubah struktur atau parameter prapemrosesan yang telah dibekukan.

### 2.5 Augmentasi Data

Augmentasi diterapkan hanya pada data pelatihan. Data validasi dan pengujian tidak menerima augmentasi pelatihan. Pada setiap dataset, baseline dan model dengan prapemrosesan menggunakan perlakuan augmentasi yang sama agar perbandingan tetap setara.

---

## 3. Rancangan Eksperimen Multi-Dataset

### 3.1 Tahap Pengembangan

Pengembangan metode dilakukan hanya pada **robusta_SNI_Dataset**. YOLO26n tanpa prapemrosesan digunakan sebagai baseline, kemudian konfigurasi referensi dan variasi prapemrosesan frekuensi-angular dibandingkan pada data pelatihan dan validasi hingga diperoleh konfigurasi final $C^*$.

Pada tahap ini digunakan kondisi $B_0$, $B_1$, $B_2$, dan $B_3$ sesuai rancangan eksperimen utama.

### 3.2 Tahap Konfirmasi Lintas Dataset

Setelah $C^*$ ditetapkan, konfigurasi tersebut dibekukan. Pada dataset konfirmasi hanya dibandingkan baseline YOLO26n ($B_0$) dan YOLO26n dengan konfigurasi final $C^*$ ($B_3$).

| Dataset | Kondisi yang dibandingkan |
|---|---|
| robusta_SNI_Dataset | $B_0$, $B_1$, $B_2$, $B_3$ |
| Coffee Bean Defect (Capstone) | $B_0$ dan $B_3$ |
| Green Coffee Bean Defects (Lulus) | $B_0$ dan $B_3$ |
| Coffee Bean Defects (Niacubilla) | $B_0$ dan $B_3$ |

Pada setiap dataset konfirmasi, $B_0$ dan $B_3$ dilatih ulang secara terpisah dari bobot awal resmi `yolo26n.pt` menggunakan seed konfirmasi **123, 2026, dan 31415**. Konfigurasi $C^*$ tidak dipilih ulang berdasarkan hasil dataset konfirmasi.

### 3.3 Evaluasi Akhir

Test set digunakan setelah konfigurasi metode, seed, aturan checkpoint, dan prosedur evaluasi ditetapkan. Hasil pada setiap dataset dilaporkan secara terpisah dan tidak digunakan untuk memilih ulang konfigurasi.

---

## 4. Evaluasi Hasil

Metrik utama adalah **mAP50–95**, sedangkan **mAP50, precision, recall**, dan AP per kelas digunakan sebagai metrik pendukung.

Pada **robusta_SNI_Dataset**, evaluasi digunakan untuk menilai baseline, pembanding CLAHE, konfigurasi referensi, dan konfigurasi frekuensi-angular terpilih. Pada **Capstone, Lulus, dan Niacubilla**, evaluasi difokuskan pada selisih kinerja $B_3$ terhadap $B_0$ pada dataset yang sama.

Hasil setiap seed dilaporkan beserta rata-rata dan simpangan bakunya. Konsistensi lintas dataset dinilai dari arah dan besarnya perubahan kinerja pada ketiga dataset konfirmasi. Nilai mAP absolut antar-dataset tidak dibandingkan secara langsung karena jumlah kelas dan karakteristik data berbeda.

---

## 5. Batas Klaim

Jika hasil pada beberapa dataset menunjukkan arah peningkatan yang konsisten, penelitian dapat menyatakan bahwa konfigurasi prapemrosesan frekuensi-angular memiliki konsistensi lintas dataset yang ditinjau.

Penelitian tidak menyatakan generalisasi lintas varietas kopi karena informasi varietas tidak tersedia secara lengkap pada seluruh dataset publik.

Detail teknis pemeriksaan versi, lisensi, duplikasi, dan kebocoran data disimpan pada dokumentasi eksperimen di repository dan tidak menjadi bagian utama narasi proposal.