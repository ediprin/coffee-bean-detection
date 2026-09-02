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

## 3. Rancangan Penggunaan Dataset dalam Eksperimen

Tahap pengembangan dilakukan pada **robusta_SNI_Dataset**. Baseline YOLO26n dan variasi prapemrosesan frekuensi-angular dibandingkan pada data pelatihan dan validasi untuk memperoleh konfigurasi final $C^*$.

Setelah $C^*$ ditetapkan, struktur dan parameter prapemrosesan dibekukan. Selanjutnya, baseline dan model dengan $C^*$ dilatih secara terpisah pada Capstone, Lulus, dan Niacubilla dari bobot awal resmi YOLO26n yang sama.

Dengan demikian, evaluasi lintas dataset tidak menguji checkpoint robusta_SNI_Dataset secara zero-shot. Evaluasi tersebut menguji apakah konfigurasi prapemrosesan yang dipilih pada dataset utama tetap memberikan pengaruh yang konsisten ketika digunakan pada dataset kopi lain.

Perbandingan utama pada dataset konfirmasi adalah perubahan kinerja model dengan $C^*$ terhadap baseline pada dataset yang sama. Nilai mAP absolut antar-dataset tidak dibandingkan secara langsung karena jumlah kelas dan karakteristik datanya berbeda.

---

## 4. Batas Klaim

Jika hasil pada beberapa dataset menunjukkan arah peningkatan yang konsisten, penelitian dapat menyatakan bahwa konfigurasi prapemrosesan frekuensi-angular memiliki konsistensi lintas dataset yang ditinjau.

Penelitian tidak menyatakan generalisasi lintas varietas kopi karena informasi varietas tidak tersedia secara lengkap pada seluruh dataset publik.

Detail teknis pemeriksaan versi, lisensi, duplikasi, dan kebocoran data disimpan pada dokumentasi eksperimen di repository dan tidak menjadi bagian utama narasi proposal.
