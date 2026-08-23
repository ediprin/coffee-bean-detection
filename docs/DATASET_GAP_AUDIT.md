# Audit Dataset dan Kelayakan Gap Penelitian

Tanggal audit: 20 Juli 2026

## Tujuan

Dokumen ini memisahkan tiga hal yang sering tercampur:

1. masalah yang dilaporkan literatur;
2. kemampuan data yang benar-benar tersedia;
3. metode yang mungkin diuji setelah gap dipilih.

Tidak ada teknik atau keluarga model yang diprioritaskan dalam audit ini.
Counting, segmentation, synthetic data, YOLO, dan DETR diperlakukan sebagai
kandidat yang bergantung pada masalah serta data, bukan sebagai gap bawaan.

## Sumber audit

- `outputs/literature_master_2026-07-17/coffee_grain_detection_literature_master_2026-07-17.xlsx`
- `docs/coffee_bean_literature_atlas_2026-07-11.xlsx`
- dataset lokal pada repo klasifikasi `bilinear-LMMD/data`
- audit Coffee Defect v11 yang dijalankan di Google Colab
- hasil baseline dan visual audit YOLO26n yang sudah dilaporkan

Atlas mencatat 54 entri dataset/studi kopi. Sebagian besar data deteksi
multikelas bersifat privat, *on request*, atau tidak menjelaskan akses secara
memadai. Karena itu, keberadaan sebuah paper tidak dianggap sama dengan
ketersediaan dataset untuk reproduksi.

## Data yang telah dikonfirmasi

| Dataset/artefak | Status akses saat audit | Tahap kopi | Unit citra dan anotasi | Kelas | Skala yang dikonfirmasi | Kemampuan yang dapat diuji | Batas klaim |
|---|---|---|---|---:|---:|---|---|
| Coffee Green Bean with 17 Defects Original | Tersedia lokal di `bilinear-LMMD/data/coffee` | Green Arabica | Satu biji, label citra | 17 | 979 citra pada salinan lokal | Fine-grained image classification, imbalance, confusion antarkelas | Tidak dapat langsung menguji object detection, kepadatan, atau oklusi |
| Coffee-17 grouped folds | Tersedia lokal di `bilinear-LMMD/data/coffee_5fold` | Green Arabica | Salinan fold dari dataset yang sama | 17 | 4.896 referensi citra lintas lima fold; bukan 4.896 objek independen | Grouped cross-validation untuk klasifikasi | Bukan dataset tambahan atau domain eksternal |
| Coffee-17 clean | Pernah dibuat di Kaggle; belum tersimpan lokal pada audit ini | Green Arabica | Satu biji, label citra | 17 | 965 citra bersih setelah 12 duplikat label-sama dan 2 konflik dikarantina | Klasifikasi dengan kontrol duplikasi | Perlu dibuat ulang atau dipersistenkan sebelum eksperimen baru |
| USK-COFFEE | Publik; pernah dipakai di Kaggle, tidak tersimpan lokal saat audit | Green Arabica | Satu biji, label citra | 4 | 8.000 citra | Benchmark klasifikasi empat kategori luas | Semua cacat digabung menjadi kelas `defect`; bukan taksonomi cacat rinci |
| Coffee Defect v11/Roboflow yang dipakai baseline | Ada pada sesi Colab/Drive, tidak disimpan dalam Git atau lokal | Post-roast | Banyak biji, bounding box | 6 | 10.959 citra dan 321.209 box pada split hasil audit | Deteksi multiobjek, density, count error, efisiensi | Bukan green-bean SNI; belum menyediakan mask dan provenance lintas lot yang memadai |
| Coffee roast-level public dataset | Publik; pernah dipakai di Kaggle, tidak tersimpan lokal saat audit | Green/light/medium/dark roast | Label citra | 4 roast levels; arsip gabungan yang ditemukan memuat 8 folder kelas | 9.600 file pada arsip yang diperiksa | Sanity-check klasifikasi roast | Ditemukan duplikat lintas split; terlalu mudah dan tidak cocok sebagai bukti utama cacat |
| Robusta Coffee Bean Defects (Roboflow) | Tercatat publik dalam atlas; belum diaudit atau diunduh lokal | Robusta | Instance segmentation | 4 | 332 citra dilaporkan | Pilot mask/instance segmentation | Kecil, user-generated, provenance dan kualitas anotasi belum diverifikasi |
| robusta_SNI_Dataset (Roboflow, Faruq) | Dapat ditemukan di Roboflow Universe; versi ekspor dan lisensi belum diverifikasi | Green Robusta | Multiobjek, instance segmentation | 20 label yang tampak mengikuti SNI, termasuk `biji_normal` | Sekitar 2.150 citra menurut indeks Roboflow | Kandidat utama audit deteksi/segmentasi multikelas SNI | Jumlah citra mendekati kelipatan korpus 107 citra pada studi SNI; parent image dan kemungkinan augmentasi harus diaudit sebelum split |
| YOLO SKRIPSI 2 (Roboflow, adrianworkspace) | Dapat ditemukan di Roboflow Universe; versi ekspor dan lisensi belum diverifikasi | Green coffee/SNI; perlu verifikasi visual | Multiobjek, object detection | 20 label, termasuk tiga ukuran batu, kelas cacat biji, `Biji tanpa cacat`, kulit kopi, dan kulit tanduk | Sekitar 7.000 citra menurut indeks Roboflow | Kandidat utama bbox detection multikelas SNI | Sangat mungkin memuat augmentasi besar; hubungan dengan korpus 107 citra/13.863 box dan dengan `robusta_SNI_Dataset` harus diuji melalui parent ID serta hash |

## Dataset relevan yang belum tersedia untuk reproduksi lokal

| Target | Bukti di literatur | Status data | Konsekuensi |
|---|---|---|---|
| Deteksi lengkap 20 kelas SNI | Bahy & Rifai: 107 citra dan 13.863 box | Data mentah tidak publik | Belum dapat mereproduksi atau membuat pembanding adil |
| Deteksi 20 kelas dan grading | Jundullah et al.: 2.000 citra | Akses data tidak dinyatakan | Belum dapat memvalidasi label, split, atau dukungan per kelas |
| RT-DETR 20 kelas SNI melalui video | Setyawan: tesis magister | Data/checkpoint belum tersedia | Hanya dapat dijadikan bukti kelayakan, bukan baseline reproduktif |
| Oriented detection tiga kelas SNI | Studi LSKNet-Oriented R-CNN | Dataset institusional; label dan lisensi perlu diverifikasi | Kandidat pembanding hanya setelah akses serta anotasi dipastikan |
| Deteksi 13 cacat dengan SAM-assisted annotation | KN-YOLOv8 | Dataset/code release belum ditemukan | Tidak dapat diasumsikan tersedia dari paper terbuka |

## Audit Coffee Defect v11: fakta dan interpretasi yang dibatasi

Audit split yang dilaporkan:

| Split | Gambar | Box |
|---|---:|---:|
| Train | 9.588 | 280.962 |
| Validation | 914 | 26.912 |
| Test | 457 | 13.335 |
| Total | 10.959 | 321.209 |

Hasil audit menyatakan tidak ada *exact/parent leakage*, tetapi menghasilkan
231.851 peringatan pasangan *near-duplicate*. Peringatan ini belum membuktikan
leakage karena citra berisi banyak biji yang homogen dapat membentuk rantai
kemiripan palsu. Sampel pasangan harus diperiksa sebelum mengambil kesimpulan.

Baseline YOLO26n menghasilkan mAP50-95 97,22% dan worst-class AP50-95 95,48%.
Pada konfigurasi visual audit yang dipakai, *exact count match* adalah 30,20%
dan bias jumlah rata-rata +1,56 objek per gambar. Temuan jumlah tersebut adalah
diagnosis konfigurasi inference pada dataset ini, bukan bukti bahwa counting
merupakan gap utama seluruh literatur. Confidence dan aturan matching harus
dikalibrasi pada validation set sebelum kesimpulan test final.

## Matriks gap terhadap kemampuan data saat ini

Skor sengaja tidak diberikan. Status hanya menunjukkan apakah hipotesis dapat
diuji secara sah dengan data yang telah dikonfirmasi.

| Kandidat gap | Dukungan literatur | Dukungan data saat ini | Penghalang utama | Status netral |
|---|---|---|---|---|
| Fine-grained klasifikasi cacat | Banyak studi melaporkan confusion kelas warna, tekstur, dan tingkat kerusakan | Coffee-17 menyediakan 17 kelas satu-biji | Bukan tugas deteksi multiobjek dan bukan seluruh 20 kelas SNI | Siap untuk klasifikasi; belum siap untuk klaim deteksi |
| Fine-grained deteksi lengkap SNI | Studi lokal menunjukkan kebutuhan dan kesulitan 20 kelas | Ada kandidat Roboflow 20 label dengan instance mask, tetapi belum diunduh dan diaudit | Versi, lisensi, parent split, distribusi kelas, dan kualitas mask | Kandidat tersedia; belum siap training |
| Deteksi padat/oklusi | Didukung studi dense beans, SAHI, NMS, conveyor, dan tanaman lain | Coffee Defect v11 menyediakan gambar multiobjek dan bbox | Post-roast enam kelas; tingkat oklusi belum diberi label eksplisit | Siap sebagai pilot, bukan klaim SNI |
| Generalisasi lintas lot/kamera | Berulang sebagai limitasi/future work | Beberapa dataset publik ada, tetapi tahap dan labelnya tidak kompatibel | Tidak ada dua domain nyata dengan taksonomi sama dan provenance jelas | Belum siap untuk evaluasi lintas domain yang adil |
| Long-tail/rare defect | Didukung pada taksonomi luas | Coffee-17 dapat menguji imbalance klasifikasi; distribusi Coffee Defect v11 perlu diaudit | Jumlah sampel bukan pengganti keragaman visual | Sebagian siap |
| Instance segmentation | Relevan ketika biji menyatu atau batas objek dibutuhkan | Satu dataset Roboflow kecil tercatat, belum diaudit | Mask nyata dan ukuran sampel | Belum siap |
| Counting/grading | Relevan pada sistem yang menghitung cacat dan menerapkan aturan mutu | Jumlah ground-truth dapat diturunkan dari bbox Coffee Defect v11 | Belum diketahui apakah error berasal dari threshold, duplikasi, miss, atau anotasi | Siap sebagai metrik diagnostik; belum terbukti sebagai gap utama |
| Efisiensi edge/conveyor | Banyak paper membahas trade-off akurasi-kecepatan | Model dapat dibenchmark pada dataset mana pun | Perangkat target dan pipeline end-to-end belum ditetapkan | Siap sebagai evaluasi sekunder |
| Robustness blur/illumination | Berulang pada pengujian conveyor dan deployment | Belum ada video/conveyor lokal terkalibrasi | Korupsi sintetis tidak menggantikan domain nyata | Belum siap untuk klaim operasional |
| Open-set/unknown defect | Sedikit diteliti langsung pada kopi | Kelas dapat ditahan secara artifisial | Held-out class bukan pengganti unknown dunia nyata | Eksploratif |
| Multimodal RGB-NIR/HSI | Relevan untuk cacat internal yang tidak terlihat pada RGB | Tidak ada sensor atau data lokal | Akuisisi, kalibrasi, dan biaya sensor | Tidak siap |

## Decision gates sebelum memilih metode

### G1 - Target ilmiah

Pilih salah satu unit masalah, tanpa memilih model dahulu:

1. klasifikasi satu biji;
2. deteksi multiobjek;
3. instance segmentation;
4. deteksi menuju grading.

Mencampur keempatnya akan mengubah kebutuhan data dan metrik secara mendasar.

### G2 - Tahap dan taksonomi kopi

Pilih apakah target utama adalah:

- green bean dengan SNI/SCA;
- post-roast dengan enam label operasional;
- atau generalisasi lintas tahap.

Dataset Coffee Defect v11 tidak boleh disebut benchmark SNI hanya karena sama-sama
memakai objek biji kopi.

### G3 - Ketersediaan anotasi

- Bbox memungkinkan deteksi dan pencacahan instance.
- Mask diperlukan untuk batas instance dan pengukuran bentuk/area.
- Label citra satu-biji tidak otomatis menjadi bbox scene padat tanpa proses
  komposisi atau anotasi tambahan.

### G4 - Domain eksternal

Klaim robustness lintas domain hanya dibuat jika ada domain test nyata yang:

- tidak dipakai saat training atau pemilihan hyperparameter;
- menggunakan label yang kompatibel;
- memiliki provenance lot/kamera/sesi;
- dan tidak berisi turunan gambar training.

## Keputusan yang dapat dibuat dari audit ini

1. Coffee-17 tetap valid untuk penelitian klasifikasi fine-grained, bukan
   baseline deteksi.
2. Coffee Defect v11 valid sebagai pilot deteksi padat post-roast enam kelas,
   bukan bukti deteksi lengkap SNI.
3. Counting dipertahankan sebagai metrik diagnostik pada data bbox, bukan
   ditetapkan sebagai kontribusi tesis.
4. Synthetic composition, P2, segmentation, count head, attention, YOLO, dan
   DETR belum dipilih. Kebutuhannya ditentukan setelah error audit dan decision
   gates di atas.
5. Bila target final adalah deteksi multiobjek berstandar SNI, hambatan pertama
   adalah akses atau pembuatan dataset bbox/mask berlabel SNI, bukan pemilihan
   backbone.

## Kandidat baru yang mengubah status audit

Tautan `robusta_SNI_Dataset` dan `YOLO SKRIPSI 2` mengubah status deteksi SNI
dari "tidak ditemukan kandidat publik" menjadi "ada dua kandidat publik yang
belum tervalidasi". Keduanya belum boleh langsung disebut benchmark siap
pakai. Sebelum training, audit minimum harus menjawab:

1. berapa gambar asli sebelum preprocessing dan augmentasi;
2. apakah train, validation, dan test berisi turunan parent image yang sama;
3. apakah 20 label merupakan 19 jenis cacat/material ditambah normal, atau tepat
   20 kategori cacat menurut tabel SNI;
4. distribusi gambar dan instance per kelas;
5. apakah polygon mask mengikuti batas objek atau dibuat otomatis dari box;
6. berapa instance per gambar, tingkat kontak, dan tingkat oklusi;
7. konsistensi anotasi kelas yang mirip;
8. lisensi versi dataset yang akan diunduh;
9. overlap exact/near-duplicate dan parent image di antara kedua proyek.

`YOLO SKRIPSI 2` bukan otomatis domain eksternal terhadap
`robusta_SNI_Dataset`. Nama kelasnya sangat dekat dan keduanya mungkin berasal
dari sumber fisik atau korpus dasar yang sama. Keduanya baru boleh disebut dua
dataset independen apabila audit menunjukkan tidak ada parent image, objek,
atau turunan augmentasi yang sama. Jika ternyata satu sumber, perannya adalah
dua bentuk anotasi/versi dataset, bukan external validation.

## Langkah berikutnya

Keputusan berikutnya adalah memilih cabang target:

- **Cabang A:** lanjutkan Coffee Defect v11 sebagai pilot metodologi deteksi
  padat post-roast; atau
- **Cabang B:** audit bersama `robusta_SNI_Dataset` dan `YOLO SKRIPSI 2`, lalu
  tentukan apakah keduanya independen, turunan satu korpus, atau dua format
  anotasi. Jika gagal pada lisensi, parent split, atau kualitas anotasi, susun
  protokol akuisisi/anotasi dataset multiobjek green bean berlabel SNI.

Kedua cabang dapat berjalan berurutan, tetapi hasil Cabang A tidak boleh
digeneralisasi menjadi klaim SNI tanpa data Cabang B.
