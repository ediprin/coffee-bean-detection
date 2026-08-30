# Audit Konsistensi BAB III

Dokumen ini mencatat audit internal terhadap `BAB_III_METODOLOGI_PENELITIAN.md` setelah konsolidasi revisi Subbab 3.2–3.12. Audit ini berfokus pada kontradiksi antar-subbab, notasi yang ambigu, keputusan yang masih tentatif, dan konsistensi gambar dengan naskah.

## A. Isu prioritas tinggi

### 1. Tahap III dan penggunaan data uji masih ambigu

Subbab 3.6.3 menggunakan istilah **"Pengujian Konfirmasi dengan Beberapa Seed"** dan menyatakan hasil per seed dilaporkan, sedangkan Subbab 3.6.5 menyatakan data uji baru dibuka setelah Tahap III. Ini dapat dibaca seolah-olah Tahap III sudah melakukan pengujian akhir, padahal test set belum boleh digunakan.

Revisi yang disarankan:

- ubah 3.6.3 menjadi **"Pelatihan Ulang dengan Seed Konfirmasi"** atau istilah setara;
- jelaskan bahwa run pada `S_conf` tetap menggunakan train/validation untuk training dan checkpoint selection;
- test set tetap tertutup selama Tahap III;
- evaluasi pada test untuk checkpoint seluruh seed konfirmasi hanya dilakukan pada 3.6.5;
- jika metrik validation per seed dilaporkan pada Tahap III, sebut sebagai analisis kestabilan run, bukan evaluasi akhir independen.

### 2. Kontrak rentang numerik keluaran prapemrosesan belum final

Subbab 3.4.6 masih membuka pilihan antara clipping, renormalisasi, atau perubahan posisi frontend terhadap normalisasi masukan. Ini merupakan bagian dari definisi metode, bukan sekadar detail logging.

Sebelum eksperimen utama harus ditetapkan satu kontrak yang jelas untuk `B0`, `B1`, `B2`, dan `B3`, sehingga perbedaan tidak berasal dari skala intensitas yang tidak terkontrol. Keputusan ini harus diverifikasi terhadap implementasi aktual dan referensi AFAB-2 sebelum dikunci; jangan dipilih berdasarkan hasil validasi/test.

### 3. Kontrak Ultralytics/checkpoint belum final

Subbab 3.7 dan 3.12 masih menyisakan tiga hal yang harus diverifikasi:

- versi Ultralytics final;
- optimizer aktual yang dihasilkan `optimizer=Auto`;
- kriteria aktual `best.pt` dan early stopping.

Redaksi `Ultralytics 8.4.96 digunakan ... apabila hasil verifikasi ...` masih terlalu tentatif untuk naskah final. Setelah verifikasi implementasi, versi dan aturan checkpoint harus ditulis secara definitif dan konsisten pada 3.7 serta 3.12.

## B. Isu internal yang dapat diperbaiki langsung

### 4. Benturan notasi `B1–B3`

Pada 3.5.3, pita radial diberi nama `B1`, `B2`, dan `B3`. Notasi yang sama sudah digunakan untuk kondisi eksperimen `B0–B3`. Ini menimbulkan ambiguitas.

Revisi yang disarankan:

- ubah pita radial menjadi `R_1`, `R_2`, `R_3` atau `\mathcal{R}_1`, `\mathcal{R}_2`, `\mathcal{R}_3`;
- jangan gunakan `b` sebagai indeks pita radial karena `b(u,v)` sudah digunakan sebagai fungsi pemetaan bin angular pada 3.4.3;
- gunakan indeks radial lain, misalnya `\ell`, sehingga notasi menjadi `D_i^c(\ell,k)`, `p_i^c(\ell,k)`, `H_i^c(\ell)`, `\tau_i^c(\ell)`, dan `q_i^c(\ell,k)`.

### 5. Klaim preservasi rasio kanal pada C5 bergantung pada pemetaan output

Pada 3.5.5, gate luminansi yang sama memang mempertahankan rasio kanal pada operasi residual sebelum pemetaan keluaran. Namun jika tahap akhir memakai clipping atau renormalisasi tertentu, sifat tersebut dapat berubah.

Revisi yang disarankan: tulis bahwa rasio kanal dipertahankan **pada operasi residual sebelum pemetaan akhir ke domain masukan model**. Setelah kontrak 3.4.6 dikunci, klaim ini harus dicek kembali.

### 6. Estimasi jumlah anotasi perlu dikaitkan dengan target nominal 200 citra

Target 180–220 citra dengan 30–50 objek/citra secara penuh memberi kisaran kasar 5.400–11.000 objek. Persamaan `N_box ≈ 6.000–10.000` konsisten dengan sasaran nominal 200 citra, tetapi kalimat sekarang dapat dibaca sebagai turunan dari seluruh rentang 180–220.

Revisi kecil: jelaskan bahwa `6.000–10.000` dihitung berdasarkan sasaran nominal sekitar 200 citra.

### 7. Kriteria split 3.2.4 perlu diselaraskan dengan syarat test 3.6.5

3.2.4 menyebut sasaran sedikitnya sekitar lima citra sumber per kelas pada test, sedangkan 3.6.5 menambahkan syarat operasional sedikitnya 10 objek pada sedikitnya 5 citra sumber. Agar split tidak perlu diperbaiki belakangan, kriteria test pada 3.2.4 sebaiknya langsung merujuk kedua syarat tersebut.

## C. Konsistensi gambar

### 8. Gambar 3.1 masih menggunakan istilah lama

`assets/alur_penelitian.svg` masih menulis:

- "Pembentukan model acuan YOLO26n";
- "Pengujian ulang beberapa seed".

Setelah revisi staging, sebaiknya diubah menjadi:

- "Pembentukan model acuan pengembangan";
- "Pelatihan ulang dengan seed konfirmasi";
- langkah berikutnya tetap "Evaluasi akhir pada data uji".

### 9. Gambar 3.2 secara umum konsisten, tetapi footer perlu mengikuti istilah staging final

`assets/arsitektur_frekuensi_yolo26.svg` sudah konsisten dengan frontend sebelum YOLO26n dan alur analisis spektral. Footer yang menyebut "pengujian multi-seed" sebaiknya mengikuti istilah final setelah 3.6.3 diperbaiki, misalnya "pelatihan ulang seed konfirmasi".

## D. Parameter yang masih perlu dibekukan sebelum eksperimen akhir

Hal berikut bukan kontradiksi internal, tetapi belum cukup konkret untuk reproduksibilitas penuh:

- jumlah replikasi dan tingkat interval pada paired bootstrap;
- jumlah warm-up dan pengulangan benchmark latency;
- versi final Ultralytics/PyTorch/CUDA;
- operating point precision/recall evaluator;
- kontrak `best.pt`/early stopping;
- kontrak numerik output frontend;
- metode CAM dan target layer final setelah kompatibilitas diverifikasi.

Parameter tersebut harus dibekukan sebelum test set dibuka dan tidak boleh ditentukan berdasarkan hasil test.

## E. Putusan audit

Secara struktur, BAB III sudah jauh lebih konsisten setelah konsolidasi. Tidak ditemukan konflik besar pada tujuan utama, definisi `C*`, grouped split, pemisahan development seed dan confirmation seeds, metrik utama, atau prinsip fairness arsitektur.

Namun BAB III **belum sebaiknya dianggap final** sebelum dua kelompok pekerjaan selesai:

1. perbaikan internal yang tidak memerlukan eksperimen: benturan notasi radial, staging Tahap III, konsistensi Gambar 3.1/3.2, penyelarasan kriteria split, dan redaksi C5;
2. verifikasi implementasi yang memang harus dilakukan sebelum eksperimen utama: kontrak rentang input/output frontend serta perilaku versi Ultralytics/checkpoint/optimizer.
