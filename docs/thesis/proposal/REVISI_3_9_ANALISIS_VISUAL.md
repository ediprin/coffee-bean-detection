# Catatan Revisi Subbab 3.9 — Analisis Visual

Dokumen ini mencatat keputusan revisi yang telah disepakati untuk Subbab 3.9 sebelum diterapkan ke `BAB_III_METODOLOGI_PENELITIAN.md`.

## 3.9 Analisis Visual

- Analisis visual tetap berfungsi sebagai pendukung interpretasi hasil kuantitatif, bukan bukti tunggal mengenai penyebab peningkatan atau penurunan kinerja.
- Perubahan visual pada tekstur, kontras, spektrum, atau aktivasi tidak boleh langsung ditafsirkan sebagai bukti bahwa representasi tersebut lebih baik bagi detektor.

## 3.9.1 Visualisasi Tahapan Prapemrosesan

- Urutan visualisasi tetap mencakup citra masukan, patch lokal, amplitudo spektrum Fourier, distribusi angular/radial-angular sesuai konfigurasi, ambang dan bobot spektral, hasil IFFT, respons rekonstruksi, dan citra hasil penggabungan residual.
- Perbandingan antara konfigurasi referensi dan konfigurasi terpilih harus menggunakan citra sumber yang sama agar perubahan yang terlihat dapat dibandingkan secara langsung.
- Visualisasi radial-angular hanya ditampilkan apabila konfigurasi yang dianalisis memang menggunakan komponen radial-angular.
- Hindari klaim subjektif seperti “lebih tajam berarti lebih baik”; visualisasi hanya digunakan untuk menjelaskan perubahan representasi.

## 3.9.2 Visualisasi Respons Model

- Eigen-CAM tetap dipertimbangkan sebagai kandidat utama, tetapi tidak dikunci sebagai metode final sebelum kompatibilitasnya dengan YOLO26, target layer, dan prosedur ekstraksi telah diverifikasi.
- Jika Eigen-CAM tidak dapat diterapkan secara konsisten, digunakan metode visualisasi aktivasi lain yang dapat diterapkan identik pada seluruh kondisi.
- Metode visualisasi, target layer, ukuran masukan, dan normalisasi heatmap harus sama pada seluruh kondisi yang dibandingkan.
- Seed 42 tidak lagi digunakan sebagai seed visualisasi akhir karena ditetapkan sebagai development seed.
- Visualisasi utama menggunakan satu seed konfirmasi dari `S_conf` yang ditetapkan sebelum hasil visual diperiksa. Seed visualisasi tidak boleh dipilih berdasarkan heatmap yang paling menguntungkan.

## 3.9.3 Visualisasi Hasil Deteksi

- Kondisi `B0`, `B1`, `B2`, dan `B3` dibandingkan pada citra yang sama menggunakan parameter prediksi yang identik, termasuk confidence threshold, IoU yang relevan, dan `max_det`.
- Seed visualisasi menggunakan seed konfirmasi yang telah ditetapkan sebelumnya, bukan seed development 42.
- Jika `C* = C0`, maka `B2` dan `B3` adalah kondisi identik dan tidak perlu ditampilkan sebagai dua panel seolah-olah merupakan dua metode berbeda.
- Contoh citra dipilih berdasarkan kategori analisis yang ditentukan sebelumnya, misalnya kasus berhasil, gagal, kelas dengan kinerja tinggi/rendah, dan kasus dengan perbedaan antar kondisi, untuk mengurangi cherry-picking.
- Penetapan kategori kinerja tinggi/rendah harus berdasarkan hasil evaluasi yang jelas, bukan pemilihan intuitif.

## Arah Redaksi

Subbab 3.9 perlu dipadatkan agar menekankan tiga prinsip: visualisasi bersifat pendukung, prosedur visual harus konsisten antar kondisi, dan pemilihan seed/contoh tidak dilakukan berdasarkan tampilan yang paling menguntungkan.
