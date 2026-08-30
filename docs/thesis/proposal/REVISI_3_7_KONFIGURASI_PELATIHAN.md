# Catatan Revisi Subbab 3.7 — Konfigurasi Pelatihan

Catatan ini merekam keputusan revisi untuk Subbab 3.7 sebelum perubahan diterapkan ke `BAB_III_METODOLOGI_PENELITIAN.md`.

## Keputusan utama

1. Seed pengembangan tetap `42`, tetapi seed tersebut tidak lagi dimasukkan ke rerata konfirmasi utama. Tahap konfirmasi menggunakan tiga seed baru yang tidak dipakai untuk memilih konfigurasi `C*`. Notasi sementara: `S_conf = {123, 2026, s3}`, dengan `s3` ditetapkan secara konkret sebelum Tahap III dimulai.
2. `optimizer=Auto` boleh dipertahankan selama versi Ultralytics dikunci. Optimizer aktual dan parameter penting yang dipilih secara internal harus dicatat dan diperiksa agar konsisten pada seluruh kondisi utama yang dibandingkan.
3. Kriteria pemilihan `best.pt` dan penghentian dini harus diverifikasi pada versi Ultralytics yang digunakan. Jika mekanisme bawaan menggunakan fitness yang tidak identik dengan `mAP50-95`, hal tersebut harus dinyatakan secara eksplisit; bila implementasinya sederhana, pemilihan checkpoint berdasarkan validation `mAP50-95` lebih selaras dengan metrik utama penelitian.
4. Batas maksimum 50 epoch dan patience 15 tetap dapat digunakan sebagai rancangan awal, tetapi baseline pengembangan harus digunakan untuk memastikan batas 50 epoch tidak memotong pelatihan yang masih jelas membaik. Jika batas perlu dinaikkan, perubahan dilakukan sebelum eksperimen utama dan diterapkan sama pada seluruh kondisi utama.
5. Ukuran masukan 640×640 dan batch 16 dipertahankan. Jika batch 16 tidak dapat digunakan pada kondisi terberat, satu ukuran batch yang layak harus ditetapkan dan digunakan sama pada seluruh kondisi utama.
6. Parameter implementasi lain seperti data loader, cache, mosaic, dan parameter prediksi ditetapkan secara tetap dan dicatat, tetapi tidak diperlakukan sebagai faktor penelitian.

## Redaksi yang disarankan

Setelah tabel konfigurasi, penjelasan dapat dipadatkan menjadi:

> Seluruh kondisi pada tahap yang sama menggunakan pembagian data, augmentasi, ukuran masukan, ukuran batch, batas epoch, penghentian dini, dan lingkungan komputasi yang sama. Perbedaan epoch berhenti diperbolehkan karena penghentian dini mengikuti kinerja validasi masing-masing run.
>
> Versi Ultralytics dikunci selama eksperimen. Jika `optimizer=Auto` digunakan, optimizer dan parameter aktual yang dipilih secara internal dicatat dan diperiksa agar konsisten antar kondisi. Kriteria pemilihan checkpoint `best.pt` dan penghentian dini juga diverifikasi pada versi perangkat lunak yang digunakan serta diterapkan identik pada seluruh run.
>
> Parameter implementasi lain seperti data loader, cache, mosaic, dan konfigurasi prediksi ditetapkan secara tetap dan dicatat, tetapi tidak diperlakukan sebagai faktor penelitian.

## Catatan konsistensi lintas subbab

- Revisi seed harus konsisten dengan Subbab 3.6.3 dan 3.6.5.
- Aturan checkpoint harus konsisten dengan metrik utama pada Subbab 3.8.
- Versi Ultralytics dan lingkungan komputasi harus konsisten dengan Subbab 3.12.
