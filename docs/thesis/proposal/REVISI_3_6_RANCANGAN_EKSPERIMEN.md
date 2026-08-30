# Catatan Revisi 3.6 Rancangan Eksperimen

Dokumen ini mencatat keputusan revisi untuk Subbab 3.6 sebelum perubahan diterapkan ke `BAB_III_METODOLOGI_PENELITIAN.md`.

## 3.6.1 Tahap I — Pembentukan Model Acuan

- Gunakan istilah **model acuan pengembangan** agar tidak rancu dengan baseline multi-seed pada Tahap III.
- Tetapkan seed pengembangan secara eksplisit sebagai `s_dev = 42`.
- Model acuan pengembangan diinisialisasi langsung dari `yolo26n.pt`.
- Checkpoint hasil model acuan tidak diwariskan ke konfigurasi C0-C5.
- Kelompok tiga kelas sulit `H` ditetapkan sekali dari hasil validasi baseline pengembangan dan kemudian dibekukan.
- Padatkan pemeriksaan awal menjadi audit format data, distribusi kelas, anotasi, pembagian dataset, dan keluaran preprocessing.

## 3.6.2 Tahap II — Pengujian Variasi Prapemrosesan

- Semua C0-C5 dimulai kembali dari `yolo26n.pt` dengan seed pengembangan 42; urutan C0→...→C5 hanya menunjukkan akumulasi desain, bukan pewarisan checkpoint.
- Seed 42 berfungsi sebagai **development selection seed**.
- C_str dipilih dari C0-C5 menggunakan mAP50-95 validasi.
- Selisih absolut mAP50-95 < 0,001 diperlakukan sebagai **aturan operasional penelitian** untuk kondisi praktis seri, bukan standar universal.
- Tie-break: AP_H, kemudian waktu pemrosesan total dengan protokol benchmark yang konsisten dengan Subbab efisiensi.
- Analisis sensitivitas dilakukan setelah C_str dipilih. C* hanya dapat berasal dari konfigurasi yang benar-benar telah dievaluasi.
- C* dibekukan setelah Tahap II dan tidak dipilih ulang berdasarkan hasil multi-seed atau data uji.
- C* boleh sama dengan C0 jika konfigurasi referensi memang terbaik.
- Ringkas pengulangan tentang larangan penggunaan test set; cukup tegaskan seluruh pemilihan konfigurasi menggunakan data validasi.

## 3.6.3 Tahap III — Pengujian Ulang dengan Beberapa Seed

- Pisahkan seed pengembangan 42 dari seed konfirmasi utama.
- Gunakan tiga seed konfirmasi baru yang belum pernah dipakai untuk memilih C*, misalnya `S_conf = {123, 2026, s3}`, dengan `s3` ditetapkan sebelum Tahap III dimulai.
- Hasil seed 42 boleh dilaporkan sebagai hasil pengembangan, tetapi tidak dicampur ke rerata konfirmasi utama.
- Pada setiap seed konfirmasi, B0-B3 dibangun langsung dari `yolo26n.pt` dengan kondisi awal parameter yang setara.
- Gunakan perbandingan berpasangan per seed: `Δ_s = M_perlakuan,s - M_B0,s`.
- Laporkan hasil per seed, rerata Δ, dan variasinya; fungsi multi-seed adalah melihat arah serta kestabilan efek.
- Jika `C* = C0`, maka B2 dan B3 identik; tidak perlu menjalankan eksperimen duplikat.
- CLAHE tetap dipertahankan sebagai kontrol konvensional.
- Pastikan kontrak rentang numerik input antar kondisi konsisten dengan keputusan revisi 3.4.6.

## 3.6.4 Evaluasi pada Arsitektur Lain — Opsional

- Pertahankan RT-DETRv3-R18 sebagai analisis tambahan untuk transfer/generalitas metode, bukan komparasi arsitektur utama.
- C* diterapkan tanpa tuning ulang khusus untuk RT-DETRv3-R18.
- Samakan split data, konfigurasi preprocessing, definisi kelas, dan protokol evaluasi.
- Hyperparameter training yang memang spesifik arsitektur boleh mengikuti konfigurasi tetap RT-DETRv3-R18, asalkan sama untuk kondisi tanpa dan dengan C*.
- Hasil RT-DETR tidak digunakan untuk memilih ulang C*.
- Eksperimen ini tetap opsional jika sumber daya komputasi tidak memadai.

## 3.6.5 Evaluasi Akhir pada Data Uji

- Hapus estimasi tetap “sekitar 30 citra sumber”. Proporsi sekitar 15% adalah target grouped split, sedangkan jumlah citra, objek, dan kelompok aktual dilaporkan setelah pembagian selesai.
- Pertahankan kriteria minimal 10 objek pada sedikitnya 5 citra sumber per kelas sebagai **kriteria operasional penelitian**, bukan standar statistik universal.
- Keterwakilan test diperiksa dari ground truth sebelum eksperimen utama, bukan dari hasil prediksi model.
- Jika dukungan test belum cukup, tambah data atau perbaiki komposisi grouped split sebelum eksperimen utama.
- Hapus grouped cross-validation sebagai fallback otomatis setelah C* dipilih, karena dapat mencampurkan kembali data pengembangan dan evaluasi.
- Test set hanya dibuka setelah C*, seed konfirmasi, aturan checkpoint, metrik, dan prosedur evaluasi dibekukan.
- Evaluasi akhir utama mengikuti seed konfirmasi, bukan mencampurkan seed development 42 ke rerata konfirmasi.
- Setelah data uji dibuka, tidak dilakukan tuning ulang atau pemilihan ulang konfigurasi.

## Ringkasan keputusan lintas 3.6

Alur final yang dituju:

`development seed 42 → pilih C* → freeze C* → confirmation seeds baru → final test`

Dengan pemisahan ini, tahap pengembangan, konfirmasi, dan evaluasi akhir memiliki fungsi yang lebih jelas dan risiko bias seleksi dapat dikurangi.
