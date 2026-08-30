# Audit Sinkronisasi BAB I terhadap BAB III Final

## Status

**PASS dengan revisi minor.** BAB I secara substansi sudah konsisten dengan BAB III final. Subbab 1.2 Rumusan Masalah dan 1.4 Tujuan Penelitian berstatus LOCKED dan tidak memerlukan perubahan.

## 1. Konsistensi yang Sudah Sesuai

- dataset utama tetap dataset primer;
- target awal tetap 20 kategori cacat fisik/benda asing SNI + 1 kelas normal;
- jumlah kelas akhir ditetapkan setelah audit kecukupan data;
- model utama YOLO26n;
- baseline tanpa prapemrosesan dan kontrol CLAHE sudah konsisten;
- optimasi diposisikan sebagai pengujian variasi desain, bukan pencarian global optimum;
- RT-DETRv3-R18 tetap opsional;
- mAP50-95 tetap metrik utama;
- ruang lingkup akuisisi terkontrol dan tidak mengklaim robustness lapangan;
- BAB I tidak memuat hasil internal/pilot repo.

## 2. Revisi Minor yang Disarankan

### 1.1 Latar Belakang

Kalimat penutup menggunakan frasa “Penelitian selanjutnya menganalisis variasi desain ...”. Redaksi ini berpotensi terbaca sebagai penelitian lain di masa depan. Ubah menjadi “Penelitian ini kemudian menganalisis variasi desain ...” atau bentuk setara.

### 1.3 Batasan Masalah — Butir 7

Sinkronkan istilah efisiensi dengan Subbab 3.11. Efisiensi utama sebaiknya menyebut latency total end-to-end, waktu prapemrosesan, waktu inferensi model, throughput/FPS pada protokol yang sama, serta peak allocated GPU memory. Jumlah parameter dapat disebut sebagai informasi tambahan karena frontend tidak menambah parameter trainable.

## 3. Hal yang Tidak Perlu Ditambahkan ke BAB I

- detail seed 42 dan S_conf;
- C0-C5 secara rinci;
- aturan DC, output [0,2], optimizer Auto, best.pt, atau end-to-end post-processing;
- paired bootstrap dan detail benchmark.

Detail tersebut sudah tepat ditempatkan di BAB III dan akan membuat BAB I terlalu teknis jika dipindahkan.

## 4. Judul Kerja

Judul kerja pada skeleton menggunakan “pada YOLO26”, sedangkan eksperimen utama memakai varian YOLO26n. Ini **bukan kontradiksi substantif**, karena YOLO26n adalah varian keluarga YOLO26. Jika ingin presisi metodologis maksimum, judul dapat menggunakan “YOLO26n”, tetapi perubahan judul tidak diperlukan untuk konsistensi BAB I dan tidak dilakukan dalam audit ini.

## Putusan

BAB I tidak memerlukan perubahan konsep. Hanya dua penyelarasan redaksi pada bagian non-locked yang direkomendasikan. Rumusan masalah dan tujuan penelitian tetap dipertahankan apa adanya.