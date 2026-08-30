# Catatan Revisi Subbab 3.4 — Prapemrosesan Citra Berbasis Frekuensi-Angular

Dokumen ini mencatat keputusan revisi untuk Subbab 3.4 sebelum diterapkan ke naskah utama BAB III.

## 1. Pembuka Subbab 3.4

- Pertahankan penjelasan bahwa metode mengadaptasi mekanisme AFAB-2 Xu et al. (2025), bukan keseluruhan LFDet atau AFAB-1.
- Pertegas pemisahan antara komponen yang mengacu pada AFAB-2 dan keputusan adaptasi penelitian.
- Komponen yang diposisikan sebagai adaptasi penelitian harus disebut secara eksplisit, termasuk pemrosesan per kanal RGB, diskretisasi angular, overlap patch, konstanta stabilitas numerik, aturan padding, penggabungan patch, dan variasi C1 sampai C5.
- Hindari pengulangan kalimat defensif seperti "bukan nilai optimal" atau "bukan bagian metode asli" pada setiap paragraf; cukup nyatakan status sumber/implementasi secara ringkas pada tempat yang relevan.

## 2. Subbab 3.4.1 — Pembentukan Patch Lokal

- Pertahankan notasi citra I dan patch Pi.
- Pertahankan m = 32 sebagai konfigurasi referensi.
- Pertahankan overlap 50% sehingga stride s = 16.
- Jelaskan secara ringkas bahwa ukuran patch 32 mengikuti konfigurasi referensi Xu et al. (2025), sedangkan overlap 50% merupakan keputusan implementasi penelitian.
- Pertahankan replicate padding untuk membentuk grid patch dan pemotongan kembali ke ukuran H x W setelah rekonstruksi.
- Padatkan penjelasan alasan patch lokal: cukup jelaskan bahwa pemrosesan lokal menjaga keterkaitan respons frekuensi dengan wilayah tertentu pada citra.

## 3. Subbab 3.4.2 — Transformasi Fourier

- Tidak ada perubahan metode.
- Pertahankan FFT dua dimensi, normalisasi ortonormal, FFT shift, amplitudo, dan fase.
- Amplitudo digunakan untuk analisis distribusi frekuensi; fase dipertahankan untuk rekonstruksi.
- Sebelum IFFT, spektrum dikembalikan dengan inverse FFT shift.
- Hilangkan pengulangan penjelasan tentang fungsi amplitudo, fase, dan pemusatan spektrum.

## 4. Subbab 3.4.3 — Distribusi Angular

- Pertahankan pemetaan koordinat non-DC ke sudut relatif terhadap pusat spektrum.
- Pertahankan 360 interval angular pada konfigurasi referensi C0 sebagai keputusan implementasi penelitian.
- Pertahankan pemrosesan per kanal RGB dan epsilon = 1e-8.
- Revisi penting: komponen DC tidak lagi dipetakan ke bin angular pertama.
- Komponen DC dikeluarkan dari perhitungan distribusi angular karena tidak memiliki arah, tetapi tetap dipertahankan pada spektrum untuk rekonstruksi.
- Densitas angular dihitung hanya untuk koordinat dengan r(u,v) > 0.
- Hindari klaim bahwa 360 bin memberikan resolusi angular efektif 1 derajat pada seluruh grid; cukup sebut sebagai diskretisasi nominal.

## 5. Subbab 3.4.4 — Ambang Adaptif Berdasarkan Entropi

- Pertahankan formulasi entropi H dan ambang tau.
- Pertahankan gamma = 0,10 sebagai konfigurasi referensi sementara, dengan atribusi ke Xu et al. yang harus diverifikasi langsung sebelum finalisasi.
- Pertahankan batas gamma/2 <= tau < gamma dan contoh rentang 0,05 sampai kurang dari 0,10 untuk gamma = 0,10.
- Padatkan penjelasan; tidak perlu mengulang bahwa ambang dibatasi atau bahwa gamma bukan nilai optimal untuk kopi pada beberapa kalimat.

## 6. Subbab 3.4.5 — Pembobotan Respons Spektral

- Pertahankan normalisasi q terhadap densitas angular maksimum.
- Pertahankan hard threshold pada konfigurasi referensi.
- Revisi istilah: respons di atas ambang bukan "dipertahankan utuh", tetapi dibobot sesuai densitas relatif q.
- Karena 0 <= w <= 1, tahap ini hanya menekan, menghilangkan, atau mempertahankan amplitudo, tanpa memperbesar koefisien Fourier di atas nilai asal.
- Tambahkan aturan eksplisit untuk DC agar konsisten dengan 3.4.3:
  - koordinat non-DC dibobot menggunakan w(b(u,v));
  - komponen DC dipertahankan tanpa pembobotan angular.

## 7. Subbab 3.4.6 — Rekonstruksi dan Penggabungan Residual

- Pertahankan IFFT, penggabungan patch overlap, normalisasi respons G, dan residual I' = I + I * G.
- Pada C0, penggabungan patch overlap tetap menggunakan rata-rata sebagai keputusan implementasi.
- Pertahankan ukuran spasial H x W sehingga bounding box tidak berubah.
- Padatkan penjelasan fase karena sudah didefinisikan pada 3.4.2.
- Revisi penting: jangan mengunci keputusan bahwa keluaran residual dibiarkan pada rentang teoritis [0,2] tanpa clipping.
- Kontrak rentang numerik input sebelum YOLO26n harus ditetapkan secara konsisten agar kondisi tanpa prapemrosesan, CLAHE, dan frekuensi-angular tidak berbeda karena skala intensitas yang tidak terkontrol.
- Keputusan final antara clipping, renormalisasi, atau penempatan frontend sebelum normalisasi YOLO harus ditentukan setelah mencocokkan implementasi AFAB-2 dan pipeline aktual.

## 8. Prinsip Redaksi

- Hindari over-wording dan pengulangan alasan yang sama.
- Satu paragraf sebaiknya memiliki satu fungsi utama.
- Gunakan istilah "mengacu pada AFAB-2" hanya untuk komponen yang benar-benar didukung sumber.
- Gunakan istilah "keputusan implementasi penelitian" untuk detail yang ditentukan sendiri.
- Jangan mengubah keputusan desain menjadi klaim optimal sebelum diuji.
