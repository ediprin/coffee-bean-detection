# Resolusi Blocker Teknis BAB III

Dokumen ini mencatat hasil verifikasi implementasi yang digunakan untuk mengunci kontrak prapemrosesan AF2 dan perilaku pelatihan/evaluasi Ultralytics 8.4.96.

## 1. Kontrak AF2 Referensi (`C0`)

Konfigurasi referensi tesis mengikuti **retained AF2 operator** yang telah digunakan pada eksperimen repo, agar `C0` tetap dapat direproduksi dan dibandingkan dengan bukti eksperimen sebelumnya.

Kontrak yang dibekukan:

- input AF2 adalah tensor RGB floating point yang telah dinormalisasi oleh pipeline YOLO ke rentang dasar `[0,1]`;
- patch size `32`, overlap `0.50`, `gamma=0.10`, `angular_bins=360`, `eps=1e-8`;
- kanal RGB diproses independen;
- patch overlap direkonstruksi dengan fold/overlap averaging;
- koordinat DC mengikuti konvensi diskret implementasi retained AF2 dan dipetakan ke bin angular `0`; aturan ini merupakan keputusan transfer implementasi, bukan definisi arah fisik komponen DC dari paper Xu et al.;
- gate spasial dibentuk melalui min-max normalization pada recovered response;
- keluaran operator adalah `x_AF2 = x + x * minmax(recover_AF2(x))`;
- tidak ada clipping atau renormalisasi tambahan setelah residual;
- apabila input berada pada `[0,1]`, rentang teoritis keluaran residual adalah `[0,2]`.

Paper Xu et al. mendukung urutan min-max recovered response, perkalian elemen dengan raw spatial domain, dan residual addition. Paper tidak menentukan aturan diskret DC maupun clipping pasca-residual. Karena itu kedua detail tersebut disebut sebagai kontrak transfer implementasi penelitian.

Implikasi metodologis: perbandingan `B2-B0` mengukur efek **frontend AF2 lengkap**, termasuk perubahan distribusi intensitas akibat residual gate. Perbandingan antar `C0-C5` mempertahankan kontrak output yang sama agar perubahan struktur prapemrosesan tidak bercampur dengan perubahan post-processing numerik.

## 2. Konsekuensi untuk Variasi `C1-C5`

Karena rancangan bersifat kumulatif dan setiap tahap dimaksudkan menambah satu perubahan, konvensi DC tidak boleh berubah diam-diam pada `C2` atau `C3`.

- `C1`: hanya menambah Hann + normalized overlap-add.
- `C2`: hanya mengubah arah bertanda menjadi orientasi tak bertanda; DC tetap mengikuti bin `0` sebagai konvensi diskret.
- `C3`: hanya menambah pemisahan radial; DC ditempatkan pada pita radial pertama dan bin angular/orientasi `0` agar tidak memperkenalkan faktor kedua.
- `C4` dan `C5` mewarisi kontrak output residual tanpa clipping.

## 3. Ultralytics 8.4.96 — Checkpoint dan Early Stopping

Verifikasi source Ultralytics 8.4.96 menunjukkan:

- validator deteksi menghasilkan `fitness` dari `DetMetrics.fitness()`;
- bobot fitness deteksi adalah `[0,0,0,1]` untuk `[P, R, mAP50, mAP50-95]`;
- dengan demikian `fitness = mAP50-95`;
- trainer menyimpan `best.pt` ketika `self.best_fitness == self.fitness`;
- `EarlyStopping` memperbarui epoch terbaik ketika fitness meningkat dan menghentikan training setelah `patience` epoch tanpa peningkatan.

Dengan demikian, pada Ultralytics 8.4.96 aturan `best.pt` dan early stopping **selaras langsung dengan metrik utama tesis `mAP50-95`**. Tidak diperlukan mekanisme checkpoint selection tambahan.

## 4. Ultralytics 8.4.96 — `optimizer=Auto`

Source Ultralytics 8.4.96 menetapkan:

- jika `iterations > 10000`, `Auto` memilih `MuSGD`;
- selain itu, `Auto` memilih `AdamW`;
- `iterations = ceil(N_train / max(batch_size, nbs)) * epochs`;
- default `nbs=64`;
- learning rate Auto untuk AdamW adalah `round(0.002*5/(4+nc), 6)` dengan beta1/momentum `0.9`.

Untuk rancangan dataset proposal (sekitar 70% dari 180–220 citra, batch 16, epoch maksimum awal 50), `iterations` jauh di bawah 10000, sehingga `optimizer=Auto` akan resolve ke **AdamW**. Optimizer aktual tetap dicatat pada log preflight/run sebagai verifikasi kontrak.

Jika `C=21`, learning rate Auto adalah `0.0004`. Jika jumlah kelas final berubah, nilai learning rate mengikuti formula Auto yang sama dan dicatat.

## 5. Precision/Recall dan Post-processing Validasi

Pada Ultralytics 8.4.96:

- apabila `conf` validasi tidak diberikan, validator menggunakan prefilter `conf=0.001` untuk tugas detect;
- precision dan recall ringkasan **tidak** semata-mata dihitung pada confidence `0.001`;
- `ap_per_class` memilih indeks pada maksimum kurva F1 rata-rata yang telah dihaluskan, lalu melaporkan precision dan recall pada indeks tersebut;
- konfigurasi YOLO26 menetapkan `end2end=True`;
- pada jalur end-to-end, fungsi post-processing Ultralytics hanya memfilter keluaran berdasarkan confidence dan membatasi jumlah prediksi dengan `max_det`, lalu langsung mengembalikannya; NMS tambahan tidak dijalankan.

Karena operating point ringkasan P/R dapat berbeda antar model, P/R tetap diperlakukan sebagai metrik deskriptif sekunder dan tidak digunakan untuk memilih `C*`. Metrik utama tetap `mAP50-95`, yang mengintegrasikan kurva precision-recall.

Untuk penelitian, `max_det=500` dipertahankan sama pada seluruh kondisi. Parameter IoU tetap relevan pada proses matching evaluasi AP, tetapi tidak boleh dijelaskan seolah-olah menjadi ambang NMS untuk keluaran end-to-end YOLO26.

## 6. Keputusan Naskah

Setelah verifikasi ini, naskah BAB III:

1. mengembalikan `C0` ke kontrak retained AF2 (DC → bin 0, residual tanpa clipping);
2. mengunci kontrak input `[0,1]` dan output residual teoritis `[0,2]`;
3. mengunci Ultralytics 8.4.96 sebagai versi eksperimen utama;
4. menyatakan `best.pt` dan early stopping menggunakan fitness yang sama dengan `mAP50-95`;
5. menyatakan `optimizer=Auto` resolve ke AdamW pada rancangan eksperimen dan mencatat optimizer aktual;
6. memperbaiki penjelasan P/R agar tidak menyebut satu operating point tetap yang sama antar model;
7. menyatakan YOLO26 sebagai end-to-end dan tidak mengatribusikan `iou=0.7` sebagai ambang NMS pada jalur evaluasi utama.