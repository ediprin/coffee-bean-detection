# Audit Final BAB III — Metodologi Penelitian

## Status

**PASS untuk tahap proposal.** Audit akhir tidak menemukan kontradiksi metodologis kritis pada `BAB_III_METODOLOGI_PENELITIAN.md` setelah kontrak AF2 dan perilaku Ultralytics 8.4.96 diverifikasi.

Naskah utama BAB III menjadi sumber keputusan metodologi aktif. Dokumen `REVISI_*.md` dipertahankan sebagai riwayat peninjauan; apabila terdapat perbedaan, keputusan pada naskah utama dan `RESOLUSI_BLOCKER_TEKNIS_BAB_III.md` berlaku.

## 1. Konsistensi Dataset dan Split

- Target awal tetap 21 kelas: 20 kategori cacat/benda asing SNI + 1 kelas normal.
- Jumlah kelas final dibekukan sebelum split dan pelatihan utama.
- Target pengumpulan 180–220 citra, nominal sekitar 200 citra, konsisten dengan estimasi 30–50 objek per citra.
- Estimasi nominal 6.000–10.000 objek tidak lagi diposisikan sebagai batas matematis seluruh rentang pengumpulan.
- Audit kelas menggunakan `N_obj,c`, `N_img,c`, dan `N_group,c`.
- Split dilakukan sebelum augmentasi dan berbasis `group_id` dengan target sekitar 70/15/15.
- Test set tetap tertutup sampai konfigurasi, seed, checkpoint rule, metrik, dan prosedur evaluasi dibekukan.

## 2. Konsistensi Model dan Kondisi Eksperimen

Empat kondisi utama konsisten di seluruh BAB III:

- `B0`: YOLO26n tanpa preprocessing tambahan;
- `B1`: CLAHE + YOLO26n;
- `B2`: `C0` + YOLO26n;
- `B3`: `C*` + YOLO26n.

Semua kondisi dimulai dari sumber `yolo26n.pt` yang sama dengan prosedur inisialisasi target-head yang setara pada seed yang sama. Checkpoint hasil baseline pengembangan tidak diwariskan ke C0–C5.

## 3. Kontrak AF2 Referensi (`C0`)

Kontrak final mengikuti retained AF2 operator di repo:

- input tensor RGB berada pada rentang dasar `[0,1]` setelah preprocessing YOLO;
- `m=32`, overlap `0.50`, `gamma=0.10`, 360 bin angular, `eps=1e-8`;
- RGB diproses per kanal;
- DC dipetakan ke bin `0` sebagai konvensi diskret implementasi, bukan sebagai arah fisik;
- rekonstruksi overlap menggunakan averaging;
- gate spasial menggunakan min-max normalization;
- residual `I' = I + I ⊙ G`;
- tidak ada clipping/renormalisasi pasca-residual;
- output teoritis untuk input `[0,1]` berada pada `[0,2]`.

Paper Xu et al. mendukung mekanisme patch-wise DFT, angular-density suppression, entropy-based threshold, amplitude remodeling, min-max gate, dan residual fusion. Diskretisasi RGB/DC serta kontrak clipping merupakan keputusan transfer implementasi penelitian.

## 4. Konsistensi `C0 → C5`

Rancangan tetap kumulatif dan bukan full factorial:

- `C1`: Hann + normalized overlap-add;
- `C2`: arah bertanda → orientasi tak bertanda;
- `C3`: tiga pita radial;
- `C4`: ambang lunak;
- `C5`: panduan luminansi bersama.

Konvensi DC dan output residual tidak berubah diam-diam antartahap. Analisis sensitivitas hanya menggunakan konfigurasi yang benar-benar dievaluasi; nilai terbaik dari parameter berbeda tidak boleh digabung tanpa run tambahan.

## 5. Staging Eksperimen dan Seed

Alur final:

`seed development 42 → pilih C* → freeze C* → confirmation seeds {123, 2026, 31415} → final test`

Seed 42 tidak dimasukkan ke rerata konfirmasi. Pada setiap seed konfirmasi, perbandingan dilakukan secara berpasangan terhadap B0. Jika `C*=C0`, run B2/B3 yang identik tidak diduplikasi.

Evaluasi RT-DETRv3-R18 tetap opsional, tidak digunakan untuk memilih C*, dan jika dijalankan sebaiknya menggunakan pasangan seed konfirmasi yang sama agar arah efek tidak bergantung pada satu realisasi acak.

## 6. Training, Checkpoint, dan Evaluasi

Eksperimen utama dikunci pada Ultralytics 8.4.96.

- Fitness deteksi = `mAP50-95`.
- `best.pt` dan early stopping menggunakan fitness yang sama dengan metrik utama.
- `optimizer=Auto` ter-resolve ke AdamW pada ukuran eksperimen yang direncanakan; optimizer dan learning rate aktual tetap dicatat.
- YOLO26 menggunakan `end2end=True`; jalur evaluasi tidak menjalankan NMS tambahan seperti head YOLO konvensional.
- `conf=0.001` pada validator merupakan prefilter, bukan operating point P/R tetap.
- Precision/recall ringkasan diperlakukan sebagai metrik sekunder; pemilihan C* menggunakan `mAP50-95`, lalu `AP_H`, lalu median latency sebagai tie-break operasional.

## 7. Metrik dan Analisis

Metrik utama dan tambahan sudah konsisten:

- utama: `mAP50-95`;
- sekunder: `mAP50`, precision, recall;
- per kelas: `AP_c,50:95`;
- kelompok sulit: `AP_H`, dengan H dibekukan dari `B0_dev` seed 42;
- indikator tail: `AP_worst`;
- multi-seed: hasil per seed, mean, SD, paired delta;
- paired bootstrap berbasis `group_id` hanya digunakan jika jumlah kelompok independen memadai.

Analisis visual dan error analysis tidak digunakan sebagai bukti tunggal atau sebagai dasar tuning ulang setelah test dibuka.

## 8. Verifikasi Sumber yang Relevan

Pemeriksaan akhir mengonfirmasi angka pembanding yang muncul di Subbab 3.2:

- Bahy & Rifai (2026): 107 citra, 20 klasifikasi cacat SNI, 13.863 anotasi;
- Tarekegn & Debelee (2025): 562 citra, 19.228 instance, 13 kelas cacat dan satu kelas normal.

Paper Xu et al. (2025) juga mengonfirmasi patch size 32, formulasi angular-density/entropy/threshold AFAB-2, `gamma=0.10` sebagai konfigurasi yang diuji, serta min-max gate + residual fusion.

## 9. Detail SOP yang Belum Menjadi Blocker Proposal

Tiga detail dapat dibekukan pada protokol eksekusi sebelum eksperimen terkait dijalankan, tanpa mengubah rancangan penelitian:

1. jumlah warm-up dan pengulangan benchmark latency;
2. metode visualisasi aktivasi/CAM yang benar-benar kompatibel dengan YOLO26;
3. konfigurasi training spesifik RT-DETRv3-R18 apabila analisis opsional tersebut dilakukan.

Ketiganya tidak digunakan untuk memilih desain utama sebelum aturan masing-masing dibekukan.

## Putusan Akhir

BAB III sudah cukup konsisten untuk digunakan sebagai metodologi proposal. Perubahan berikutnya sebaiknya hanya dilakukan apabila ada perubahan nyata pada dataset, kontrak eksperimen, atau hasil preflight implementasi—notasi dan keputusan metodologi utama tidak perlu dibuka ulang tanpa alasan baru.