# Audit Sinkronisasi BAB II terhadap BAB III Final

## Status

**PASS dengan revisi minor.** Struktur dan argumen BAB II sudah menopang metodologi BAB III final. Tidak ditemukan konflik konsep mengenai model utama, posisi preprocessing, AFAB-2, CLAHE, wavelet, maupun RT-DETRv3-R18.

## 1. Bagian yang Sudah Konsisten

- SNI digunakan sebagai konteks taksonomi, sedangkan jumlah kelas final ditentukan oleh dataset dan audit kecukupan.
- Literatur dataset deteksi membahas jumlah citra sekaligus jumlah objek, sesuai rancangan dataset primer BAB III.
- YOLO26n ditempatkan sebagai model utama dengan backbone/neck/head tetap.
- YOLO26 dijelaskan sebagai model end-to-end dengan jalur utama tanpa NMS tambahan, konsisten dengan implementasi yang dikunci pada BAB III.
- RT-DETRv3-R18 hanya menjadi evaluasi transfer antararsitektur yang opsional dan tidak digunakan untuk memilih C*.
- Fine-grained dijelaskan berdasarkan kedekatan karakteristik visual, bukan semata-mata jumlah kelas.
- CLAHE diposisikan sebagai kontrol peningkatan kontras lokal, dan hasil Syauqi et al. tidak diatribusikan kepada CLAHE saja karena pipeline mereka komposit.
- Wavelet dibahas sebagai alternatif konseptual dan bukan baseline utama.
- AFAB-2 disebut sebagai sumber Xu et al. (2025), bukan metode yang diciptakan penelitian ini.
- Penelitian tidak mengadopsi AFAB-1, CGFI, FTIF, atau keseluruhan LFDet.
- Tidak ada klaim bahwa cacat biji kopi memiliki frequency signature khusus yang sudah terbukti.
- Eigen-CAM tetap kandidat visualisasi, bukan bukti kausal dan bukan bagian dari pemilihan C*.

## 2. Verifikasi Langsung Xu et al. (2025)

Pemeriksaan paper primer mengonfirmasi bahwa:

- Subbab 3.1.1 paper adalah 2D-DFT;
- Persamaan (1) adalah DFT, (2) amplitudo, (3) fase, dan (4) inverse DFT;
- patch-wise DFT berada pada Subbab 3.3.1 dan menggunakan m=32;
- AFAB-2/patch-specific chaotic amplitude suppressor membentuk angular density, entropy, adaptive threshold, dan amplitude remodeling pada Persamaan (9)–(13);
- rekonstruksi menggunakan fase asli;
- recovered spatial domain dinormalisasi min-max, dikalikan dengan raw spatial domain, lalu digabung secara residual.

Dengan demikian, atribusi utama di Subbab 2.8 sudah didukung sumber primer.

## 3. Revisi Minor yang Disarankan

### 2.5 YOLO26 dan Pembanding Arsitektur

Pada paragraf yang membahas rancangan penelitian, gunakan `YOLO26n` ketika merujuk secara spesifik pada model yang dipakai eksperimen, bukan `YOLO26` secara generik. Penjelasan keluarga/model secara teori tetap boleh menggunakan istilah YOLO26.

### 2.9 Visualisasi Aktivasi Model

Paragraf pembuka saat ini menyebut perbandingan respons “YOLO26 tanpa prapemrosesan dan YOLO26 dengan prapemrosesan frekuensi-angular”. Sinkronkan dengan desain akhir menjadi perbandingan kondisi utama yang relevan (B0, B1, B2, B3), dengan metode/layer/normalisasi yang sama. Jika C*=C0, kondisi identik tidak perlu diduplikasi.

### 2.10 Penelitian Terkait

Baris penelitian yang diusulkan sudah benar. Jika redaksi efisiensi diperjelas, gunakan istilah biaya komputasi atau latency end-to-end secara umum; detail timing tetap berada di BAB III.

## 4. Hal yang Tidak Perlu Dipindahkan ke BAB II

- seed development/confirmation;
- aturan pemilihan C* dan tie-break;
- konvensi diskret DC;
- kontrak output residual [0,2];
- optimizer Auto dan best.pt;
- max_det, paired bootstrap, atau protokol benchmark rinci.

Detail tersebut adalah kontrak implementasi/metodologi dan sudah tepat berada di BAB III.

## 5. Temuan Repo di Luar BAB II

`01_PROPOSAL_SKELETON.md` masih memuat beberapa kontrak BAB III versi lama, khususnya seed Tahap III `42,123,2026`, seed visualisasi `42`, dan struktur beberapa subbab BAB III yang telah berubah. Skeleton perlu disinkronkan setelah audit BAB I–II agar tidak menjadi sumber instruksi yang bertentangan dengan naskah final.

## Putusan

BAB II tidak memerlukan perubahan teori atau posisi penelitian. Revisi hanya berupa penyelarasan istilah model spesifik dan cakupan visualisasi terhadap desain eksperimen akhir.