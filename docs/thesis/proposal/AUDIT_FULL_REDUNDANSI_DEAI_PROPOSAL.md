# Audit Full Redundansi dan De-AI Proposal

Status: **FULL EDITORIAL AUDIT — DIAGNOSIS ONLY**

Ruang lingkup:
- `BAB_I_PENDAHULUAN.md`
- `BAB_II_TINJAUAN_PUSTAKA.md`
- `BAB_III_METODOLOGI_PENELITIAN.md`

Tujuan audit ini bukan memeriksa ulang kebenaran metodologi, melainkan mencari residu penulisan yang membuat naskah terasa seperti keluaran generatif: pengulangan gagasan, penjelasan yang sudah selesai tetapi dijelaskan kembali, kalimat defensif berlebihan, ringkasan berulang, transisi kosong, dan detail audit/debugging yang terlalu masuk ke naskah formal.

**Subbab 1.2 Rumusan Masalah dan 1.4 Tujuan Penelitian tetap LOCKED dan tidak menjadi target rewrite tanpa perintah eksplisit pengguna.**

## 1. Putusan umum

Naskah **belum lolos audit de-AI**. Struktur ilmiah dan kontrak metodologinya sudah kuat, tetapi masih terdapat cukup banyak pengulangan konseptual, terutama di BAB II dan BAB III. Masalah utama bukan fakta yang salah, melainkan satu gagasan yang dijelaskan penuh berkali-kali pada lokasi berbeda.

Prinsip revisi yang disarankan:

> Satu gagasan dijelaskan lengkap satu kali pada lokasi yang paling tepat. Pada lokasi lain cukup dirujuk, dinyatakan implikasinya, atau dihapus jika tidak menambah fungsi baru.

## 2. Peta gagasan yang paling sering berulang

### R1 — Arsitektur YOLO26n dipertahankan; perubahan berasal dari input

Muncul pada:
- BAB I 1.1 paragraf akhir;
- BAB I 1.3 butir 3;
- BAB II 2.5;
- BAB II 2.7 penutup;
- BAB III 3.1;
- BAB III 3.3;
- pembukaan 3.4.

**Diagnosis:** terlalu sering dijelaskan kembali. 1.3 boleh menyebutnya sebagai batasan. Penjelasan metodologis lengkap sebaiknya berada di 3.3. Di BAB II cukup satu kalimat penghubung atau bahkan tidak perlu.

### R2 — Data uji tidak digunakan untuk tuning/pemilihan C*

Muncul pada:
- 3.2.4;
- 3.5.6;
- 3.6.2;
- 3.6.3;
- 3.6.5;
- 3.8.

**Diagnosis:** aturan penting, tetapi terlalu banyak diulang sebagai kalimat lengkap. Aturan umum cukup dikunci di 3.2.4; 3.6.5 menjelaskan prosedur pembukaan test. Bagian lain cukup merujuk.

### R3 — Semua run berasal dari `yolo26n.pt`, bukan checkpoint sebelumnya

Muncul pada 3.3, 3.6.1, 3.6.2, dan 3.6.3.

**Diagnosis:** sebaiknya menjadi satu aturan umum eksperimen di awal 3.6. Setelah itu tidak perlu diulang pada setiap tahap.

### R4 — C* boleh sama dengan C0

Muncul pada 3.3.1, 3.6.2, 3.6.3, dan 3.9.3.

**Diagnosis:** cukup dijelaskan pada aturan pemilihan C* di 3.6.2. Di bagian lain cukup ditangani secara operasional tanpa mengulang pembelaannya.

### R5 — RT-DETRv3-R18 bersifat opsional dan bukan dasar memilih C*

Muncul pada 1.3, 2.5, 3.3.1, pembukaan 3.6, 3.6.4, 3.11, dan 3.12.

**Diagnosis:** terlalu banyak. Batasan 1.3 boleh menyebut opsional. Penjelasan metodologi cukup di 3.6.4. Bagian lain hanya perlu referensi singkat bila benar-benar diperlukan.

### R6 — Wavelet relevan tetapi bukan baseline utama

Muncul pada 1.3, 2.7, dan 3.3.1.

**Diagnosis:** cukup di 1.3 sebagai batasan dan 2.7 sebagai literatur. Penjelasan ulang di 3.3.1 dapat dibuang.

### R7 — AFAB-2 diadaptasi, bukan seluruh LFDet/AFAB

Muncul panjang di 2.8.3 dan kembali panjang di pembukaan 3.4.

**Diagnosis:** BAB II adalah tempat menjelaskan posisi terhadap LFDet. BAB III cukup satu kalimat sumber/adaptasi lalu langsung mendefinisikan operator dan keputusan implementasi.

### R8 — Visualisasi bukan bukti kausal / tidak menggantikan metrik

Muncul di 2.9, pembukaan 3.9, 3.9.1, dan 3.9.2.

**Diagnosis:** kehati-hatian ini benar, tetapi tiga kali di dalam 3.9 terlalu defensif. Cukup satu kali pada pembukaan 3.9. BAB II dapat menyebut keterbatasan CAM secara umum.

### R9 — Konvensi DC

Didefinisikan di 3.4.3, lalu diulang di 3.4.5, 3.5.2, dan 3.5.3.

**Diagnosis:** definisi lengkap cukup di 3.4.3. 3.5.2 cukup menyatakan konvensi diwarisi. 3.5.3 hanya perlu menyebut DC masuk pita radial pertama jika memang dibutuhkan untuk definisi pita.

### R10 — Operator tidak mengamplifikasi Fourier / residual tidak di-clipping

Muncul pada 3.4.5, 3.4.6, pembukaan 3.5, 3.5.4, dan 3.5.5.

**Diagnosis:** sifat operator referensi cukup dibuktikan di 3.4.5–3.4.6. Bagian variasi hanya perlu menyebut jika sebuah variasi mengubah sifat tersebut; jika tidak, tidak perlu ditegaskan kembali.

## 3. Audit BAB I

### 1.1 Latar Belakang

**Status: perlu pruning ringan–sedang.**

1. Paragraf awal tentang inspeksi manual masih relevan, tetapi pembuka “kualitas ... faktor penting ... penilaian kualitas” agak tautologis. Dapat dipadatkan tanpa kehilangan García et al.
2. Paragraf YOLO dan paragraf kinerja antarkelas memiliki fungsi berbeda dan layak dipertahankan.
3. Paragraf fine-grained sebagian mengulang paragraf sebelumnya. Bagian “tidak seluruh kategori harus sama sulitnya” terlalu defensif untuk latar belakang dan dapat dipadatkan.
4. Paragraf modifikasi internal model diperlukan untuk membentuk kontras dengan pendekatan input-space.
5. Paragraf preprocessing umum dan paragraf preprocessing pada komoditas pertanian mengulang fungsi yang sama: membuktikan preprocessing dapat ditempatkan sebelum detektor. Keduanya sebaiknya dipadatkan atau digabung secara konseptual.
6. Paragraf domain frekuensi adalah inti jembatan menuju metode dan perlu dipertahankan.
7. Paragraf akhir terlalu banyak mengulang kontrak metode: YOLO26n tetap, variasi desain, B0, CLAHE, biaya komputasi. Sebagai penutup latar belakang, cukup menyatakan metode usulan, objek evaluasi, dan pembanding utama secara ringkas.

### 1.2 Rumusan Masalah

**LOCKED. Tidak diubah.** Pengulangan terhadap latar belakang bersifat struktural dan dapat diterima.

### 1.3 Batasan Masalah

**Status: mayoritas perlu dipertahankan.** Pengulangan terhadap 1.1 bersifat fungsional karena batasan harus eksplisit. Yang perlu diperbaiki hanya gaya pembuka “Adapun ... agar ... yaitu” dan beberapa butir yang dapat dibuat lebih langsung.

### 1.4 Tujuan Penelitian

**LOCKED. Tidak diubah.**

### 1.5 Manfaat Penelitian

**Status: redundansi nyata.** Butir 1–3 pada dasarnya mengulang tujuan penelitian dalam tiga bentuk: kajian metode, pengaruh metode, dan perbandingan metode. Manfaat sebaiknya difokuskan pada keluaran/kontribusi yang dapat digunakan, bukan mengulang aktivitas penelitian. Kandidat: gabungkan butir 1–3 menjadi 1–2 butir yang lebih substantif; butir 4 dapat dipertahankan.

## 4. Audit BAB II

### 2.1 Biji Kopi Hijau, Cacat Fisik, dan Benda Asing

**Status: relatif bersih.** Empat paragraf memiliki fungsi berbeda: standar, variasi kelas dataset, dan implikasi untuk skala dataset. Kalimat penutup tiap paragraf dapat dipadatkan tetapi tidak ada redundansi besar.

### 2.2 Inspeksi Mutu Biji Kopi

**Status: pruning sedang.**

- Paragraf inspeksi manual mengulang masalah yang sudah dibangun pada 1.1. BAB II cukup menyatakan mekanisme inspeksi dan kebutuhan fitur visual secara lebih singkat.
- Paragraf De Oliveira relevan sebagai riwayat fitur warna/manual.
- Paragraf terakhir tentang perkembangan CNN/Transformer sangat generik dan berfungsi sebagai transisi, bukan landasan khusus. Ini kandidat kuat untuk dihapus atau digabung ke awal 2.3.

### 2.3 Deteksi Objek

**Status: mayoritas relevan.**

- Definisi deteksi, one-stage/two-stage, dan IoU layak dipertahankan.
- Paragraf klasifikasi–lokalisasi cukup panjang dibanding perannya. Karena penelitian tidak lagi membuat decomposition error formal, tiga sumber tentang task misalignment dapat diringkas menjadi satu penjelasan singkat atau satu-dua sumber paling relevan.

### 2.4 YOLO

**Status: pruning ringan–sedang.**

- Dua paragraf awal cukup sebagai sejarah ringkas.
- Paragraf YOLO pada kopi mengulang bukti yang telah muncul di 1.1, 2.6, dan akan muncul lagi pada Tabel 2.1. Di BAB II masih boleh ada, tetapi cukup satu paragraf yang lebih padat tanpa kembali membangun gap fine-grained.

### 2.5 YOLO26 dan Pembanding Arsitektur

**Status: ada methodology creep.**

- Dua paragraf awal tentang YOLO26 adalah landasan teori dan perlu.
- Paragraf “Pada penelitian ini, YOLO26n diposisikan...” mengulang 1.3 dan 3.3; kandidat dihapus atau dipersingkat satu kalimat.
- Paragraf RT-DETR terlalu defensif: “bukan menentukan mana yang lebih unggul”, “setelah konfigurasi ditetapkan”, dan “arah pengaruh”. BAB II cukup menjelaskan RT-DETRv3 dan alasan ia relevan sebagai arsitektur pembanding; aturan eksperimennya berada di 3.6.4.

### 2.6 Fine-Grained Object Detection

**Status: pruning sedang.**

- Definisi fine-grained perlu.
- Paragraf Xie kembali menyinggung klasifikasi–lokalisasi seperti 2.3. Salah satunya harus dipadatkan agar konsep tidak dijelaskan dua kali.
- Paragraf bukti kopi berguna tetapi mengulang 1.1 dan 2.4; cukup dibuat padat.
- Paragraf terakhir terlalu defensif dan masuk metodologi: “tidak semua kelas harus sama sulit” dan “penelitian tetap mengevaluasi seluruh kelas...”. Bagian pertama dapat menjadi satu kalimat konseptual; keputusan evaluasi seharusnya di BAB III.

### 2.7 Prapemrosesan Citra untuk Deteksi Objek

**Status: salah satu sumber redundansi terbesar BAB II.**

- Definisi preprocessing perlu.
- Syauqi: literatur relevan, tetapi alasan panjang mengapa CLAHE dipakai sebagai kontrol cukup satu klausa; detail kontrol di BAB III.
- Wavelet: penjelasan literatur relevan, tetapi daftar alasan mengapa tidak dijadikan baseline (“keluarga, level, subband, threshold, rekonstruksi”) terlalu defensif dan diulang lagi di BAB III.
- IA-YOLO/DENet memberi gagasan penting bahwa preprocessing dinilai dari tugas deteksi, bukan estetika visual; pertahankan.
- FE-YOLO relevan; pertahankan.
- Paragraf penutup mengulang desain penelitian (frequency-angular + CLAHE + fixed YOLO26n) yang sudah muncul di 2.5 dan akan muncul lagi di 2.8 dan 2.10. Kandidat dihapus.

### 2.8.1 DFT dan FFT

**Status: teori inti bersih, contoh literatur berulang.** Formula DFT/IDFT/FFT perlu. Paragraf terakhir yang kembali menyebut Yang, Li, dan Xu sebaiknya dipindahkan/diwakili oleh 2.8.4, karena fungsi yang sama akan dijelaskan lagi di sana.

### 2.8.2 Amplitudo dan Fase

**Status: teori inti bersih, paragraf contoh terlalu penuh.** Formula dan makna amplitudo/fase perlu. Contoh Yang/Li/Xu kembali mengulang 2.7 dan 2.8.1; cukup pertahankan satu contoh yang paling langsung jika diperlukan.

### 2.8.3 Representasi Radial dan Angular

**Status: penting tetapi terlalu defensif pada bagian akhir.**

- Definisi radial/angular dan dua referensi tekstur perlu.
- Uraian AFAB/AFAB-2 adalah dasar utama metode dan perlu.
- Paragraf “tidak mengadopsi keseluruhan LFDet...” terlalu panjang dan sebagian merupakan aturan metodologi. Cukup satu kalimat batas adaptasi. Daftar AFAB-1/CGFI/FTIF tidak perlu dijelaskan ulang di BAB III.
- Kalimat “transfer dari pesawat ke kopi adalah hipotesis, bukan efektivitas yang terbukti” benar secara ilmiah tetapi mengulang kehati-hatian pada 1.1 dan 2.8.4. Pilih satu lokasi saja.
- Klarifikasi bahwa “angular” bukan OBB berguna dan cukup disebut satu kali.

### 2.8.4 Pemrosesan Frekuensi pada Computer Vision

**Status: redundansi tinggi.**

- Paragraf input-space (Yang, Li, Xu) mengulang 2.7, 2.8.1, dan 2.8.2. Kandidat utama untuk dipangkas keras.
- Paragraf feature-space berguna untuk membedakan preprocessing input dengan operasi internal jaringan; pertahankan.
- Paragraf akhir tentang tidak adanya bukti “frequency signature” dan perlunya pengujian kembali mengulang gap yang sudah jelas. Dapat dipadatkan menjadi satu kalimat posisi penelitian.

### 2.9 Visualisasi Aktivasi Model

**Status: methodology creep + pengulangan internal.**

- BAB II cukup menjelaskan Grad-CAM, Eigen-CAM, dan sifat/keterbatasan keduanya.
- Paragraf pertama sudah menyebut B0–B3, C*=C0, layer, input, normalisasi, dan klaim kausal; sebagian besar merupakan BAB III.
- Paragraf ketiga dan keempat mengulang aturan kompatibilitas, target layer, dan normalisasi. Paragraf keempat hampir seluruhnya redundant dengan paragraf pertama dan 3.9.

### 2.10 Penelitian Terkait

**Status: tabel berguna; penutup berulang.**

- Tabel 2.1 adalah ringkasan yang sah meskipun sebagian isinya mengulang narasi sebelumnya.
- Paragraf akhir kembali merangkum “YOLO pada kopi + preprocessing/frequency di luar kopi + penelitian ini menghubungkan keduanya”, yaitu gap yang sudah dibangun berulang sejak BAB I. Dapat dipadatkan menjadi satu kalimat atau dihapus jika posisi penelitian sudah jelas dari tabel.

## 5. Audit BAB III

### 3.1 Rancangan Umum Penelitian

**Status: redundansi tinggi / kandidat pruning besar.**

- Paragraf pertama menyebut arsitektur YOLO26n tetap; konsep ini lebih tepat dijelaskan lengkap di 3.3.
- Paragraf kedua adalah daftar hampir seluruh isi BAB III, lalu Gambar 3.1 mengulang alur yang sama. Kandidat utama dipangkas menjadi satu-dua kalimat.
- Persamaan `YOLO26n(I)` versus `YOLO26n(P_FA(I))` secara matematis benar tetapi informasinya trivial dan tidak dipakai untuk derivasi berikutnya. Ini kandidat “equation theater”: terlihat formal tetapi tidak menambah definisi yang belum bisa dinyatakan satu kalimat. Disarankan dihapus kecuali pembimbing memang meminta formalisasi konseptual.
- Kalimat “tidak mengasumsikan preprocessing selalu meningkatkan kinerja” defensif dan tidak perlu dalam metodologi.

### 3.2.1 Sumber dan Karakteristik Dataset Primer

**Status: pruning ringan.**

- Target 21 kelas dan status jumlah final penting.
- Penjelasan bahwa jumlah kelas final ditetapkan setelah audit muncul lagi di 3.2.2; pilih satu lokasi utama.
- Daftar “lot/batch, pemasok, kebun, koperasi...” dan kalimat “tidak diasumsikan pada tahap proposal” terasa seperti respons terhadap audit, bukan naskah metodologi. Cukup nyatakan metadata sumber fisik dicatat sesuai sumber aktual.

### 3.2.2 Target Pengumpulan dan Pemeriksaan Kecukupan Data

**Status: banyak residu defensif dari proses audit sebelumnya.**

- Penjelasan target nominal 6.000–10.000 kemudian dilanjutkan dengan rentang teoritis 5.400–11.000 adalah pembelaan aritmetika yang tidak perlu masuk naskah. Cukup gunakan satu target operasional konsisten.
- Kalimat “bukan batas universal” benar tetapi defensif; cukup sebut “target operasional”.
- Angka studi Bahy/Tarekegn sudah dibahas di BAB II; tidak perlu diulang di metodologi kecuali benar-benar digunakan untuk menjustifikasi target secara eksplisit.
- Audit `N_obj`, `N_img`, `N_group` dan aturan jika kelas kurang data adalah substansi penting dan perlu dipertahankan.

### 3.2.3 Akuisisi Citra dan Anotasi

**Status: relatif bersih.**

- Definisi `group_id` sedikit panjang dan sebagian akan diulang di 3.2.4. Definisikan unit group di sini; 3.2.4 cukup merujuk.
- Bagian akuisisi, referensi skala, anotasi, dan validasi label relevan.

### 3.2.4 Pembagian Data dan Pencegahan Kebocoran

**Status: rumah utama aturan split; pertahankan tetapi kurangi pengulangan ke bagian lain.**

- Grouped split dan persamaan disjoint penting.
- Keterwakilan val/test dan hash relevan.
- Paragraf terakhir adalah tempat yang tepat untuk mengunci fungsi validation dan test. Setelah aturan ini ada, bagian 3.6 tidak perlu mengulang seluruh larangan tuning test berkali-kali.

### 3.2.5 Augmentasi Data

**Status: bersih dengan satu pengulangan kecil.** Train-only dan konfigurasi sama perlu. Kalimat tentang CLAHE/FA bukan augmentasi dapat dipadatkan; posisi frontend dijelaskan di 3.4.

### 3.3 Model Dasar YOLO26n

**Status: ini rumah utama gagasan “arsitektur tetap”.** Dua paragrafnya relevan. Setelah dipertahankan di sini, penjelasan penuh yang sama di 2.5 dan 3.1 harus dipangkas.

### 3.3.1 Kondisi Eksperimen Utama dan Pembanding

**Status: tabel dan fungsi perbandingan penting; paragraf terakhir terlalu defensif.**

- Tabel B0–B3 perlu.
- Konfigurasi CLAHE perlu.
- Perbandingan B2–B0, B3–B2, B3–B1 perlu.
- Penjelasan wavelet tidak menjadi baseline mengulang 1.3/2.7 dan dapat dihapus dari sini.
- RT-DETR cukup dirujuk ke 3.6.4.

### 3.4 Pembukaan metode frekuensi-angular

**Status: redundansi tinggi.**

- Dua paragraf pertama dapat digabung: satu kalimat sumber AFAB-2 dan satu kalimat yang memisahkan komponen adaptasi vs keputusan penelitian.
- Paragraf pipeline tensor [0,1], posisi frontend, dan output contract relevan tetapi dapat lebih ringkas.
- Paragraf “Secara umum, proses terdiri atas ...” hanya mengulang subjudul 3.4.1–3.4.6 dan Gambar 3.2; kandidat dihapus.

### 3.4.1–3.4.6

**Status: inti matematis perlu, tetapi beberapa kalimat pembelaan dapat dibuang.**

- 3.4.1: hapus kalimat generik “pemrosesan lokal digunakan agar respons ...”.
- 3.4.3: definisi DC cukup satu kali. Hilangkan pembelaan panjang bahwa DC tidak memiliki arah fisik; satu klausa cukup.
- 3.4.4: kalimat bahwa gamma diuji lagi pada sensitivity mengulang 3.5.6. Rentang tau dapat dipertahankan bila dianggap berguna.
- 3.4.5: sifat `w<=1` penting satu kali. Tidak perlu diulang di variasi C4.
- 3.4.6: “mengikuti retained AF2 operator yang digunakan dalam eksperimen repo” terlalu berbau provenance/debugging. Naskah formal cukup mendefinisikan kontrak operator. Frasa “bukan post-processing yang dituning terpisah” juga defensif dan dapat dihapus.

### 3.5 Pembukaan dan Tabel 3.2

**Status: tabel berguna; pembukaan dapat dipadatkan.** “Bukan faktorial” penting. Kalimat mewarisi kontrak no-clipping mengulang 3.4.6 dan tidak perlu.

### 3.5.1–3.5.5

**Status: pola repetitif.** Setiap subbagian sudah memiliki tujuan pada Tabel 3.2, tetapi banyak subbagian ditutup lagi dengan “Variasi ini menguji apakah...”. Pilih salah satu: tujuan di tabel atau kalimat penutup, tidak keduanya.

- 3.5.1: padding dapat merujuk kontrak umum, tidak perlu dijelaskan ulang penuh.
- 3.5.2: cukup “konvensi DC diwarisi”; tidak perlu mengulang penempatan secara panjang.
- 3.5.3: kalimat “bukan batas fisik optimal” defensif; cukup sebut pembagian sama lebar secara operasional.
- 3.5.4: `w_soft<=q<=1` mengulang sifat non-amplifikasi; dapat dihapus jika sudah established.
- 3.5.5: tujuan pengujian sudah ada di tabel; kalimat penutup dapat dihapus.

### 3.5.6 Analisis Sensitivitas

**Status: substansi penting.** Kandidat parameter dan OFAT perlu. Penjelasan ulang fungsi gamma/T dapat dipendekkan karena sudah didefinisikan sebelumnya. Aturan tidak menggabungkan best-of-sweeps tanpa evaluasi tambahan penting.

### 3.6 Pembukaan

**Status: satu kalimat cukup.** Kalimat pembuka hanya merangkum subjudul. Bisa dipertahankan sangat singkat.

### 3.6.1–3.6.3

**Status: pengulangan terbesar BAB III bersama aturan test.**

- Aturan seluruh kondisi dimulai dari `yolo26n.pt` dan tidak mewarisi checkpoint sebaiknya dinyatakan satu kali di pembukaan 3.6.
- 3.6.1 tidak perlu lagi menegaskan checkpoint B0dev bukan bobot C0–C5 jika aturan umum sudah ada.
- 3.6.2 tidak perlu mencetak ulang `C0→...→C5`; sudah ada di 3.5.
- 3.6.3 tidak perlu mengulang tabel B0–B3; tabel sudah ada di 3.3.1.
- “data uji tetap tertutup” tidak perlu diulang setelah 3.2.4 dan 3.6.5.
- `C*=C0` tidak perlu kembali dijelaskan di 3.6.3.
- Rumus paired delta dan aturan seed tetap penting.

### 3.6.4 RT-DETR opsional

**Status: ini rumah metodologis utama untuk RT-DETR.** Dapat dipadatkan menjadi satu paragraf tetapi substansinya tepat. Referensi detail opsional di 3.3/3.11/3.12 dapat dipersingkat.

### 3.6.5 Evaluasi Akhir pada Data Uji

**Status: redundansi tinggi dengan 3.2.4.**

- Proporsi 15%, grouped split, kecukupan 10 objek/5 citra, dan larangan overlap sudah dijelaskan di 3.2.4.
- 3.6.5 seharusnya fokus pada urutan final: konfigurasi/checkpoint/protokol dibekukan → checkpoint seed konfirmasi dievaluasi pada test → tidak ada retuning setelah test dibuka.
- Dengan demikian bagian ini dapat dipangkas kira-kira menjadi satu paragraf padat plus rujukan ke 3.2.4.

### 3.7 Konfigurasi Pelatihan

**Status: tabel baik; dua paragraf terlalu mirip catatan source-code audit.**

- Tabel konfigurasi perlu.
- Fairness kondisi dan aturan max epoch/batch masih metodologis.
- Penjelasan `fitness=[0,0,0,1]`, checkpoint internals, threshold 10.000 iterasi, `nbs=64`, dan contoh learning rate 0,0004 adalah detail verifikasi implementasi yang lebih cocok pada catatan audit/reproducibility, bukan narasi utama proposal.
- Naskah cukup menyatakan versi Ultralytics dikunci; `best.pt`/early stopping mengikuti mAP50–95 pada versi tersebut; `optimizer=Auto` ter-resolve ke AdamW untuk konfigurasi yang digunakan dan konfigurasi aktual dicatat.

### 3.8 Evaluasi Kinerja Deteksi

**Status: metrik inti baik; satu paragraf terlalu implementation-heavy.**

- AP/mAP, hard group, worst AP, paired delta, dan bootstrap adalah substansi.
- Paragraf yang menjelaskan `conf=0.001`, `end2end=True`, no-NMS, dan indeks confidence maksimum F1 terlalu dekat dengan dokumentasi source code. Proposal cukup mengatakan evaluator/version/parameter prediksi dibuat sama, max_det=500, dan P/R merupakan metrik sekunder dengan operating point bawaan evaluator yang dicatat.
- “tidak digunakan memilih C*” kembali mengulang aturan pemilihan; cukup satu rujukan.

### 3.9 Analisis Visual

**Status: pengulangan internal jelas.**

- Pembukaan sudah mengatakan visualisasi pendukung dan bukan bukti kausal.
- 3.9.1 mengulang bahwa perubahan visual bukan bukti lebih baik — dapat dihapus.
- 3.9.2 kembali mengatakan CAM hanya pendukung dan tidak menggantikan metrik — dapat dihapus.
- 3.9.3 tidak perlu mengulang `s_vis=123`; cukup “menggunakan seed pada 3.9.2”.
- Aturan seleksi contoh untuk mengurangi cherry-picking relevan.

### 3.10 Analisis Kesalahan

**Status: cukup baik dengan satu kalimat hipotetis yang tidak perlu.**

- AP per kelas, delta, confusion matrix, FP/FN relevan.
- Pengelompokan cue visual dapat dibuat lebih ringkas.
- Kalimat “Jika pemisahan jenis kesalahan kelak dihitung secara kuantitatif...” adalah defensif/hipotetis. Karena rancangan saat ini tidak menjadikannya metrik, cukup berhenti pada “analisis ini bersifat deskriptif”.

### 3.11 Evaluasi Efisiensi

**Status: relatif bersih.** Detail batas timing, end-to-end, CPU/GPU, median, dan memory punya fungsi reproducibility yang jelas. Hanya beberapa frasa dapat dipadatkan, tetapi bukan sumber utama AI residue.

### 3.12 Lingkungan Implementasi

**Status: bersih.** Reproducibility dan metadata run relevan. Kalimat RT-DETR terakhir dapat diringkas atau dirujuk ke 3.6.4, tetapi bukan masalah besar.

## 6. Bentuk “AI residue” yang harus dihapus secara sistematis

### A. Kalimat penutup yang hanya merangkum paragraf sebelumnya

Pola contoh:
- “Penelitian tersebut menunjukkan bahwa...”
- “Berdasarkan penelitian tersebut...”
- “Dengan demikian...”
- “Oleh karena itu...”

Tidak semuanya salah, tetapi banyak yang hanya mengulang isi dua kalimat sebelumnya tanpa menghasilkan inferensi baru. Setiap kemunculan perlu diuji: **apakah kalimat ini menambah klaim baru?** Jika tidak, hapus.

### B. Defensive stacking

Satu caveat ilmiah sering diulang dalam beberapa bentuk:
- “tidak diasumsikan...”
- “tidak dipaksakan...”
- “tidak dimaksudkan...”
- “bukan nilai optimal...”
- “bukan batas universal...”
- “tidak digunakan untuk...”
- “tidak dianggap sebagai bukti...”

Caveat yang mencegah overclaim tetap perlu, tetapi cukup dikunci satu kali pada lokasi yang relevan. Pengulangan caveat membuat naskah terdengar seperti sedang berdebat dengan reviewer yang belum bertanya.

### C. Methodology creep di BAB II

BAB II berkali-kali menjelaskan:
- kondisi B0–B3;
- alasan CLAHE menjadi control;
- wavelet tidak dipilih;
- YOLO26 tidak dimodifikasi;
- RT-DETR hanya opsional;
- C*=C0;
- metode CAM harus sama antar kondisi.

Sebagian besar detail tersebut harus berada di BAB III. BAB II cukup menjelaskan literatur dan hubungan konseptualnya dengan pertanyaan penelitian.

### D. Audit/debugging details masuk ke proposal

Contoh paling jelas:
- retained AF2 operator “yang digunakan dalam repo”;
- vektor fitness Ultralytics `[0,0,0,1]`;
- threshold 10.000 iterations untuk optimizer Auto;
- `nbs=64` dan contoh LR 0,0004;
- detail prefilter confidence 0,001 dan internal max-F1 operating point.

Semua informasi ini berguna untuk audit reproducibility, tetapi naskah utama hanya membutuhkan kontrak metodologis akhirnya. Detail provenance/source-code dapat tetap disimpan pada dokumen audit repo.

### E. Equation theater

Persamaan konseptual pada 3.1:

`Y_N = YOLO26n(I)` dan `Y_P = YOLO26n(P_FA(I))`

benar tetapi hampir tidak menambah informasi dibanding satu kalimat “frontend ditempatkan sebelum YOLO26n”. Berbeda dengan persamaan FFT, entropi, bobot, residual, dan metrik yang memang mendefinisikan metode. Persamaan 3.1 disarankan dihapus agar formalisme hanya digunakan ketika benar-benar mendefinisikan operator/prosedur.

## 7. Prioritas pruning

### Prioritas A — wajib sebelum proposal dianggap bersih

1. Pangkas 3.1 roadmap + equation theater.
2. Pangkas defensive arithmetic/target explanation di 3.2.2.
3. Hilangkan pengulangan aturan test dan checkpoint di 3.6.
4. Hilangkan tabel B0–B3 kedua di 3.6.3.
5. Ringkas 3.7 dari source-code audit menjadi kontrak pelatihan.
6. Ringkas paragraf validator internal di 3.8.
7. Hilangkan tiga pengulangan “visual bukan bukti” di 3.9.
8. Buang methodology creep berulang di 2.5, 2.7, 2.9.
9. Pangkas pengulangan contoh Fourier pada 2.7–2.8.4.
10. Pangkas penutup generatif 2.10.

### Prioritas B — penting untuk gaya akademik natural

1. Gabungkan bagian fine-grained yang berulang di 1.1/2.6.
2. Padatkan preprocessing umum vs pertanian di 1.1.
3. Kurangi paragraf generik 2.2.
4. Kurangi pengulangan klasifikasi–lokalisasi 2.3/2.6.
5. Definisikan DC sekali, bukan empat kali.
6. Hapus kalimat “variasi ini menguji...” jika tujuan sudah ada di Tabel 3.2.
7. Ringkas 3.6.4, 3.10, dan referensi RT-DETR di 3.11/3.12.
8. Gabungkan manfaat BAB I yang mengulang tujuan.

## 8. Bagian yang jangan dipangkas agresif

- 1.2 Rumusan Masalah — LOCKED.
- 1.4 Tujuan Penelitian — LOCKED.
- definisi dan formula DFT/amplitudo/fase/radial-angular yang menjadi landasan operator;
- formula C0–C5 yang benar-benar membedakan mekanisme;
- grouped split dan definisi `group_id`;
- aturan pemilihan C*;
- seed development vs confirmation;
- mAP50–95, AP_H, AP_worst, paired deltas, dan bootstrap;
- batas timing end-to-end;
- metadata reproducibility.

## 9. Kesimpulan audit

Proposal saat ini **kuat secara desain tetapi belum bersih secara editorial**. Residu generatif terutama berasal dari proses audit metodologi sebelumnya: setiap koreksi ditambahkan ke naskah sebagai kalimat pembelaan baru, sementara kalimat lama tidak selalu dipangkas. Akibatnya, naskah menyimpan banyak “guardrail” yang seharusnya cukup hidup di dokumen audit repo, bukan semuanya di teks proposal.

Revisi berikutnya sebaiknya bukan rewrite bebas. Lakukan **surgical pruning** dengan aturan:

1. jangan ubah substansi atau kontrak eksperimen;
2. jangan ubah 1.2 dan 1.4;
3. pindahkan detail provenance/debugging kembali ke dokumen audit;
4. setiap gagasan memiliki satu rumah utama;
5. referensi silang menggantikan penjelasan ulang;
6. caveat ilmiah dipertahankan hanya ketika mencegah overclaim yang nyata;
7. hapus kalimat ringkasan yang tidak menambah informasi.

Setelah pruning, lakukan audit ulang khusus untuk memastikan tidak ada informasi metodologis yang hilang dan build DOCX tetap lolos.