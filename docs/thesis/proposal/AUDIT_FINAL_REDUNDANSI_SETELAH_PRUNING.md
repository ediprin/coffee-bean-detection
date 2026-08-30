# Audit Final Redundansi Setelah Pruning

Status: **PASS dengan pengulangan fungsional terbatas**

Dokumen yang diaudit:

- `BAB_I_PENDAHULUAN.md`
- `BAB_II_TINJAUAN_PUSTAKA.md`
- `BAB_III_METODOLOGI_PENELITIAN.md`

Audit ini merupakan tindak lanjut dari `AUDIT_FULL_REDUNDANSI_DEAI_PROPOSAL.md`. Dokumen audit sebelumnya merekam diagnosis sebelum pruning; status naskah aktif setelah pruning mengikuti dokumen ini dan naskah BAB I–III terbaru.

## 1. Hasil utama

Surgical pruning telah dilakukan tanpa mengubah Subbab 1.2 Rumusan Masalah dan 1.4 Tujuan Penelitian yang berstatus locked, serta tanpa mengubah kontrak metodologis utama pada BAB III.

Perubahan terhadap baseline sebelum pruning (`b8f4c61442f43e6f114338216079f2285a8d182a`) meliputi:

- BAB I: 22 baris ditambahkan dan 25 baris dihapus sebagai hasil pemadatan/redaksi ulang;
- BAB II: 36 baris ditambahkan dan 66 baris dihapus;
- BAB III: 110 baris ditambahkan dan 205 baris dihapus;
- tabel dua kolom B0–B3 yang mengulang Tabel 3.1 pada Tahap III dihapus;
- pipeline DOCX diselaraskan menjadi empat tabel formal yang memiliki fungsi berbeda.

Render proposal berubah dari sekitar 53 halaman sebelum pruning menjadi 39 halaman setelah pruning, dengan struktur utama, persamaan metode, tabel penelitian terkait, Gambar 3.1, Gambar 3.2, dan daftar pustaka tetap dipertahankan.

## 2. Pemeriksaan repetisi tekstual

Pemeriksaan terhadap paragraf substantif pada hasil DOCX tidak menemukan pasangan paragraf panjang yang mendekati duplikasi tekstual pada ambang kemiripan yang digunakan dalam audit internal.

Frasa generik/defensif yang sebelumnya sering muncul telah dibersihkan. Pada hasil audit badan BAB I–III, frasa berikut tidak lagi muncul:

- `Berdasarkan uraian tersebut`;
- `Berdasarkan penelitian tersebut`;
- `Dengan demikian`;
- `Oleh karena itu`;
- `Meskipun demikian`;
- `tidak diasumsikan`;
- `tidak dipaksakan`;
- `bukan batas`;
- `tidak dimaksudkan`;
- `tidak digunakan sebagai bukti`.

Kalimat transisi dan caveat tidak dihapus berdasarkan daftar kata semata; yang dihapus adalah kalimat yang tidak menambah fungsi argumentatif atau prosedural baru.

## 3. Pengulangan yang sengaja dipertahankan

Tidak semua kemunculan konsep yang sama merupakan redundansi. Pengulangan berikut dipertahankan karena berada pada fungsi akademik yang berbeda.

### Data uji

- Subbab 3.2.4 menetapkan keterwakilan dan status holdout data uji.
- Subbab 3.6.5 menetapkan waktu pembukaan data uji dan checkpoint yang dievaluasi.
- Subbab 3.8 menjelaskan bootstrap berbasis kelompok pada data uji.

Ketiganya tidak mengulang penjelasan yang sama.

### C* dapat sama dengan C0

- Subbab 3.3.1 mendefinisikan bahwa hasil pemilihan dapat menghasilkan `C*=C0`.
- Subbab 3.6.3 menjelaskan konsekuensi operasional: run B2/B3 yang identik tidak diulang.
- Subbab 3.9.3 menjelaskan konsekuensi visual: kondisi identik tidak ditampilkan dua kali.

### RT-DETRv3-R18

Kemunculan lintas bab dibatasi pada fungsi yang berbeda: batasan penelitian, landasan arsitektur, protokol analisis tambahan, dan pencatatan eksperimen bila analisis tersebut benar-benar dilakukan.

### Wavelet

Wavelet tetap muncul sebagai literatur prapemrosesan dan sebagai penegasan ruang lingkup bahwa metode tersebut bukan baseline utama. Tidak ada optimasi wavelet di BAB III.

## 4. Elemen yang dipangkas karena bersifat audit/debugging

Naskah utama tidak lagi memuat riwayat pembelaan teknis secara berlebihan, antara lain:

- perhitungan rentang ekstrem 5.400–11.000 objek setelah target nominal 6.000–10.000 dijelaskan;
- uraian provenance `retained AF2 operator` sebagai bahasa repo/debugging;
- penjelasan berulang bahwa semua kondisi tidak mewarisi checkpoint;
- tabel B0–B3 kedua pada Tahap III;
- rincian internal bobot fitness `[0,0,0,1]`, batas 10.000 iterasi optimizer Auto, `nbs=64`, dan contoh learning rate 0,0004;
- uraian validator tentang prefilter confidence dan implementasi end-to-end yang tidak diperlukan untuk menjelaskan metrik utama;
- caveat visualisasi yang diulang pada setiap subsubbab;
- persamaan konseptual di 3.1 yang hanya menuliskan kembali kalimat `preprocessing -> YOLO26n` tanpa menambah definisi matematis metode.

Detail-detail yang diperlukan untuk provenance atau audit implementasi tetap tersedia pada dokumen audit/resolusi teknis repo dan tidak perlu diulang pada naskah formal.

## 5. Validasi struktur dan render

- `Proposal Math Source Check` untuk naskah hasil pruning: **success**.
- `Build Proposal DOCX` setelah pipeline tabel diselaraskan: **success**.
- Render DOCX hasil akhir: **39 halaman**.
- Seluruh halaman telah diperiksa secara visual.
- Tidak ditemukan clipping, overlap, glyph rusak, atau tabel terpotong secara kritis.
- Tabel 2.1 tetap terbaca dan terbagi secara wajar pada beberapa halaman.
- Daftar tabel sekarang konsisten dengan empat tabel formal:
  - Tabel 2.1 Penelitian Terkait;
  - Tabel 3.1 Kondisi Utama Eksperimen;
  - Tabel 3.2 Variasi Desain Prapemrosesan;
  - Tabel 3.3 Konfigurasi Utama Pelatihan YOLO26n.

## 6. Putusan akhir editorial

BAB I–III tidak lagi menunjukkan pola pengulangan gagasan yang dominan atau defensive stacking seperti versi sebelum pruning. Pengulangan yang tersisa dapat ditelusuri ke fungsi akademik/prosedural yang berbeda dan tidak digunakan sekadar untuk memperpanjang naskah.

Prinsip editorial yang berlaku untuk revisi selanjutnya:

> Satu gagasan dijelaskan lengkap pada satu lokasi utama. Kemunculan berikutnya hanya diperbolehkan bila mempunyai fungsi baru, seperti batasan, definisi operasional, konsekuensi eksperimen, atau prosedur evaluasi.

Setiap penambahan teks berikutnya harus diperiksa terhadap prinsip ini agar redundansi tidak masuk kembali ke proposal.
