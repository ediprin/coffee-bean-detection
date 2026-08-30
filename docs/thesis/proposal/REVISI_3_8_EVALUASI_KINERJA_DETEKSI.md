# Catatan Revisi Subbab 3.8 — Evaluasi Kinerja Deteksi

Dokumen ini mencatat keputusan revisi untuk Subbab 3.8 sebelum perubahan diterapkan ke `BAB_III_METODOLOGI_PENELITIAN.md`.

## Keputusan Utama

1. `mAP50-95` tetap menjadi metrik utama penelitian, sedangkan `mAP50`, precision, dan recall menjadi metrik sekunder.
2. `max_det=500` tetap digunakan sama untuk seluruh kondisi.
3. Precision dan recall harus menggunakan prosedur evaluator dan operating point yang sama pada versi Ultralytics yang dikunci. Operating point aktual perlu dicatat dan tidak boleh dituning per kondisi.
4. Kelompok tiga kelas sulit ditetapkan satu kali dari model acuan pengembangan pada data validasi:

   $$
   \mathcal{H}=\operatorname{Bottom3}\left(AP_{c,50:95}^{val}(B_0^{dev})\right),
   $$

   dengan seed pengembangan 42. Setelah ditetapkan, `\mathcal{H}` dibekukan dan tidak dipilih ulang berdasarkan data uji.
5. Rerata AP kelas sulit tetap dihitung sebagai:

   $$
   AP_{\mathcal H}=\frac{1}{3}\sum_{c\in\mathcal H}AP_{c,50:95}.
   $$

6. `AP_worst` tetap dipertahankan sebagai indikator tambahan untuk mendeteksi penurunan ekstrem pada satu kelas, bukan sebagai dasar utama pemilihan konfigurasi.
7. Statistik multi-seed utama hanya menggunakan seed konfirmasi yang tidak dipakai pada tahap pengembangan. Seed 42 dilaporkan terpisah sebagai hasil pengembangan bila diperlukan.
8. Selain metrik per seed, laporkan perubahan berpasangan terhadap baseline pada seed yang sama:

   $$
   \Delta_s=M_{perlakuan,s}-M_{B_0,s},
   $$

   beserta `mean(Delta)` dan `SD(Delta)`.
9. Paired bootstrap pada evaluasi akhir menggunakan `group_id` sebagai unit resampling, bukan citra tunggal. Kelompok yang sama harus digunakan untuk seluruh kondisi yang dibandingkan pada setiap replikasi bootstrap.
10. Jika bootstrap digabungkan dengan beberapa seed konfirmasi, target utama adalah distribusi selisih rata-rata antar-seed pada setiap sampel bootstrap. Interval tersebut terutama merefleksikan ketidakpastian akibat sampel data uji, bukan seluruh variasi training seed.
11. Jangan menetapkan batas minimum jumlah group secara arbitrer. Paired bootstrap hanya digunakan bila jumlah group independen pada test memungkinkan resampling yang bermakna; jika terlalu sedikit, keterbatasan dilaporkan dan bootstrap tidak dijadikan dasar inferensi.

## Redaksi yang Perlu Dipadatkan

- Hindari pengulangan bahwa data uji tidak digunakan untuk memilih ulang kelas sulit.
- Jelaskan sekali bahwa precision/recall mengikuti evaluator yang sama dan operating point-nya dibekukan.
- Hubungkan langsung definisi `\mathcal H` dengan `B_0^{dev}` dan validation, bukan hanya menulis "model acuan" secara umum.
- Pada bagian multi-seed, utamakan laporan per-seed, rata-rata, simpangan baku, serta paired delta.

## Hal yang Perlu Diverifikasi Sebelum Naskah Final

- Operating point precision/recall pada Ultralytics versi yang dikunci.
- Implementasi praktis paired bootstrap berbasis `group_id` untuk metrik deteksi yang digunakan.
