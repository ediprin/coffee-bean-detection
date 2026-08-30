# Catatan Revisi Subbab 3.10 — Analisis Kesalahan dan Kinerja Per Kelas

Dokumen ini mencatat keputusan revisi untuk Subbab 3.10 sebelum perubahan diterapkan ke `BAB_III_METODOLOGI_PENELITIAN.md`.

## Keputusan utama

1. Analisis per kelas tetap menggunakan `AP_{c,50:95}`, matriks kebingungan, false positive, dan false negative.
2. Pada pengujian beberapa seed, tambahkan perubahan AP per kelas secara berpasangan terhadap baseline:

   $$
   \Delta AP_{c,s}=AP_{c,s}^{perlakuan}-AP_{c,s}^{B_0}.
   $$

   Rerata perubahan per kelas dihitung hanya pada seed konfirmasi yang telah dibekukan pada Tahap III.
3. Jangan membuat threshold tambahan untuk melabeli kelas sebagai "meningkat", "stabil", atau "menurun" tanpa dasar yang telah ditetapkan. Lebih aman melaporkan nilai perubahan AP secara kontinu.
4. Confusion matrix, FP, dan FN harus menggunakan prosedur evaluator, confidence threshold, IoU/matching, `max_det`, dan versi perangkat lunak yang sama pada seluruh kondisi.
5. Pengelompokan kelas berdasarkan karakteristik visual tetap bersifat deskriptif. Pemetaan kelas ke karakteristik visual utama harus ditetapkan sebelum hasil eksperimen diperiksa. Jika suatu kelas bergantung pada lebih dari satu cue visual, hal tersebut boleh dicatat secara eksplisit.
6. Hapus formulasi yang membuat salah kelas dan kesalahan lokalisasi seolah-olah sudah menjadi kategori kuantitatif formal hanya berdasarkan IoU 0,50. Pada citra multiobjek, aturan tersebut belum cukup karena memerlukan prosedur matching one-to-one dan definisi yang lebih lengkap.
7. Untuk proposal ini, salah kelas, ketidaktepatan lokalisasi, objek terlewat, dan prediksi tambahan dipertahankan sebagai analisis deskriptif terhadap contoh kesalahan, bukan metrik evaluasi baru. Jika kelak ingin dihitung secara kuantitatif, gunakan prosedur error-analysis yang terdefinisi dan bersumber, lalu bekukan protokolnya sebelum evaluasi akhir.

## Redaksi ringkas yang disarankan

> Analisis per kelas dilakukan menggunakan $AP_{c,50:95}$, matriks kebingungan, *false positive*, dan *false negative*. Pada pengujian beberapa seed, perubahan AP setiap kelas terhadap baseline juga dilaporkan secara berpasangan untuk melihat konsistensi pengaruh prapemrosesan.
>
> Kelas dapat dikelompokkan secara deskriptif berdasarkan karakteristik visual utama yang diperlukan oleh definisi label. Pemetaan kelas ke kelompok ditetapkan sebelum hasil eksperimen diperiksa dan tidak digunakan untuk memilih konfigurasi.
>
> Beberapa kesalahan juga ditinjau secara visual untuk mengidentifikasi indikasi kebingungan kelas, ketidaktepatan lokalisasi, objek terlewat, dan prediksi tambahan. Analisis ini hanya digunakan untuk interpretasi dan tidak membentuk metrik evaluasi baru.
