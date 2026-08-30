# Catatan Revisi Subbab 3.5 — Analisis Variasi Desain Prapemrosesan

Dokumen ini mencatat keputusan revisi untuk Subbab 3.5 sebelum perubahan diterapkan ke naskah utama BAB III.

## Prinsip Umum

- Rangkaian $C_0\rightarrow C_1\rightarrow C_2\rightarrow C_3\rightarrow C_4\rightarrow C_5$ tetap dipertahankan sebagai pengujian bertahap dan kumulatif.
- Setiap tahap menambahkan satu perubahan terhadap konfigurasi sebelumnya; rancangan ini bukan eksperimen faktorial seluruh kombinasi faktor.
- Redaksi dipadatkan agar tidak berulang menjelaskan bahwa konfigurasi bukan global optimum.
- Tabel variasi tetap digunakan, dengan kolom perubahan utama dipahami sebagai perubahan baru pada tahap tersebut.

## 3.5.1 Variasi Fungsi Jendela

- Pertahankan periodic square-root Hann window.
- Pertahankan normalized overlap-add sebagai konsekuensi teknis penggunaan window, bukan faktor eksperimen terpisah.
- Pertahankan replicate padding untuk konteks tepi pada $C_1$ dan konfigurasi berikutnya.
- Redaksi dipadatkan; motivasi cukup menyatakan bahwa variasi ini menguji pengaruh pengurangan diskontinuitas batas patch terhadap kinerja deteksi.
- Hindari klaim bahwa Hann pasti meningkatkan hasil.

## 3.5.2 Variasi Representasi Orientasi

- Pertahankan $\theta_o=\theta\bmod\pi$.
- Pertahankan 180 interval pada rentang $[0,\pi)$ sehingga resolusi nominal tetap $1^\circ$.
- Tujuan perbandingan $C_1$ ke $C_2$ adalah mengubah arah bertanda menjadi orientasi tak bertanda tanpa mengubah resolusi angular nominal.
- Komponen DC tetap tidak dimasukkan ke statistik orientasi, mengikuti keputusan revisi 3.4.3.
- Penjelasan mengenai bin kosong dipadatkan; bin tidak digabung secara bergantung-data.

## 3.5.3 Variasi Radial-Angular

- Pertahankan tiga pita radial dengan batas $1/3$ dan $2/3$ sebagai keputusan desain penelitian.
- Komponen DC ($\rho=0$) tidak dimasukkan ke statistik radial-angular, tetapi tetap dipertahankan untuk rekonstruksi.
- Normalisasi $p$ dan $q$ dilakukan terpisah pada tiap pita.
- Perjelas bahwa metode menguji seleksi angular pada wilayah radial berbeda, bukan membandingkan energi absolut antarpita.
- Istilah frekuensi rendah, menengah, dan tinggi diperlakukan sebagai kategori operasional berdasarkan radius ternormalisasi.

## 3.5.4 Variasi Ambang Lunak

- Pertahankan $w_{soft}(q,\tau)=q\,\sigma((q-\tau)/T)$.
- Pertahankan nilai awal $T=0{,}02$ sebagai keputusan desain penelitian.
- Tegaskan bahwa $T$ mengatur lebar transisi di sekitar ambang.
- Redaksi dipadatkan; tidak perlu mengulang penjelasan mengenai hard threshold.
- Karena $0\le w_{soft}\le q\le1$, tahap ini tetap tidak memperbesar amplitudo Fourier di atas nilai asal.

## 3.5.5 Variasi Panduan Luminansi

- Pertahankan luminansi BT.709: $Y=0{,}2126R+0{,}7152G+0{,}0722B$.
- Bobot/gate dihitung dari satu panduan luminansi dan diterapkan bersama pada ketiga kanal RGB; keluaran tetap RGB.
- Padatkan argumen preservasi rasio kanal lokal; tidak perlu dijelaskan dua kali.
- Kontrak skala/rentang output harus sama dengan konfigurasi lain dan mengikuti keputusan akhir pada revisi 3.4.6.

## 3.5.6 Analisis Sensitivitas Terbatas

- Pertahankan pengujian satu parameter pada satu waktu (OFAT).
- Kandidat tetap:
  - $m\in\{16,32,64\}$,
  - $\gamma\in\{0{,}05,0{,}10,0{,}15\}$,
  - $T\in\{0{,}01,0{,}02,0{,}05\}$ hanya jika konfigurasi struktur menggunakan ambang lunak.
- Ketika $m$ berubah, overlap tetap 50% sehingga $s=m/2$.
- Revisi konsep: perubahan $m$ memengaruhi sekaligus skala wilayah lokal dan resolusi/grid diskret representasi Fourier, bukan hanya ukuran wilayah lokal.
- Nilai terbaik dari pengujian parameter yang berbeda tidak boleh langsung digabung menjadi konfigurasi baru tanpa evaluasi tambahan.
- Hanya konfigurasi yang benar-benar telah dievaluasi yang dapat menjadi kandidat akhir.
- Detail aturan pemilihan $C^*$ dipusatkan di Subbab 3.6.2 agar tidak redundan dengan 3.5.6.

## Status

Catatan ini belum mengubah naskah utama BAB III. Revisi akan diterapkan setelah seluruh bagian metodologi selesai ditinjau agar struktur dan istilah dapat diperbarui secara konsisten.