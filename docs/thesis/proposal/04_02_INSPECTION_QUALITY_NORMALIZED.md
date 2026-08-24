# 2.2 Inspeksi Mutu Biji Kopi: Metode Konvensional dan Tantangannya

Status: **normalized replacement for §2.2**. This module should replace the older §2.2 block in `04_LITERATURE_REVIEW.md` when the chapter is assembled/exported.

Tujuan normalisasi: menghapus ketergantungan berulang pada `COF-07` Kesiman dan `COF-08` Arwatchananukul di §2.2. Kedua paper tersebut dipertahankan pada fungsi yang lebih kuat, yaitu taxonomy (§2.1) dan fine-grained diagnosis (§2.6).

---

Inspeksi mutu green coffee secara konvensional banyak bergantung pada pengamatan visual tenaga manusia. García, Candelo-Becerra, dan Hoyos menjelaskan bahwa pemilihan manual dilakukan oleh personel yang mengamati karakteristik biji untuk menentukan biji yang dianggap baik. Mereka mencatat bahwa hasil pemilihan dapat menjadi tidak seragam akibat jam kerja panjang, kurangnya pelatihan, dan faktor operator; prosedur manual juga memerlukan waktu yang besar ketika jumlah biji yang harus diperiksa meningkat [COF-17]. Studi tersebut juga membedakan pemilihan manual dari mechanical sorting berbasis ukuran, yang tidak mampu mengevaluasi seluruh karakteristik tampilan fisik biji [COF-17].

Kebutuhan otomatisasi tersebut telah mendorong penggunaan machine vision jauh sebelum dominasi deep learning. García et al. membangun sistem computer vision yang terdiri atas image acquisition, preprocessing/processing, feature extraction, dan klasifikasi KNN. Sistem mereka menggunakan karakteristik warna, ukuran, morfologi, dan bentuk untuk inspeksi kualitas dan defect green coffee [COF-17]. De Oliveira et al. menggunakan akuisisi citra terkontrol, kalibrasi warna, dan representasi CIE L*a*b* bersama computational-intelligence classifiers untuk klasifikasi biji kopi [COF-10]. Kedua studi tersebut menunjukkan pola umum pendekatan klasik: kondisi akuisisi dan feature engineering harus dirancang secara eksplisit sebelum klasifikasi.

Literatur klasik juga sudah memperlihatkan bahwa karakteristik defect tidak selalu mempunyai tingkat keterpisahan yang sama. García et al. melaporkan bahwa black bean relatif mudah dikenali karena fitur warnanya lebih distinctive, sedangkan sour bean lebih sulit karena rentang warnanya bervariasi dan dapat menghasilkan feature similarity dengan kelas lain [COF-17]. Temuan ini penting sebagai sejarah masalah representasi visual, tetapi tidak digunakan untuk mengklaim bahwa sistem KNN 2019 setara dengan fine-grained object detector modern.

Perkembangan berikutnya menggeser representasi dari handcrafted features menuju CNN, Transformer, dan deployment edge. Tinjauan Motta et al. memetakan berbagai penerapan machine learning dan computer vision pada inspeksi/kualitas kopi, sedangkan Muchtar et al. menunjukkan contoh modern penggunaan deep-learning models dan perangkat edge untuk otomatisasi penilaian green coffee [REV-01][COF-14]. Perubahan ini mengurangi ketergantungan pada feature engineering manual, tetapi tidak secara otomatis menghilangkan masalah ketika kelas memiliki perbedaan visual yang kecil.

Dengan demikian, tantangan inspeksi kopi dapat dipisahkan menjadi dua lapis. Lapisan pertama adalah kebutuhan otomasi karena inspeksi manual memerlukan waktu dan dipengaruhi faktor operator [COF-17]. Lapisan kedua adalah kebutuhan representasi yang mampu membedakan kategori yang visualnya berdekatan, yang dibahas secara khusus pada Subbab 2.6 menggunakan literatur fine-grained dan multi-class coffee yang lebih relevan. Pemisahan ini mencegah paper taxonomy/fine-grained dipakai berulang sebagai sumber umum untuk setiap bagian Bab II.

## Source-role note

Primary roles pada subbab ini:

```text
COF-17 García 2019 -> manual/mechanical limitations + classical machine vision + early feature-similarity evidence
COF-10 de Oliveira 2016 -> controlled handcrafted/computational-intelligence precedent
REV-01 Motta review -> literature landscape only
COF-14 Muchtar 2025 -> modern deep-learning/edge transition
```

`COF-07` dan `COF-08` sengaja tidak digunakan di sini supaya peran utamanya tetap:

```text
COF-07 -> §2.1 taxonomy + §2.6 coarse-to-fine difficulty
COF-08 -> §2.1 taxonomy + §2.6 17-class/unseen behavior
```
