# BAB I — Bagian 1.2–1.6

## 1.2 Rumusan Masalah

Deteksi cacat fisik biji kopi dengan taksonomi yang rinci masih menghadapi perbedaan kinerja antarkelas karena beberapa kategori memiliki ciri visual yang subtil dan saling berdekatan. Sebagian besar penelitian yang ditinjau meningkatkan kemampuan diskriminasi melalui modifikasi di dalam model, sedangkan pengaruh preprocessing citra berbasis frekuensi-angular pada input YOLO26 belum diketahui pada domain biji kopi. Di sisi lain, AF2 memiliki beberapa keputusan desain yang dapat memengaruhi hasil dan perlu dianalisis sebelum digunakan sebagai konfigurasi akhir. Oleh karena itu, diperlukan penelitian yang secara sistematis menganalisis dan mengoptimasi preprocessing frekuensi-angular AF2, kemudian mengevaluasi konfigurasi terpilih pada YOLO26 dibandingkan native YOLO26 berdasarkan performa agregat, performa kelas-kelas yang sulit, pola perubahan diskriminasi kelas dan aksesibilitas proposal atau lokalisasi mentah, serta trade-off antara kinerja deteksi dan efisiensi komputasi.

## 1.3 Tujuan Penelitian

Tujuan dari penelitian ini adalah:

1. Merancang dan mengimplementasikan preprocessing citra berbasis frekuensi-angular AF2 sebagai frontend sebelum YOLO26 untuk deteksi fine-grained cacat biji kopi.
2. Menganalisis dan mengoptimasi keputusan desain AF2 melalui perbandingan kandidat struktural secara satu faktor pada satu waktu dan sensitivity analysis terbatas untuk memperoleh konfigurasi AF2 yang paling layak digunakan pada eksperimen konfirmatori.
3. Mengevaluasi dan membandingkan performa native YOLO26 dan AF2-YOLO26 berdasarkan Macro mAP50–95, mAP50, mAP50–95, Bottom-3, Worst-class, dan per-class AP pada kondisi pelatihan yang dipasangkan.
4. Menganalisis pengaruh AF2 terhadap kelas-kelas cacat yang sulit dibedakan melalui diagnostic discrimination-versus-proposal accessibility, visualisasi transformasi atau aktivasi yang kompatibel, serta analisis kesalahan rescue dan regression.
5. Mengevaluasi trade-off performa dan efisiensi AF2-YOLO26 berdasarkan parameter count, latency, throughput, dan penggunaan memori pada lingkungan pengujian yang sama.

## 1.4 Batasan Penelitian

Batasan dari penelitian ini adalah:

1. Penelitian berfokus pada object detection cacat fisik dan kategori objek yang tersedia pada dataset green coffee bean yang digunakan, dengan total 21 kelas.
2. Penelitian menggunakan YOLO26n sebagai detector utama dan tidak melakukan modifikasi pada backbone, neck, maupun detection head.
3. Kontribusi utama yang diuji adalah preprocessing citra berbasis frekuensi-angular AF2 pada ruang input sebelum citra diproses oleh YOLO26.
4. Analisis optimasi AF2 dibatasi pada konfigurasi referensi dan kandidat AF2WIN, AF2ORI, AF2POL, AF2SOFT, serta AF2LUM dengan pendekatan satu faktor pada satu waktu, disertai sensitivity analysis terbatas terhadap parameter yang relevan.
5. Pemilihan konfigurasi AF2 dilakukan menggunakan data pengembangan. Data uji yang dikunci tidak digunakan untuk model selection, tuning, maupun penentuan hyperparameter.
6. Eksperimen konfirmatori membandingkan native YOLO26n dengan AF2-YOLO26n menggunakan pretrained checkpoint, pembagian data, augmentation, training budget, target-head initialization, dan seed yang dipasangkan.
7. Evaluasi utama menggunakan Macro mAP50–95, mAP50, mAP50–95, Bottom-3 mAP50–95, Worst-class mAP50–95, per-class AP, analisis kesalahan, diagnostik proposal/classification, serta evaluasi latency, throughput, parameter count, dan penggunaan memori.
8. Penelitian tidak menjadikan counting, open-set recognition, grading cita rasa, roasting quality, maupun segmentasi sebagai tujuan utama.

## 1.5 Manfaat Penelitian

Penelitian ini diharapkan dapat memberikan beberapa manfaat sebagai berikut:

1. Memberikan bukti empiris mengenai penggunaan preprocessing citra berbasis frekuensi-angular AF2 pada YOLO26 untuk deteksi fine-grained cacat biji kopi.
2. Menghasilkan dasar berbasis eksperimen dalam memilih konfigurasi AF2 melalui analisis terfaktor, sensitivity analysis, dan evaluasi lower-tail classes.
3. Memperkaya literatur mengenai penggunaan input-space frequency-aware preprocessing pada object detection pertanian, khususnya pada domain green coffee bean dengan taksonomi multi-class yang rinci.
4. Memberikan informasi mengenai kelas-kelas yang memperoleh manfaat maupun mengalami regresi melalui per-class AP, confusion/error analysis, serta rescue-regression transition analysis.
5. Memberikan gambaran trade-off akurasi dan efisiensi AF2-YOLO26 sebagai pertimbangan bagi pengembangan penelitian selanjutnya.

## 1.6 Sistematika Penulisan

1. **Bab 1 – Pendahuluan.** Bab ini berisi latar belakang penelitian, rumusan masalah, tujuan penelitian, batasan penelitian, manfaat penelitian, serta sistematika penulisan.
2. **Bab 2 – Tinjauan Pustaka.** Bab ini membahas biji kopi hijau dan cacat fisik biji kopi, inspeksi mutu biji kopi, object detection, YOLO, YOLO26, fine-grained object detection, preprocessing citra untuk object detection, representasi citra pada domain frekuensi, serta penelitian terdahulu yang relevan.
3. **Bab 3 – Metode Penelitian.** Bab ini menjelaskan dataset penelitian, baseline YOLO26, arsitektur AF2-YOLO26, preprocessing frekuensi-angular AF2, strategi analisis dan optimasi AF2, rancangan eksperimen konfirmatori, konfigurasi pelatihan, metrik evaluasi, analisis mekanisme, visualisasi, analisis kesalahan, dan evaluasi efisiensi.
4. **Bab 4 – Hasil dan Pembahasan.** Bab ini menyajikan hasil optimasi AF2, perbandingan native YOLO26 dan AF2-YOLO26, analisis per kelas dan lower-tail, visualisasi, analisis kesalahan, mechanism diagnostics, serta pembahasan trade-off performa dan efisiensi.
5. **Bab 5 – Kesimpulan dan Saran.** Bab ini berisi kesimpulan berdasarkan hasil penelitian dan saran untuk pengembangan penelitian selanjutnya.
