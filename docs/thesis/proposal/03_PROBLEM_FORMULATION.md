# BAB I — Bagian 1.2–1.5

## 1.2 Rumusan Masalah

Deteksi cacat fisik biji kopi dengan taksonomi yang rinci masih menghadapi perbedaan kinerja antarkelas karena beberapa kategori memiliki ciri visual yang subtil dan saling berdekatan. Sebagian besar penelitian yang ditinjau meningkatkan kemampuan diskriminasi melalui modifikasi di dalam model, sedangkan pengaruh preprocessing citra berbasis frekuensi-angular pada input YOLO26 belum diketahui pada domain biji kopi. Di sisi lain, AF2 memiliki beberapa keputusan desain yang dapat memengaruhi hasil dan perlu dianalisis sebelum digunakan sebagai konfigurasi akhir. Oleh karena itu, diperlukan penelitian untuk menganalisis dan mengoptimasi konfigurasi AF2, menguji pengaruhnya terhadap kinerja deteksi fine-grained dan kelas-kelas yang sulit, serta menilai apakah pola perubahan kinerja lebih konsisten dengan peningkatan diskriminasi kelas dibandingkan peningkatan aksesibilitas proposal atau lokalisasi mentah.

## 1.3 Batasan Masalah

Adapun batasan-batasan masalah yang dibuat pada penelitian ini agar fokus pada ruang penelitian yang telah ditentukan yaitu:

1. Penelitian berfokus pada object detection cacat fisik dan kategori objek yang tersedia pada dataset green coffee bean yang digunakan, dengan total 21 kelas.
2. Penelitian menggunakan YOLO26n sebagai detector utama dan tidak melakukan modifikasi pada backbone, neck, maupun detection head.
3. Kontribusi utama yang diuji adalah preprocessing citra berbasis frekuensi-angular AF2 pada ruang input sebelum citra diproses oleh YOLO26.
4. Analisis optimasi AF2 dibatasi pada konfigurasi referensi dan kandidat AF2WIN, AF2ORI, AF2POL, AF2SOFT, serta AF2LUM dengan pendekatan satu faktor pada satu waktu, disertai sensitivity analysis terbatas terhadap parameter yang relevan.
5. Pemilihan konfigurasi AF2 dilakukan menggunakan data pengembangan. Data uji yang dikunci tidak digunakan untuk model selection, tuning, maupun penentuan hyperparameter.
6. Eksperimen konfirmatori membandingkan native YOLO26n dengan AF2-YOLO26n menggunakan pretrained checkpoint, pembagian data, augmentation, training budget, target-head initialization, dan seed yang dipasangkan.
7. Evaluasi utama menggunakan Macro mAP50–95, mAP50, Bottom-3 mAP50–95, Worst-class mAP50–95, per-class AP, analisis kesalahan, diagnostik proposal/classification, serta evaluasi latency, throughput, parameter count, dan penggunaan memori.
8. Penelitian tidak menjadikan counting, open-set recognition, grading cita rasa, roasting quality, maupun segmentasi sebagai tujuan utama.

## 1.4 Tujuan Penelitian

Penelitian ini bertujuan untuk menganalisis dan mengoptimasi preprocessing citra berbasis frekuensi-angular AF2 pada YOLO26 untuk deteksi fine-grained cacat biji kopi, memilih konfigurasi AF2 melalui analisis terfaktor dan sensitivity analysis pada data pengembangan, membandingkan kinerja AF2-YOLO26 dengan native YOLO26 pada eksperimen konfirmatori yang dipasangkan, menganalisis pengaruhnya terhadap kelas-kelas dengan kinerja rendah, serta mengevaluasi pola perubahan diskriminasi kelas, aksesibilitas proposal, dan trade-off antara kinerja deteksi dan efisiensi komputasi.

## 1.5 Manfaat Penelitian

Adapun manfaat penelitian yang diharapkan pada penelitian ini adalah:

1. Memberikan bukti empiris mengenai penggunaan preprocessing citra berbasis frekuensi-angular AF2 pada YOLO26 untuk deteksi fine-grained cacat biji kopi.
2. Memberikan informasi mengenai pengaruh AF2 terhadap kinerja agregat, kelas-kelas dengan performa rendah, serta kelas yang mengalami peningkatan maupun regresi setelah preprocessing.
3. Memberikan informasi mengenai trade-off kinerja deteksi dan efisiensi komputasi sehingga hasil penelitian dapat menjadi acuan bagi penelitian selanjutnya pada inspeksi visual biji kopi maupun objek pertanian dengan karakteristik fine-grained serupa.
