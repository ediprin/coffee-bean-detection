# BAB I — Bagian 1.2–1.5

## 1.2 Rumusan Masalah

Deteksi cacat fisik biji kopi dengan jumlah kategori yang rinci masih menghadapi kesulitan dalam membedakan kelas-kelas yang memiliki ciri visual yang subtil dan saling berdekatan. Penelitian terdahulu pada domain biji kopi umumnya meningkatkan kemampuan diskriminasi melalui perubahan di dalam model, sedangkan penggunaan preprocessing citra berbasis frekuensi-angular pada input detector belum banyak dikaji pada konteks deteksi fine-grained cacat biji kopi. Selain itu, preprocessing frekuensi-angular yang diusulkan memiliki beberapa keputusan desain yang dapat memengaruhi hasil sehingga konfigurasi yang digunakan perlu dianalisis sebelum dibandingkan dengan detector tanpa preprocessing. Oleh karena itu, diperlukan penelitian untuk menganalisis dan mengoptimasi preprocessing citra berbasis frekuensi-angular pada YOLO26 serta mengevaluasi pengaruhnya terhadap kinerja deteksi secara keseluruhan dan pada kelas-kelas cacat yang sulit dibedakan.

## 1.3 Batasan Masalah

Adapun batasan-batasan masalah yang dibuat pada penelitian ini agar fokus pada ruang penelitian yang telah ditentukan yaitu:

1. Penelitian berfokus pada object detection cacat fisik dan kategori objek yang tersedia pada dataset green coffee bean yang digunakan, dengan total 21 kelas.
2. Penelitian menggunakan YOLO26n sebagai detector utama.
3. Kontribusi utama penelitian berada pada preprocessing citra berbasis frekuensi-angular yang bekerja pada input sebelum citra diproses oleh YOLO26, bukan pada modifikasi backbone, neck, maupun detection head.
4. Preprocessing frekuensi-angular yang digunakan merupakan implementasi yang diadaptasi dari mekanisme AFAB-2 dan dalam penelitian ini disebut AF2.
5. Analisis optimasi dibatasi pada keputusan desain AF2 yang dapat diuji secara terkontrol, yaitu konfigurasi referensi dan kandidat AF2WIN, AF2ORI, AF2POL, AF2SOFT, serta AF2LUM, disertai sensitivity analysis terbatas terhadap parameter yang relevan.
6. Pemilihan konfigurasi preprocessing dilakukan menggunakan data pengembangan, sedangkan data uji yang dikunci tidak digunakan untuk model selection, tuning, atau penentuan hyperparameter.
7. Eksperimen konfirmatori membandingkan YOLO26n tanpa preprocessing dengan YOLO26n yang menggunakan konfigurasi preprocessing frekuensi-angular terpilih pada kondisi pretrained checkpoint, pembagian data, augmentation, training budget, target-head initialization, dan seed yang dipasangkan.
8. Evaluasi utama menggunakan Macro mAP50–95, mAP50, Bottom-3 mAP50–95, Worst-class mAP50–95, per-class AP, analisis kesalahan, diagnostic classification-versus-proposal accessibility, serta evaluasi latency, throughput, parameter count, dan penggunaan memori.
9. Penelitian tidak menjadikan counting, open-set recognition, grading cita rasa, roasting quality, maupun segmentasi sebagai tujuan utama.

## 1.4 Tujuan Penelitian

Penelitian ini bertujuan untuk menganalisis dan mengoptimasi preprocessing citra berbasis frekuensi-angular pada YOLO26 untuk deteksi fine-grained cacat biji kopi, memilih konfigurasi preprocessing yang layak melalui analisis terfaktor dan sensitivity analysis pada data pengembangan, serta mengevaluasi pengaruh konfigurasi terpilih terhadap kinerja deteksi secara keseluruhan, kinerja kelas-kelas cacat yang sulit dibedakan, pola perubahan diskriminasi kelas, dan efisiensi komputasi dibandingkan YOLO26 tanpa preprocessing.

## 1.5 Manfaat Penelitian

Adapun manfaat penelitian yang diharapkan pada penelitian ini adalah:

1. Memberikan bukti empiris mengenai penggunaan preprocessing citra berbasis frekuensi-angular pada YOLO26 untuk deteksi fine-grained cacat biji kopi.
2. Memberikan informasi mengenai pengaruh preprocessing terhadap kinerja agregat, kelas-kelas dengan performa rendah, serta kelas yang mengalami peningkatan maupun regresi setelah preprocessing.
3. Memberikan informasi mengenai trade-off kinerja deteksi dan efisiensi komputasi sehingga hasil penelitian dapat menjadi acuan bagi penelitian selanjutnya pada inspeksi visual biji kopi maupun objek pertanian dengan karakteristik fine-grained serupa.
