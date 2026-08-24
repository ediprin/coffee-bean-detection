# BAB I — Bagian 1.2–1.5

## 1.2 Rumusan Masalah

Deteksi cacat fisik biji kopi dengan taksonomi yang rinci menghadapi dua persoalan yang saling berkaitan. Pertama, beberapa kategori cacat memiliki perbedaan visual yang halus sehingga kinerja model dapat berbeda cukup besar antarkelas. Kedua, sebagian besar penelitian yang ditinjau meningkatkan kemampuan diskriminasi melalui modifikasi di dalam model, sedangkan penggunaan preprocessing citra berbasis frekuensi-angular sebagai perlakuan pada input sebelum detector masih perlu diuji secara terkontrol pada domain biji kopi. AF2 sebagai kandidat preprocessing juga memiliki beberapa keputusan desain yang perlu dianalisis sebelum konfigurasi akhir dipilih. Berdasarkan kondisi tersebut, rumusan masalah penelitian ini adalah sebagai berikut.

1. Bagaimana pengaruh keputusan desain utama AF2 terhadap kinerja deteksi fine-grained cacat biji kopi, dan konfigurasi preprocessing frekuensi-angular seperti apa yang paling layak dipilih berdasarkan analisis terfaktor dan sensitivity analysis pada data pengembangan?
2. Apakah konfigurasi AF2 yang telah dipilih dapat meningkatkan kinerja YOLO26 dalam mendeteksi cacat biji kopi secara fine-grained dibandingkan YOLO26 tanpa preprocessing pada kondisi eksperimen yang dipasangkan?
3. Bagaimana pengaruh preprocessing frekuensi-angular terhadap kelas-kelas cacat yang memiliki kinerja rendah atau sulit dibedakan?
4. Apakah pola perubahan kinerja yang dihasilkan lebih konsisten dengan peningkatan diskriminasi kelas daripada peningkatan aksesibilitas proposal atau lokalisasi mentah?

Rumusan masalah keempat digunakan sebagai analisis diagnostik. Indikator aksesibilitas proposal tidak diperlakukan sebagai pengukuran lengkap kualitas regresi bounding box sehingga hasilnya tidak digunakan untuk membuat klaim kausal mengenai seluruh proses lokalisasi.

## 1.3 Batasan Masalah

Agar penelitian tetap terarah sesuai ruang lingkup tesis, batasan masalah ditetapkan sebagai berikut.

1. Penelitian berfokus pada object detection cacat fisik dan kategori objek yang tersedia pada dataset green coffee bean yang digunakan, dengan total 21 kelas. Penelitian tidak dimaksudkan sebagai implementasi lengkap seluruh prosedur grading mutu kopi berdasarkan bobot sampel atau nilai cacat.
2. Penelitian tidak menjadikan counting, open-set recognition, penilaian cita rasa, roasting quality, maupun segmentasi sebagai tujuan utama.
3. Detector yang digunakan adalah YOLO26n. Kontribusi utama yang diuji berada pada preprocessing citra di ruang input, bukan pada modifikasi backbone, neck, atau detection head YOLO26.
4. AF2 diperlakukan sebagai preprocessing frekuensi-angular tanpa learned preprocessing parameters. Istilah parameter-free tidak diartikan sebagai bebas biaya komputasi.
5. Analisis optimasi dibatasi pada keputusan desain AF2 yang dapat diuji secara terkontrol, yaitu konfigurasi referensi dan kandidat AF2WIN, AF2ORI, AF2POL, AF2SOFT, serta AF2LUM, disertai sensitivity analysis terbatas terhadap parameter yang relevan. Penelitian tidak melakukan broad module stacking.
6. Pemilihan konfigurasi AF2 dilakukan menggunakan data pengembangan. Data uji yang dikunci tidak digunakan untuk memilih konfigurasi, tuning, atau menentukan hyperparameter.
7. Eksperimen konfirmatori membandingkan native YOLO26n dengan AF2-YOLO26n menggunakan sumber pretrained model, pembagian data, inisialisasi target head, augmentation, training budget, dan seed yang dipasangkan. Seed yang direncanakan adalah 42, 123, dan 2026.
8. Evaluasi utama menggunakan Macro mAP50–95, dilengkapi mAP50, Bottom-3 mAP50–95, Worst-class mAP50–95, per-class AP, analisis kesalahan, diagnostik proposal/classification, serta evaluasi latency, throughput, parameter count, dan penggunaan memori. Hasil diagnostik dan visualisasi digunakan untuk mendukung interpretasi, bukan sebagai bukti kausal tunggal.

## 1.4 Tujuan Penelitian

Tujuan penelitian ini adalah sebagai berikut.

1. Menganalisis pengaruh keputusan desain utama AF2 dan memilih konfigurasi preprocessing frekuensi-angular yang layak melalui analisis terfaktor dan sensitivity analysis terbatas pada data pengembangan.
2. Mengevaluasi efektivitas konfigurasi AF2 terpilih pada YOLO26 melalui eksperimen konfirmatori yang dipasangkan dengan native YOLO26.
3. Menganalisis dampak preprocessing frekuensi-angular terhadap kelas-kelas dengan kinerja rendah melalui Bottom-3, Worst-class, per-class AP, dan paired error analysis.
4. Menganalisis pola perubahan kinerja melalui perbandingan indikator aksesibilitas proposal dengan indikator klasifikasi yang dikondisikan pada lokalisasi untuk menilai apakah perubahan lebih konsisten dengan peningkatan diskriminasi kelas.
5. Mengevaluasi trade-off antara kinerja deteksi dan efisiensi komputasi melalui parameter count, latency, throughput, dan penggunaan memori.

## 1.5 Manfaat Penelitian

Manfaat yang diharapkan dari penelitian ini adalah sebagai berikut.

1. Memberikan bukti empiris mengenai kelayakan preprocessing citra berbasis frekuensi-angular sebagai alternatif input-space untuk deteksi fine-grained cacat biji kopi menggunakan YOLO26.
2. Memberikan kerangka evaluasi yang lebih terkontrol untuk menganalisis pengaruh preprocessing, termasuk analisis konfigurasi AF2, kinerja agregat, per-class, Bottom-3, Worst-class, serta paired error transitions.
3. Memberikan informasi mengenai kelas-kelas cacat yang memperoleh manfaat atau mengalami regresi setelah preprocessing sehingga hasil penelitian tidak hanya bergantung pada satu nilai mAP agregat.
4. Memberikan informasi mengenai konsekuensi komputasi penggunaan AF2 sehingga peningkatan kinerja dapat dipertimbangkan bersama latency, throughput, dan penggunaan memori sebagai bahan acuan bagi penelitian lanjutan pada inspeksi visual biji kopi maupun objek pertanian dengan karakteristik fine-grained serupa.
