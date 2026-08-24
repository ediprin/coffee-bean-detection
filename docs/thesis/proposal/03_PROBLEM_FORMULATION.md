# BAB I — Bagian 1.2–1.5

## 1.2 Rumusan Masalah

Deteksi cacat biji kopi dengan jumlah kategori yang rinci memiliki tantangan karena beberapa jenis cacat mempunyai karakteristik visual yang relatif serupa sehingga kemampuan model dalam mengenali setiap kelas dapat berbeda. Dalam literatur biji kopi yang ditinjau, peningkatan kinerja umumnya dilakukan melalui modifikasi komponen di dalam model, sedangkan pengolahan citra berdasarkan informasi frekuensi dan arah sebelum proses deteksi masih perlu dikaji lebih lanjut pada kasus cacat biji kopi. Oleh karena itu, permasalahan dalam penelitian ini adalah bagaimana menerapkan dan mengoptimasi *preprocessing* citra berbasis frekuensi-angular pada YOLO26 serta menganalisis pengaruhnya terhadap kinerja deteksi *fine-grained* cacat biji kopi.

## 1.3 Batasan Masalah

Adapun batasan masalah pada penelitian ini agar penelitian tetap berada pada ruang lingkup yang telah ditentukan yaitu:

1. Penelitian berfokus pada *object detection* biji kopi hijau menggunakan dataset penelitian yang terdiri dari 21 kelas.
2. Model *object detection* yang digunakan adalah YOLO26 dengan varian YOLO26n.
3. Pengembangan metode difokuskan pada *preprocessing* citra berbasis frekuensi-angular sebelum citra diproses oleh YOLO26 dan tidak melakukan modifikasi pada *backbone*, *neck*, maupun *detection head* YOLO26.
4. Optimasi dilakukan terhadap faktor rancangan dan parameter utama pada *preprocessing* frekuensi-angular yang ditentukan dalam metodologi penelitian.
5. Kinerja model dievaluasi menggunakan metrik *object detection*, meliputi mAP50, mAP50–95, *precision*, *recall*, dan kinerja setiap kelas.
6. Efisiensi metode dievaluasi berdasarkan biaya komputasi yang ditimbulkan oleh *preprocessing*, seperti *latency*, *throughput*, dan penggunaan memori.
7. Penelitian tidak membahas penilaian cita rasa, proses *roasting*, maupun keseluruhan proses penentuan grade mutu kopi.

## 1.4 Tujuan Penelitian

Penelitian ini bertujuan untuk menganalisis dan mengoptimasi *preprocessing* citra berbasis frekuensi-angular pada YOLO26 untuk deteksi *fine-grained* cacat biji kopi serta mengevaluasi pengaruhnya terhadap kinerja deteksi dan biaya komputasi yang dihasilkan.

## 1.5 Manfaat Penelitian

Adapun manfaat yang diharapkan dari penelitian ini adalah:

1. Memberikan kajian mengenai penerapan *preprocessing* citra berbasis frekuensi-angular pada deteksi *fine-grained* cacat biji kopi menggunakan YOLO26.
2. Memberikan informasi mengenai pengaruh *preprocessing* frekuensi-angular terhadap kinerja deteksi secara keseluruhan maupun pada kelas cacat yang lebih sulit dikenali.
3. Menjadi referensi bagi penelitian selanjutnya dalam pengembangan sistem inspeksi mutu biji kopi berbasis *computer vision*, khususnya yang menggunakan pengolahan citra pada domain frekuensi.
