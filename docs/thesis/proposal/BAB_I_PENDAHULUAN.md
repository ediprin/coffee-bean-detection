# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang

Kualitas biji kopi merupakan salah satu faktor penting dalam proses pengendalian mutu karena kondisi fisik biji menjadi salah satu dasar dalam penilaian kualitas kopi. Biji kopi dapat memiliki berbagai jenis cacat fisik yang perlu diidentifikasi pada proses pemeriksaan dan pemilahan. Pada praktiknya, proses tersebut masih dapat dilakukan melalui pengamatan visual manusia sehingga konsistensi hasil pemeriksaan dapat dipengaruhi oleh pengalaman, kondisi pengamatan, pelatihan, dan beban kerja pemeriksa. García et al. (2019) menunjukkan bahwa proses seleksi manual biji kopi hijau dapat dipengaruhi oleh faktor operator dan waktu kerja. Kondisi tersebut mendorong pengembangan sistem berbasis *computer vision* untuk membantu proses identifikasi cacat secara lebih konsisten.

Perkembangan *deep learning* memungkinkan inspeksi biji kopi dilakukan secara otomatis melalui klasifikasi maupun *object detection*. Pada *object detection*, model tidak hanya menentukan kategori objek, tetapi juga menentukan lokasi objek pada citra. Salah satu keluarga model yang banyak digunakan adalah *You Only Look Once* (YOLO) karena mampu melakukan proses klasifikasi dan lokalisasi dalam satu sistem deteksi. Hong et al. (2026) menggunakan pengembangan YOLOv10 untuk mendeteksi tujuh kategori cacat biji kopi, sedangkan Gope et al. (2024) membandingkan beberapa varian YOLO pada deteksi biji kopi hijau. Penelitian tersebut menunjukkan bahwa keluarga YOLO dapat diterapkan pada deteksi cacat biji kopi. Namun, kinerja yang baik pada jumlah kategori yang relatif terbatas belum menunjukkan bahwa deteksi cacat dengan jumlah kategori yang lebih rinci telah terselesaikan.

Pada penelitian dengan jumlah kategori yang lebih banyak, perbedaan kemampuan model dalam mengenali setiap kelas menjadi lebih terlihat. Bahy dan Rifai (2026) menerapkan YOLOv5s pada 20 kategori fisik biji kopi dan menunjukkan adanya variasi kinerja antarkelas. Jundullah et al. (2026) juga menggunakan 20 kelas cacat dan kontaminan dengan YOLOv8s serta melaporkan bahwa kategori dengan karakteristik visual yang lebih khas cenderung lebih mudah dikenali dibandingkan beberapa kategori yang memiliki kemiripan visual. Hebert dan Alamsyah (2026) menemukan bahwa beberapa kategori cacat memiliki nilai AP yang jauh lebih rendah dibandingkan kategori lain. Kesulitan tersebut antara lain berkaitan dengan tanda cacat yang berukuran kecil, tekstur permukaan biji, dan kemiripan karakteristik visual antarjenis cacat.

Kondisi tersebut menunjukkan adanya karakteristik *fine-grained*, yaitu ketika beberapa kategori masih berasal dari jenis objek yang sama tetapi memiliki perbedaan visual yang relatif kecil. Kesiman et al. (2023) menunjukkan bahwa peningkatan jumlah kategori dari tiga kelas menjadi 17 kelas cacat menyebabkan penurunan akurasi yang besar pada model klasifikasi yang diuji. Hu et al. (2025) menyebutkan bahwa perbedaan visual yang halus antarjenis cacat menjadi salah satu tantangan dalam pengenalan cacat biji kopi. Temuan tersebut menunjukkan bahwa kemampuan model dalam membedakan karakteristik visual antarkelas menjadi aspek penting pada deteksi cacat biji kopi dengan kategori yang rinci.

Sejumlah penelitian telah mengembangkan metode untuk meningkatkan kemampuan tersebut. Hong et al. (2026) melakukan modifikasi pada proses ekstraksi dan pemrosesan fitur pada YOLOv10. Jiao et al. (2025) menggunakan *Swin Transformer*, *multistage feature fusion*, dan *selective attention* untuk meningkatkan representasi fitur pada proses grading dan identifikasi cacat biji kopi. Hu et al. (2025) menggunakan *Siamese network* untuk meningkatkan kemampuan model dalam membedakan kategori dengan karakteristik visual yang serupa. Berdasarkan literatur biji kopi yang ditinjau, sebagian besar pendekatan tersebut berfokus pada peningkatan representasi melalui perubahan komponen di dalam model.

Selain melalui modifikasi arsitektur, pengolahan citra sebelum masuk ke model deteksi juga telah digunakan pada berbagai penelitian *object detection*. Liu et al. (2022) melalui IA-YOLO menggunakan serangkaian proses pengolahan citra sebelum YOLO dan menyesuaikannya berdasarkan kebutuhan tugas deteksi. Qin et al. (2022) memanfaatkan pemisahan informasi frekuensi rendah dan tinggi sebelum citra hasil rekonstruksi diberikan kepada *detector*. Li et al. (2025) menggunakan pemrosesan pada domain Fourier terhadap komponen amplitudo dan fase sebelum citra diproses oleh YOLO. Penelitian-penelitian tersebut menunjukkan bahwa perubahan pada citra masukan dapat memengaruhi informasi yang diterima oleh model deteksi dan dapat dievaluasi berdasarkan kinerja deteksi yang dihasilkan.

Pendekatan serupa juga ditemukan pada objek pertanian. Syauqi et al. (2025) menggunakan *preprocessing* citra sebelum YOLOv8 pada deteksi cacat *white pepper*, sedangkan Chen et al. (2024) menggunakan kombinasi pengolahan berbasis *wavelet* dan peningkatan citra sebelum YOLOv8 untuk mendeteksi keretakan pada biji jagung. Penelitian tersebut memberikan contoh bahwa pengolahan pada citra masukan dapat ditempatkan sebagai bagian tersendiri sebelum proses deteksi. Meskipun demikian, hasil pada komoditas dan kondisi citra tersebut tidak dapat langsung digunakan untuk menyimpulkan efektivitas pendekatan yang sama pada cacat biji kopi.

Salah satu bentuk pengolahan citra yang dapat digunakan adalah pemrosesan pada domain frekuensi. Melalui transformasi Fourier, informasi citra dapat direpresentasikan berdasarkan komponen frekuensinya. Selain besarnya komponen frekuensi, distribusi energi spektral juga dapat dianalisis berdasarkan arah. Cao et al. (2019) menunjukkan penggunaan distribusi radial dan angular dari spektrum Fourier dalam analisis tekstur, sedangkan Zhang dan Tan (2003) menunjukkan bahwa distribusi orientasi spektral dapat digunakan sebagai informasi diskriminatif. Pada tugas *fine-grained object detection*, Xu et al. (2025) menggunakan pemrosesan frekuensi lokal dan informasi angular pada deteksi kategori pesawat yang memiliki perbedaan visual yang halus. Hasil tersebut menunjukkan bahwa informasi frekuensi dan arah layak dipertimbangkan sebagai salah satu bentuk representasi citra, tetapi efektivitasnya pada deteksi cacat biji kopi masih perlu diuji.

Berdasarkan uraian tersebut, penelitian ini mengusulkan penggunaan *preprocessing* citra berbasis frekuensi-angular sebelum proses deteksi menggunakan YOLO26. Citra akan dianalisis secara lokal pada domain Fourier sehingga informasi spektral dapat diproses berdasarkan distribusi arahnya, kemudian hasil pengolahan dikembalikan ke domain spasial sebelum diberikan kepada model deteksi. Pendekatan ini ditempatkan pada citra masukan sehingga arsitektur utama YOLO26 tidak menjadi objek modifikasi dalam penelitian.

Rancangan *preprocessing* frekuensi-angular memiliki beberapa faktor dan parameter yang dapat memengaruhi citra hasil pengolahan. Oleh karena itu, penelitian ini tidak hanya menerapkan metode tersebut, tetapi juga melakukan analisis dan optimasi terhadap rancangan *preprocessing* yang digunakan. Konfigurasi yang diperoleh selanjutnya akan dibandingkan dengan YOLO26 tanpa *preprocessing* untuk mengetahui pengaruhnya terhadap kinerja deteksi secara keseluruhan maupun terhadap kelas-kelas cacat yang lebih sulit dikenali. Selain kinerja deteksi, biaya komputasi yang ditimbulkan oleh proses *preprocessing* juga akan dievaluasi.

Berdasarkan latar belakang tersebut, penelitian ini dilakukan dengan judul **“Analisis dan Optimasi Preprocessing Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi.”**

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