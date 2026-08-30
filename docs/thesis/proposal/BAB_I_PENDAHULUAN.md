# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang

Kualitas biji kopi merupakan salah satu faktor penting dalam pengendalian mutu karena kondisi fisik biji menjadi dasar dalam proses pemeriksaan dan pemilahan. Pemeriksaan tersebut masih dapat dilakukan melalui pengamatan visual manusia sehingga konsistensinya dipengaruhi oleh pengalaman, kondisi pengamatan, pelatihan, dan beban kerja pemeriksa. García et al. (2019) menunjukkan bahwa seleksi manual biji kopi hijau dapat dipengaruhi oleh faktor operator dan waktu kerja. Kondisi ini mendorong pengembangan sistem berbasis *computer vision* untuk membantu identifikasi cacat secara lebih konsisten.

Perkembangan *deep learning* memungkinkan inspeksi biji kopi dilakukan melalui klasifikasi maupun *object detection*. Pada *object detection*, model menentukan kategori sekaligus lokasi objek pada citra. Hong et al. (2026) menggunakan pengembangan YOLOv10 untuk mendeteksi tujuh kategori cacat biji kopi, sedangkan Gope et al. (2024) membandingkan beberapa varian YOLO pada deteksi biji kopi hijau. Hasil tersebut menunjukkan kelayakan keluarga YOLO pada domain biji kopi, tetapi belum mewakili deteksi dengan susunan kategori yang lebih rinci.

Pada jumlah kategori yang lebih besar, perbedaan kinerja antarkelas menjadi lebih terlihat. Bahy dan Rifai (2026) menerapkan YOLOv5s pada 20 kategori fisik biji kopi dan menunjukkan variasi kinerja antarkelas. Jundullah et al. (2026) melaporkan bahwa kategori dengan karakteristik visual khas cenderung lebih mudah dikenali dibandingkan kategori yang mirip secara visual, sedangkan Hebert dan Alamsyah (2026) menemukan beberapa kategori dengan AP jauh lebih rendah daripada kategori lain. Kesulitan tersebut antara lain berkaitan dengan tanda cacat berukuran kecil, tekstur permukaan, dan kemiripan visual antarjenis cacat.

Karakteristik tersebut berkaitan dengan *fine-grained recognition*, yaitu pembedaan kategori yang berasal dari jenis objek serupa tetapi memiliki perbedaan visual kecil. Kesiman et al. (2023) menunjukkan penurunan akurasi yang besar ketika jumlah kategori klasifikasi diperluas dari tiga menjadi 17 kelas cacat, sementara Hu et al. (2025) menempatkan kemiripan visual antarjenis cacat sebagai salah satu tantangan utama. Tidak seluruh kelas harus memiliki tingkat kesulitan yang sama; perhatian *fine-grained* terutama diperlukan pada kelas dengan ciri visual yang berdekatan.

Sejumlah penelitian kopi meningkatkan kemampuan diskriminasi melalui komponen di dalam model. Hong et al. (2026) memodifikasi ekstraksi dan pemrosesan fitur pada YOLOv10, Jiao et al. (2025) menggunakan *Swin Transformer*, *multistage feature fusion*, dan *selective attention*, sedangkan Hu et al. (2025) menggunakan *Siamese network*. Arah lain adalah mengubah representasi citra sebelum model deteksi. Liu et al. (2022) melalui IA-YOLO menggunakan pengolahan citra adaptif sebelum YOLO, Qin et al. (2022) memisahkan informasi frekuensi rendah dan tinggi sebelum rekonstruksi, dan Li et al. (2025) memproses amplitudo serta fase pada domain Fourier sebelum deteksi.

Prapemrosesan sebelum detektor juga digunakan pada objek pertanian. Syauqi et al. (2025) menerapkan rangkaian prapemrosesan sebelum YOLOv8 untuk deteksi cacat *white pepper*, sedangkan Chen et al. (2024) menggunakan pengolahan berbasis *wavelet* dan peningkatan citra sebelum YOLOv8 untuk mendeteksi keretakan biji jagung. Hasil pada komoditas tersebut menunjukkan kelayakan pendekatan berbasis citra masukan, tetapi efektivitasnya pada cacat biji kopi perlu diuji tersendiri.

Pada domain frekuensi, transformasi Fourier merepresentasikan citra melalui komponen spektral yang juga dapat dianalisis berdasarkan arah. Cao et al. (2019) menggunakan distribusi radial dan angular spektrum Fourier dalam analisis tekstur, sedangkan Zhang dan Tan (2003) menunjukkan bahwa distribusi orientasi spektral dapat menjadi ciri diskriminatif. Pada *fine-grained object detection*, Xu et al. (2025) menggunakan pemrosesan frekuensi lokal dan informasi angular untuk mendeteksi kategori pesawat yang memiliki perbedaan visual halus.

Berdasarkan landasan tersebut, penelitian ini menguji prapemrosesan citra berbasis frekuensi-angular sebelum YOLO26n. Citra diproses secara lokal pada domain Fourier dan direkonstruksi kembali ke domain spasial sebelum deteksi. Variasi desain prapemrosesan dianalisis dengan YOLO26n tanpa prapemrosesan sebagai acuan dan CLAHE sebagai pembanding peningkatan kontras lokal, kemudian dievaluasi berdasarkan kinerja deteksi dan biaya komputasi.

## 1.2 Rumusan Masalah

Deteksi cacat biji kopi dengan jumlah kategori yang rinci memiliki tantangan karena beberapa jenis cacat mempunyai karakteristik visual yang relatif serupa sehingga kemampuan model dalam mengenali setiap kelas dapat berbeda. Dalam literatur biji kopi yang ditinjau, peningkatan kinerja umumnya dilakukan melalui modifikasi komponen di dalam model, sedangkan pengolahan citra berdasarkan informasi frekuensi dan arah sebelum proses deteksi masih perlu dikaji lebih lanjut pada kasus cacat biji kopi. Untuk itu, diperlukan penelitian mengenai penerapan prapemrosesan citra berbasis frekuensi-angular pada YOLO26n, variasi desain prapemrosesan tersebut, serta pengaruhnya terhadap kinerja deteksi *fine-grained* cacat biji kopi.

## 1.3 Batasan Masalah

Batasan penelitian ini adalah:

1. Penelitian berfokus pada *object detection* biji kopi hijau menggunakan dataset primer. Daftar kelas awal menargetkan 20 kategori cacat fisik dan benda asing yang digunakan dalam SNI 2907:2008 serta satu kelas biji normal; jumlah kelas akhir ditetapkan setelah pemeriksaan kecukupan data.
2. Model utama adalah YOLO26n. RT-DETRv3-R18 hanya digunakan sebagai analisis tambahan apabila sumber daya memungkinkan.
3. Pengembangan metode difokuskan pada prapemrosesan citra berbasis frekuensi-angular tanpa memodifikasi *backbone*, *neck*, atau *detection head* YOLO26n pada eksperimen utama.
4. Optimasi dibatasi pada variasi desain dan parameter prapemrosesan yang ditetapkan dalam metodologi, bukan pencarian *global optimum* seluruh konfigurasi.
5. YOLO26n tanpa prapemrosesan digunakan sebagai acuan dan CLAHE sebagai pembanding peningkatan kontras lokal. Wavelet tidak menjadi pembanding utama.
6. Metrik utama kinerja deteksi adalah mAP50–95; mAP50, *precision*, *recall*, dan AP per kelas digunakan sebagai metrik tambahan.
7. Efisiensi dievaluasi melalui waktu prapemrosesan, waktu inferensi, latency *end-to-end*, throughput/FPS pada protokol yang sama, dan *peak allocated GPU memory*. Jumlah parameter model dilaporkan sebagai informasi tambahan.
8. Akuisisi citra utama dilakukan pada kondisi yang dikendalikan, meliputi latar belakang, posisi kamera, jarak pengambilan, dan pencahayaan. Kesimpulan utama dibatasi pada kondisi akuisisi yang sebanding.
9. Penelitian tidak membahas cita rasa, proses *roasting*, atau keseluruhan proses penentuan grade mutu kopi.

## 1.4 Tujuan Penelitian

Penelitian ini bertujuan untuk menerapkan dan menganalisis prapemrosesan citra berbasis frekuensi-angular pada YOLO26n untuk deteksi *fine-grained* cacat biji kopi, menentukan konfigurasi dari variasi desain yang diuji, serta mengevaluasi pengaruhnya terhadap kinerja deteksi dan biaya komputasi.

## 1.5 Manfaat Penelitian

Manfaat yang diharapkan dari penelitian ini adalah:

1. Memberikan kajian mengenai penerapan prapemrosesan frekuensi-angular pada deteksi *fine-grained* cacat biji kopi menggunakan YOLO26n.
2. Menunjukkan pengaruh prapemrosesan terhadap kinerja keseluruhan dan kelas yang lebih sulit dikenali, termasuk perbandingannya dengan citra tanpa prapemrosesan dan CLAHE.
3. Menjadi referensi bagi pengembangan sistem inspeksi mutu biji kopi berbasis *computer vision* yang memanfaatkan pengolahan citra pada domain frekuensi.
