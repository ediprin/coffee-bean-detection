# BAB I
# PENDAHULUAN

## 1.1 Latar Belakang

Kualitas biji kopi merupakan salah satu faktor penting dalam proses pengendalian mutu karena kondisi fisik biji menjadi salah satu dasar dalam penilaian kualitas kopi. Biji kopi dapat memiliki berbagai jenis cacat fisik yang perlu diidentifikasi pada proses pemeriksaan dan pemilahan. Pada praktiknya, proses tersebut masih dapat dilakukan melalui pengamatan visual manusia sehingga konsistensi hasil pemeriksaan dapat dipengaruhi oleh pengalaman, kondisi pengamatan, pelatihan, dan beban kerja pemeriksa. García et al. (2019) menunjukkan bahwa proses seleksi manual biji kopi hijau dapat dipengaruhi oleh faktor operator dan waktu kerja. Kondisi tersebut mendorong pengembangan sistem berbasis *computer vision* untuk membantu proses identifikasi cacat secara lebih konsisten.

Perkembangan *deep learning* memungkinkan inspeksi biji kopi dilakukan secara otomatis melalui klasifikasi maupun *object detection*. Pada *object detection*, model tidak hanya menentukan kategori objek, tetapi juga menentukan lokasi objek pada citra. Salah satu keluarga model yang banyak digunakan adalah *You Only Look Once* (YOLO) karena mampu melakukan proses klasifikasi dan lokalisasi dalam satu sistem deteksi. Hong et al. (2026) menggunakan pengembangan YOLOv10 untuk mendeteksi tujuh kategori cacat biji kopi, sedangkan Gope et al. (2024) membandingkan beberapa varian YOLO pada deteksi biji kopi hijau. Penelitian tersebut menunjukkan bahwa keluarga YOLO dapat diterapkan pada deteksi cacat biji kopi. Namun, kinerja yang baik pada jumlah kategori yang relatif terbatas belum menunjukkan bahwa deteksi cacat dengan jumlah kategori yang lebih rinci telah terselesaikan.

Pada penelitian dengan jumlah kategori yang lebih banyak, perbedaan kemampuan model dalam mengenali setiap kelas menjadi lebih terlihat. Bahy dan Rifai (2026) menerapkan YOLOv5s pada 20 kategori fisik biji kopi dan menunjukkan adanya variasi kinerja antarkelas. Jundullah et al. (2026) melaporkan evaluasi sistem YOLOv8s pada 20 kelas cacat dan kontaminan serta menyatakan bahwa kategori dengan karakteristik visual yang lebih khas cenderung lebih mudah dikenali dibandingkan beberapa kategori yang memiliki kemiripan visual. Hebert dan Alamsyah (2026) menemukan bahwa beberapa kategori cacat memiliki nilai AP yang jauh lebih rendah dibandingkan kategori lain. Kesulitan tersebut antara lain berkaitan dengan tanda cacat yang berukuran kecil, tekstur permukaan biji, dan kemiripan karakteristik visual antarjenis cacat.

Kondisi tersebut menunjukkan adanya karakteristik *fine-grained*, yaitu ketika beberapa kategori masih berasal dari jenis objek yang sama tetapi memiliki perbedaan visual yang relatif kecil. Kesiman et al. (2023) menunjukkan bahwa peningkatan jumlah kategori dari tiga kelas menjadi 17 kelas cacat menyebabkan penurunan akurasi yang besar pada model klasifikasi yang diuji. Hu et al. (2025) menyebutkan bahwa perbedaan visual yang halus antarjenis cacat menjadi salah satu tantangan dalam pengenalan cacat biji kopi. Temuan tersebut menunjukkan bahwa kemampuan model dalam membedakan karakteristik visual antarkelas menjadi aspek penting pada deteksi cacat biji kopi dengan kategori yang rinci. Meskipun demikian, tidak seluruh kategori yang nantinya dipertahankan harus memiliki tingkat kesulitan visual yang sama. Beberapa kategori dapat memiliki ciri yang lebih jelas, sedangkan analisis *fine-grained* terutama diperlukan untuk melihat perilaku model pada kategori yang memiliki kemiripan visual lebih tinggi.

Sejumlah penelitian telah mengembangkan metode untuk meningkatkan kemampuan tersebut. Hong et al. (2026) melakukan modifikasi pada proses ekstraksi dan pemrosesan fitur pada YOLOv10. Jiao et al. (2025) menggunakan *Swin Transformer*, *multistage feature fusion*, dan *selective attention* untuk meningkatkan representasi fitur pada proses grading dan identifikasi cacat biji kopi. Hu et al. (2025) menggunakan *Siamese network* untuk meningkatkan kemampuan model dalam membedakan kategori dengan karakteristik visual yang serupa. Penelitian-penelitian tersebut menunjukkan bahwa salah satu arah yang telah banyak dikaji adalah peningkatan representasi melalui komponen di dalam model.

Selain melalui modifikasi arsitektur, pengolahan citra sebelum masuk ke model deteksi juga telah digunakan pada berbagai penelitian *object detection*. Liu et al. (2022) melalui IA-YOLO menggunakan serangkaian proses pengolahan citra sebelum YOLO dan menyesuaikannya berdasarkan kebutuhan tugas deteksi. Qin et al. (2022) memanfaatkan pemisahan informasi frekuensi rendah dan tinggi sebelum citra hasil rekonstruksi diberikan kepada model deteksi. Li et al. (2025) menggunakan pemrosesan pada domain Fourier terhadap komponen amplitudo dan fase sebelum citra diproses oleh YOLO. Penelitian-penelitian tersebut menunjukkan bahwa perubahan pada citra masukan dapat memengaruhi informasi yang diterima oleh model deteksi dan dapat dievaluasi berdasarkan kinerja deteksi yang dihasilkan.

Pendekatan serupa juga ditemukan pada objek pertanian. Syauqi et al. (2025) menggunakan prapemrosesan citra sebelum YOLOv8 pada deteksi cacat *white pepper*, sedangkan Chen et al. (2024) menggunakan kombinasi pengolahan berbasis *wavelet* dan peningkatan citra sebelum YOLOv8 untuk mendeteksi keretakan pada biji jagung. Penelitian tersebut memberikan contoh bahwa pengolahan pada citra masukan dapat ditempatkan sebagai bagian tersendiri sebelum proses deteksi. Meskipun demikian, hasil pada komoditas dan kondisi citra tersebut tidak dapat langsung digunakan untuk menyimpulkan efektivitas pendekatan yang sama pada cacat biji kopi.

Salah satu bentuk pengolahan citra yang dapat digunakan adalah pemrosesan pada domain frekuensi. Melalui transformasi Fourier, informasi citra dapat direpresentasikan berdasarkan komponen frekuensinya. Selain besarnya komponen frekuensi, distribusi energi spektral juga dapat dianalisis berdasarkan arah. Cao et al. (2019) menunjukkan penggunaan distribusi radial dan angular dari spektrum Fourier dalam analisis tekstur, sedangkan Zhang dan Tan (2003) menunjukkan bahwa distribusi orientasi spektral dapat digunakan sebagai informasi diskriminatif. Pada tugas *fine-grained object detection*, Xu et al. (2025) menggunakan pemrosesan frekuensi lokal dan informasi angular pada deteksi kategori pesawat yang memiliki perbedaan visual yang halus. Hasil tersebut menunjukkan bahwa informasi frekuensi dan arah layak dipertimbangkan sebagai salah satu bentuk representasi citra, tetapi efektivitasnya pada deteksi cacat biji kopi masih perlu diuji.

Berdasarkan uraian tersebut, penelitian ini mengusulkan penggunaan prapemrosesan citra berbasis frekuensi-angular sebelum proses deteksi menggunakan YOLO26n. Citra akan dianalisis secara lokal pada domain Fourier sehingga informasi spektral dapat diproses berdasarkan distribusi arahnya, kemudian hasil pengolahan dikembalikan ke domain spasial sebelum diberikan kepada model deteksi. Pendekatan ini ditempatkan pada citra masukan sehingga arsitektur utama YOLO26n tidak menjadi objek modifikasi dalam penelitian.

Rancangan prapemrosesan frekuensi-angular memiliki beberapa keputusan desain yang dapat memengaruhi citra hasil pengolahan. Oleh karena itu, optimasi pada penelitian ini dilakukan dengan menguji beberapa variasi desain yang telah ditetapkan terlebih dahulu, bukan melalui pencarian *global optimum*. Konfigurasi referensi dan konfigurasi terpilih akan dibandingkan dengan YOLO26n tanpa prapemrosesan. CLAHE juga digunakan sebagai pembanding peningkatan kontras lokal konvensional agar pengaruh prapemrosesan frekuensi-angular tidak hanya dibandingkan terhadap citra asli. Selain kinerja deteksi secara keseluruhan dan per kelas, biaya komputasi tambahan akibat prapemrosesan juga akan dievaluasi.

Penelitian menggunakan dataset primer yang akan dikumpulkan secara langsung. Daftar kelas awal menargetkan 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008, ditambah satu kelas biji normal. Jumlah kelas akhir ditetapkan setelah kecukupan data pada setiap kelas diperiksa sehingga kategori yang sangat langka tidak dipaksakan menjadi kelas evaluasi hanya melalui augmentasi.

Berdasarkan latar belakang tersebut, penelitian ini dilakukan dengan judul **“Analisis dan Optimasi Prapemrosesan Citra Berbasis Frekuensi-Angular pada YOLO26 untuk Deteksi Fine-Grained Cacat Biji Kopi.”**

## 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, rumusan masalah penelitian ini adalah:

1. Bagaimana prapemrosesan citra berbasis frekuensi-angular yang mengadaptasi prinsip AFAB-2 dapat diterapkan pada citra masukan YOLO26n tanpa mengubah arsitektur utama model deteksi?
2. Bagaimana pengaruh variasi desain prapemrosesan yang meliputi fungsi jendela, representasi orientasi, informasi radial, fungsi ambang, dan panduan luminansi terhadap kinerja deteksi pada tahap pengembangan, serta bagaimana konfigurasi kandidat dipilih berdasarkan prosedur yang telah ditetapkan?
3. Bagaimana kinerja konfigurasi frekuensi-angular terpilih dibandingkan dengan YOLO26n tanpa prapemrosesan dan CLAHE pada data uji akhir ditinjau dari kinerja deteksi secara keseluruhan, kinerja per kelas, kesalahan deteksi, dan biaya komputasi?

## 1.3 Batasan Masalah

Adapun batasan masalah pada penelitian ini agar penelitian tetap berada pada ruang lingkup yang telah ditentukan yaitu:

1. Penelitian berfokus pada *object detection* biji kopi hijau menggunakan dataset primer yang dikumpulkan secara langsung. Daftar kelas awal menargetkan 20 kategori cacat fisik dan benda asing yang digunakan dalam penilaian SNI 2907:2008 serta satu kelas biji normal, sedangkan jumlah kelas akhir ditetapkan setelah kecukupan data tiap kelas diperiksa.
2. Model utama *object detection* yang digunakan adalah YOLO26 dengan varian YOLO26n. Evaluasi menggunakan RT-DETRv3-R18 dapat dilakukan sebagai analisis tambahan dan tidak menjadi dasar pemilihan metode utama.
3. Pengembangan metode difokuskan pada prapemrosesan citra berbasis frekuensi-angular sebelum citra diproses oleh YOLO26n dan tidak melakukan modifikasi pada *backbone*, *neck*, maupun *detection head* YOLO26n pada eksperimen utama.
4. Optimasi dilakukan melalui pengujian variasi desain dan parameter utama prapemrosesan frekuensi-angular yang telah ditetapkan dalam metodologi penelitian. Penelitian tidak melakukan pencarian *global optimum* terhadap seluruh kemungkinan konfigurasi.
5. YOLO26n tanpa prapemrosesan digunakan sebagai model acuan, sedangkan CLAHE digunakan sebagai pembanding peningkatan kontras lokal konvensional. Wavelet tidak menjadi pembanding utama.
6. Metrik utama kinerja deteksi adalah mAP50–95. Metrik mAP50, *precision*, *recall*, serta AP setiap kelas digunakan sebagai metrik tambahan.
7. Efisiensi metode dievaluasi berdasarkan biaya komputasi yang ditimbulkan oleh prapemrosesan, seperti waktu prapemrosesan, waktu inferensi, jumlah citra yang dapat diproses per detik, dan penggunaan memori.
8. Pengambilan citra utama dilakukan pada kondisi yang dikendalikan, meliputi latar belakang, posisi kamera, jarak pengambilan, dan pencahayaan. Oleh karena itu, kesimpulan utama penelitian dibatasi pada kondisi akuisisi yang sebanding dan tidak dimaksudkan sebagai klaim ketahanan pada seluruh kondisi lapangan yang tidak terkendali.
9. Penelitian tidak membahas penilaian cita rasa, proses *roasting*, maupun keseluruhan proses penentuan grade mutu kopi.

## 1.4 Tujuan Penelitian

Berdasarkan rumusan masalah tersebut, tujuan penelitian ini adalah:

1. Menerapkan prapemrosesan citra berbasis frekuensi-angular yang mengadaptasi prinsip AFAB-2 pada citra masukan YOLO26n dengan mempertahankan arsitektur utama model deteksi.
2. Menganalisis pengaruh variasi desain prapemrosesan yang meliputi fungsi jendela, representasi orientasi, informasi radial, fungsi ambang, dan panduan luminansi serta menentukan konfigurasi kandidat berdasarkan prosedur pengembangan yang telah ditetapkan.
3. Mengevaluasi konfigurasi frekuensi-angular terpilih dan membandingkannya dengan YOLO26n tanpa prapemrosesan serta CLAHE pada data uji akhir berdasarkan kinerja deteksi keseluruhan, kinerja per kelas, kesalahan deteksi, dan biaya komputasi.

## 1.5 Manfaat Penelitian

Adapun manfaat yang diharapkan dari penelitian ini adalah:

1. Memberikan kajian mengenai penerapan prapemrosesan citra berbasis frekuensi-angular pada deteksi *fine-grained* cacat biji kopi menggunakan YOLO26n.
2. Memberikan informasi mengenai pengaruh prapemrosesan frekuensi-angular terhadap kinerja deteksi secara keseluruhan maupun pada kelas cacat yang lebih sulit dikenali.
3. Memberikan perbandingan antara prapemrosesan frekuensi-angular, citra tanpa prapemrosesan, dan peningkatan kontras lokal menggunakan CLAHE pada arsitektur deteksi yang sama.
4. Menjadi referensi bagi penelitian selanjutnya dalam pengembangan sistem inspeksi mutu biji kopi berbasis *computer vision*, khususnya yang menggunakan pengolahan citra pada domain frekuensi.
