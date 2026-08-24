# BAB II — TINJAUAN PUSTAKA

Status: citation-ready rewrite in progress following the adopted USU/campus proposal pattern. Sections 2.1–2.6 have been rewritten from verified primary/source-level evidence; Sections 2.7–2.9 remain working prose until their own source-level audit is closed.

All bracketed citation keys must resolve through `docs/thesis/sources/CANONICAL_SOURCE_KEYS.md` or the master reference workbook. Numerical and methodological claims remain paper-scoped and must be traceable to the primary source.

---

## 2.1 Biji Kopi Hijau dan Cacat Fisik Biji Kopi

Menurut SNI 01-2907-2008, kopi didefinisikan sebagai biji dari tanaman *Coffea* spp. dalam bentuk bugil dan belum disangrai. Standar tersebut mencakup biji kopi Robusta dan Arabika serta menetapkan penggolongan, persyaratan mutu, cara pengujian, penandaan, dan pengemasan. Dengan demikian, konteks biji kopi hijau pada penelitian ini merujuk pada biji yang belum mengalami penyangraian dan masih dinilai berdasarkan kondisi fisiknya sebelum proses hilir berikutnya [STD-01].

SNI 01-2907-2008 mendefinisikan sejumlah kondisi fisik yang menjadi dasar inspeksi mutu, antara lain biji hitam, biji hitam sebagian, biji hitam pecah, kopi gelondong, biji coklat, kulit kopi dan kulit tanduk dalam beberapa ukuran, biji pecah, biji muda, biji berlubang akibat serangga, serta ranting, tanah, atau batu sebagai kotoran. Standar tersebut juga menggunakan sistem nilai cacat untuk menggolongkan mutu kopi; berdasarkan jumlah nilai cacat, kopi dikelompokkan ke dalam enam tingkat mutu, dengan pembagian tambahan mutu 4a dan 4b pada kopi Robusta [STD-01]. Oleh sebab itu, istilah *cacat fisik* dalam tesis ini dipakai untuk menunjuk karakteristik kondisi biji atau material asing yang dapat diamati secara visual, bukan untuk menyatakan bahwa sistem computer vision yang dibangun merekonstruksi seluruh prosedur grading SNI.

Implementasi taxonomy cacat pada penelitian computer vision tidak selalu identik satu sama lain. Kesiman et al. membangun dataset yang mengacu pada SNI 01-2907-2008. Pada tahap pengumpulan, ahli mengidentifikasi sampel berdasarkan 20 jenis cacat yang tercantum dalam standar, tetapi subset akhir untuk klasifikasi cacat terdiri atas 17 kelas karena satu jenis cacat tidak ditemukan pada korpus dan tiga ukuran material asing digabungkan menjadi satu kelas [COF-07]. Secara independen, Arwatchananukul et al. membangun dataset Thai Arabica yang juga terdiri atas 17 jenis cacat, yaitu *broken, cut, dry cherry, fade, floater, full black, full sour, fungus damage, husk, immature, parchment, partial black, partial sour, severe insect damage, shell, slight insect damage,* dan *withered* [COF-08]. Pada sisi object detection, Bahy dan Rifai menggunakan 20 kategori cacat fisik yang disejajarkan dengan SNI 01-2907-2008 [COF-02].

Perbedaan tersebut penting secara metodologis. Standar mutu menyediakan vocabulary dan konteks penilaian, sedangkan label yang benar-benar dipelajari model ditentukan oleh taxonomy dataset dan protokol anotasi penelitian. Karena itu, kelas pada eksperimen tesis harus dijelaskan berdasarkan dataset yang digunakan dan tidak boleh disamakan secara otomatis dengan keseluruhan sistem nilai cacat atau proses grading manual pada SNI [STD-01][COF-07][COF-08][COF-02].

---

## 2.2 Inspeksi Mutu Biji Kopi: Metode Konvensional dan Tantangannya

Identifikasi cacat biji kopi secara konvensional masih bergantung pada pengamatan manusia. Dalam pengembangan dataset berbasis SNI, Kesiman et al. menjelaskan bahwa identifikasi biji cacat dan jenis cacat pada umumnya dilakukan secara manual; kendala yang mereka soroti adalah keterbatasan waktu dan tenaga serta sulitnya memperoleh pekerja berpengalaman untuk melakukan identifikasi tersebut [COF-07]. Arwatchananukul et al. juga menempatkan ketergantungan pada tenaga manual sebagai persoalan pada proses pemilahan green coffee bean, sedangkan Muchtar et al. menekankan bahwa sortasi manual memerlukan waktu, rentan terhadap kesalahan akibat kelelahan, dan dapat menghasilkan kualitas yang tidak konsisten [COF-08][COF-14]. Pernyataan ini mendukung kebutuhan otomasi inspeksi, tetapi tidak berarti seluruh kesalahan grading manual dapat direplikasi atau diselesaikan hanya dengan model vision.

Sebelum dominasi deep learning, otomasi inspeksi kopi telah dikembangkan melalui pengolahan citra dan *computational intelligence*. De Oliveira et al. menggunakan ruang pengambilan gambar terkontrol, sistem pencahayaan, kalibrasi warna, serta pemetaan dari RGB ke CIE L*a*b* untuk memperoleh pengukuran warna green coffee bean yang konsisten. Pada kondisi eksperimen mereka, karakteristik warna kemudian digunakan untuk klasifikasi berbasis *Naive Bayes* dan dibandingkan dengan inspeksi visual oleh ahli [COF-10]. Studi tersebut menunjukkan bahwa informasi visual biji kopi telah lama dimanfaatkan secara komputasional, tetapi juga memperlihatkan bahwa pipeline klasik bergantung pada rekayasa kondisi akuisisi dan representasi fitur yang ditentukan sebelumnya.

Perkembangan berikutnya memperluas solusi dari fitur buatan menuju CNN, Transformer, serta sistem yang dirancang untuk deployment. Tinjauan Motta et al. memetakan penggunaan beragam teknik machine learning dan computer vision untuk klasifikasi kopi, sementara studi Muchtar et al. menunjukkan contoh implementasi modern dengan membandingkan beberapa arsitektur CNN dan Transformer serta menjalankan sistem pada perangkat edge untuk klasifikasi green coffee bean [REV-01][COF-14]. Perubahan ini menunjukkan bahwa otomasi inspeksi tidak lagi terbatas pada satu jenis fitur atau satu model, tetapi mencakup keseluruhan pipeline akuisisi, representasi, klasifikasi, dan deployment.

Namun, keberadaan sistem otomatis tidak dengan sendirinya menyelesaikan persoalan pembedaan kelas yang sangat mirip. Ketika taxonomy diperinci, kebutuhan terhadap representasi yang mampu mempertahankan perbedaan visual antarkelas menjadi lebih besar. Persoalan tersebut dibahas secara khusus pada Subbab 2.6 agar bukti tentang *fine-grained discrimination* tidak tercampur dengan pembahasan umum mengenai keterbatasan inspeksi manual.

---

## 2.3 Object Detection

*Object detection* merupakan tugas computer vision yang memprediksi **kategori objek** sekaligus **lokasinya** pada sebuah citra. Dua keluarga historis yang penting untuk memahami rancangan detector modern adalah pendekatan *two-stage* dan *one-stage*. Faster R-CNN, misalnya, menggunakan *Region Proposal Network* (RPN) yang berbagi fitur konvolusional dengan jaringan deteksi untuk menghasilkan kandidat lokasi objek sebelum kandidat tersebut diproses oleh tahap deteksi berikutnya [DET-02]. Sebaliknya, YOLO merumuskan deteksi sebagai persoalan regresi langsung dari citra penuh menuju koordinat bounding box dan probabilitas kelas dalam satu jaringan [DET-03]. Pembagian ini digunakan sebagai konteks arsitektural; penelitian ini sendiri berfokus pada keluarga YOLO.

Untuk suatu bounding box prediksi \(B_p\) dan ground-truth \(B_g\), kesesuaian spasial keduanya dapat dinyatakan melalui *Intersection over Union* (IoU):

\[
IoU(B_p,B_g)=\frac{|B_p \cap B_g|}{|B_p \cup B_g|}.
\]

Nilai IoU meningkat ketika area irisan prediksi dan ground-truth semakin besar relatif terhadap area gabungannya. Dalam evaluasi detector, informasi lokalisasi ini kemudian berinteraksi dengan prediksi kelas dan confidence untuk menentukan apakah suatu prediksi dianggap benar pada threshold evaluasi tertentu.

Walaupun classification dan localization dilatih dalam satu sistem deteksi, kedua tugas tersebut tidak identik. TOOD menunjukkan adanya *task misalignment* antara classification dan localization pada one-stage detectors dan secara eksplisit merancang head serta proses pembelajaran untuk menyelaraskan keduanya [DIAG-01]. Wu et al. menunjukkan melalui eksperimen head bahwa representasi yang menguntungkan classification tidak selalu identik dengan representasi yang paling sesuai untuk bounding-box regression [DIAG-02]. IoU-Net selanjutnya memisahkan *classification confidence* dari estimasi *localization confidence* dan menggunakan prediksi IoU sebagai ukuran kualitas lokalisasi [DIAG-03].

Pemisahan konseptual tersebut menjadi penting pada tesis ini. Apabila sebuah perlakuan preprocessing menaikkan mAP, kenaikan tersebut tidak langsung ditafsirkan sebagai perbaikan lokalisasi. Analisis tambahan perlu melihat apakah perubahan lebih konsisten dengan peningkatan akses terhadap objek, kualitas lokalisasi, atau kemampuan model memberi kelas dan confidence yang benar pada proposal yang sudah dapat dilokalisasi.

---

## 2.4 YOLO (You Only Look Once)

Redmon et al. memperkenalkan YOLO dengan merumuskan object detection sebagai satu masalah regresi dari citra penuh menuju bounding box dan probabilitas kelas. Satu jaringan konvolusional melakukan prediksi tersebut secara langsung dalam satu evaluasi, sehingga pipeline dapat dioptimalkan secara end-to-end dan dirancang untuk inferensi real-time [DET-03]. Gagasan utama ini membedakan YOLO dari pipeline region-based yang memisahkan pembentukan proposal dan klasifikasi objek ke beberapa tahap.

Dalam perkembangan berikutnya, keluarga YOLO mengalami perubahan besar pada backbone, feature aggregation, detection head, assignment strategy, loss, serta prosedur inferensi. Karena tesis ini tidak meneliti evolusi historis tiap versi YOLO, pembahasan dibatasi pada dua hal yang relevan: YOLO sebagai keluarga *one-stage / real-time detector* dan bukti penggunaannya pada inspeksi green coffee bean. Dengan batas tersebut, performa YOLO versi lama tidak dipakai untuk menyimpulkan karakteristik YOLO26.

Pada domain kopi, Gope et al. membandingkan beberapa generasi YOLO pada deteksi green coffee bean dan menunjukkan bahwa keluarga YOLO dapat digunakan secara efektif pada taxonomy empat kelas di dataset mereka [COF-06]. Hong et al. kemudian menggunakan YOLOv10 sebagai basis deteksi cacat kopi dan memodifikasi representasi internal untuk menghadapi kategori cacat yang memiliki perbedaan visual subtil [COF-01]. Pada taxonomy yang lebih besar, Bahy dan Rifai menerapkan YOLOv5s pada 20 kategori fisik berbasis SNI dan melaporkan heterogenitas performa yang cukup besar antar kelas [COF-02]. Ketiga studi tersebut tidak dapat dibandingkan angka performanya secara langsung karena dataset, jumlah kelas, dan protokolnya berbeda, tetapi bersama-sama menunjukkan bahwa YOLO merupakan keluarga detector yang relevan untuk domain green coffee sekaligus bahwa meningkatnya granularitas kelas tetap menuntut analisis per kelas.

---

## 2.5 YOLO26

YOLO26 merupakan keluarga model real-time vision yang diperkenalkan Jocher et al. melalui preprint arXiv tahun 2026; karena status sumber utamanya masih **preprint**, penelitian ini tidak memberinya label quartile jurnal [DET-01]. Untuk object detection, salah satu perubahan utamanya adalah desain *dual-head* yang mendukung inferensi end-to-end tanpa NMS sebagai jalur native, disertai penghapusan Distribution Focal Loss (DFL) dari regression head. Paper tersebut juga memperkenalkan MuSGD untuk optimisasi, Progressive Loss untuk menggeser supervisi menuju head yang digunakan saat inferensi, serta STAL untuk assignment yang dirancang menjamin positive coverage bagi objek kecil [DET-01].

Secara arsitektural, diagram resmi YOLO26 menunjukkan alur backbone–neck–detect head dengan feature pyramid P3/P4/P5. Backbone menggunakan blok konvolusional dan C3k2, dilanjutkan SPPF dan C2PSA pada tingkat fitur dalam; neck melakukan upsampling, concatenation, dan downsampling untuk menggabungkan fitur lintas skala sebelum tiga detect head menghasilkan prediksi pada resolusi yang berbeda [DET-01]. Keluarga YOLO26 tersedia dalam skala n/s/m/l/x dan paper melaporkan dukungan untuk beberapa tugas vision, tetapi tesis ini menggunakan varian detection yang telah ditentukan dalam protokol eksperimen, bukan seluruh keluarga model.

Posisi YOLO26 dalam penelitian ini adalah **detector terkontrol**, bukan objek utama modifikasi arsitektur. Kontribusi yang diuji ditempatkan sebelum detector, yaitu pada citra input. Secara konseptual perbandingan utama ditulis sebagai:

\[
I \xrightarrow{\text{YOLO26}} \hat{Y}_{\text{native}}
\]

berhadapan dengan

\[
I \xrightarrow{\text{AF2}} I' \xrightarrow{\text{YOLO26}} \hat{Y}_{\text{AF2}}.
\]

Dengan rancangan ini, struktur internal YOLO26 dipertahankan sama pada baseline dan treatment. Perbedaan yang hendak diisolasi adalah representasi citra yang diterima detector, sehingga AF2 tidak boleh disebut sebagai modul backbone, neck, atau detection head YOLO26.

---

## 2.6 Fine-Grained Object Detection

*Fine-grained recognition* membahas pengenalan kategori yang berada pada tingkat subordinat dan memiliki kemiripan visual tinggi, sehingga keputusan kelas bergantung pada perbedaan yang relatif subtil [FG-03]. Ketika persoalan tersebut digabungkan dengan object detection, tugas tidak hanya menuntut model menemukan lokasi objek, tetapi juga mengklasifikasikannya secara tepat ke subkategori yang berdekatan secara visual. Xie et al. mendefinisikan fine-grained object detection (FGOD) dengan karakteristik tersebut dan menunjukkan bahwa kesulitan utamanya bukan semata-mata memperoleh bounding box, tetapi juga membangun representasi yang cukup diskriminatif untuk klasifikasi pada tingkat subkategori [FG-02]. Dengan demikian, banyaknya kelas saja tidak otomatis menjadikan suatu dataset *fine-grained*; yang lebih penting adalah kedekatan visual antar kelas yang harus dibedakan.

Masalah representasi tersebut juga berhubungan dengan relasi classification dan localization. Xie et al. menunjukkan bahwa penggunaan representasi yang sama untuk fine-grained classification dan localization dapat menimbulkan *representation conflict*, sedangkan representasi yang sepenuhnya terpisah masih dapat mengalami keterbatasan karena *feature misalignment*. DRNet yang mereka usulkan menambahkan fine-grained branch, dual refinement, dan confusion-minimized loss agar model memberi perhatian lebih besar pada sampel yang sulit dibedakan [FG-02]. Paper ini digunakan sebagai landasan bahwa FGOD memerlukan kemampuan diskriminasi yang spesifik; komponen DRNet sendiri tidak diasumsikan sebagai solusi untuk kopi.

Bukti pada domain kopi menunjukkan pola yang konsisten bahwa granularitas taxonomy memperbesar kesulitan diskriminasi. Pada benchmark Kesiman et al., klasifikasi tiga kelas kasar menghasilkan test accuracy 92.52% pada MobileNet dan 91.29% pada InceptionResNetV2, tetapi ketika tugas diperinci menjadi 17 jenis cacat, hasilnya turun menjadi 39.82% dan 53.35% [COF-07]. Arwatchananukul et al. juga menggunakan 17 jenis cacat green Arabica; meskipun 5-fold cross-validation berada pada rentang 98.78–99.84%, pengujian pada unseen data turun menjadi 88.63% [COF-08]. Kedua studi tersebut merupakan classification evidence, sehingga tidak digunakan sebagai hasil object detection, tetapi relevan untuk menunjukkan bahwa pembedaan kelas cacat yang rinci dapat menjadi problem representasi tersendiri.

Pada object detection, ketimpangan antar kelas terlihat lebih langsung. Jundullah et al. menggunakan YOLOv8s untuk 20 kelas cacat dan kontaminan serta melaporkan rata-rata mAP@0.5 sebesar 0.75. Namun performa kelas berbeda jauh: *partially black seeds* dilaporkan 0.00 mAP@0.5, *whole black seeds* 0.42, dan *sour bean* 0.35, sementara beberapa kelas dengan ciri visual yang lebih khas mencapai nilai yang jauh lebih tinggi. Penulis secara eksplisit menyatakan bahwa kelas dengan karakteristik visual yang distinctive lebih mudah dikenali, sedangkan defect yang visually similar cenderung memiliki performa lebih rendah [COF-05]. Hebert dan Alamsyah juga melaporkan ketimpangan besar pada 15 kategori defect, termasuk AP yang sangat rendah pada *floater*, *fungus damage*, dan *slight insect damage*; penjelasan penulis mengaitkannya dengan ciri yang kecil, tertutup tekstur, atau menyerupai permukaan/normal bean [COF-04]. Pada studi yang lebih kecil, Samudra dan Rachmawati secara khusus mengidentifikasi kebingungan antara *black* dan *partially black* akibat kemiripan visual [COF-03].

Literatur kopi juga menunjukkan bahwa respons terhadap masalah tersebut umumnya dilakukan dengan meningkatkan representasi **di dalam model**. Jiao et al. menggunakan Swin Transformer, feature fusion dari beberapa stage melalui HS-FPN, dan selective attention untuk meningkatkan discriminative power sebelum klasifikasi; dataset mereka memuat kelompok defect yang disubdivisi menjadi sembilan jenis [COF-12]. Hu et al. secara eksplisit menyebut *subtle visual differences between defect categories* sebagai tantangan dan menggunakan Siamese network untuk similarity-based few-shot recognition; pada dataset enam jenis defect, pendekatan tersebut mencapai accuracy 94.95% dibandingkan 74.35% pada CNN konvensional dalam protokol mereka [COF-13]. Temuan ini tidak membuktikan bahwa satu arsitektur tertentu harus dipakai, tetapi memperlihatkan bahwa literatur kopi sendiri memandang kualitas representasi diskriminatif sebagai bagian penting dari persoalan.

Berdasarkan rangkaian bukti tersebut, penelitian ini menempatkan **fine-grained visual discrimination** sebagai problem utama yang hendak dianalisis pada deteksi cacat biji kopi. Posisi ini berbeda dari klaim bahwa bottleneck telah terbukti berada pada domain frekuensi. Paper kopi di atas mendukung adanya kemiripan visual, ketimpangan per kelas, dan kebutuhan representasi yang lebih diskriminatif, tetapi tidak menunjukkan bahwa informasi frekuensi atau angular merupakan penyebab kesulitan tersebut. Oleh karena itu, pemrosesan frekuensi-angular pada penelitian ini diperlakukan sebagai **candidate solution space** yang perlu dibuktikan secara empiris, bukan konsekuensi kausal yang sudah ditetapkan oleh literatur kopi.

---

## 2.7 Preprocessing Citra untuk Object Detection

Preprocessing citra merupakan transformasi yang dilakukan sebelum citra diproses oleh model utama. Tujuannya dapat berupa normalisasi data, peningkatan kontras, pengurangan noise, penajaman detail, atau transformasi representasi agar informasi yang relevan lebih mudah dimanfaatkan oleh detector.

Pendekatan preprocessing dapat dibedakan secara sederhana menjadi dua kelompok. Pertama, preprocessing tetap (*fixed preprocessing*) seperti CLAHE, filtering, denoising, dan sharpening. Kedua, preprocessing yang parameter atau transformasinya dipelajari secara *task-driven* bersama detector.

Syauqi et al. menggunakan pipeline preprocessing berbasis CLAHE sebelum YOLOv8m pada white pepper dan membandingkannya dengan data tanpa preprocessing [PRE-04]. Chen et al. menggunakan kombinasi wavelet denoising dan image enhancement sebelum YOLOv8 pada maize seed crack detection [PRE-05]. IA-YOLO menunjukkan bahwa transformasi citra dapat dipelajari dengan tujuan meningkatkan downstream detection [PRE-01], sedangkan DENet menggunakan dekomposisi multiskala/frekuensi untuk menghasilkan enhancement yang diarahkan oleh kebutuhan deteksi [PRE-02].

Literatur ini mendukung penggunaan preprocessing sebagai ruang solusi yang sah, tetapi tidak membuktikan bahwa satu jenis preprocessing tertentu akan efektif pada cacat biji kopi.

---

## 2.8 Representasi Citra pada Domain Frekuensi

### 2.8.1 Discrete Fourier Transform dan Fast Fourier Transform

Transformasi Fourier merepresentasikan citra dari domain spasial ke domain frekuensi. Untuk citra diskrit dua dimensi \(f(x,y)\) berukuran \(M \times N\), *Discrete Fourier Transform* dapat dituliskan sebagai:

\[
F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)
\exp\left[-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
\]

Transformasi balik merekonstruksi citra spasial dari representasi frekuensi:

\[
f(x,y)=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)
\exp\left[j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
\]

Fast Fourier Transform (FFT) merupakan algoritma komputasional efisien untuk menghitung DFT dan menjadi dasar implementasi pemrosesan spektral pada penelitian ini.

### 2.8.2 Magnitudo/Amplitudo dan Fase Spektrum

Representasi kompleks Fourier dapat dinyatakan melalui magnitudo dan fase:

\[
A(u,v)=|F(u,v)|,
\]

\[
\phi(u,v)=\operatorname{atan2}(\Im(F(u,v)),\Re(F(u,v))).
\]

Magnitudo menggambarkan kekuatan komponen frekuensi, sedangkan fase berkaitan dengan susunan spasial informasi pada citra. Literatur Fourier-based enhancement seperti FE-YOLO memanfaatkan pemrosesan amplitude/phase sebelum rekonstruksi citra dan deteksi [PRE-03].

Penelitian ini tidak mengasumsikan bahwa magnitude atau phase merupakan bottleneck yang telah terbukti pada kopi.

### 2.8.3 Representasi Radial dan Angular pada Spektrum Fourier

Spektrum dua dimensi dapat dianalisis menggunakan koordinat polar berdasarkan radius frekuensi \(r\) dan sudut \(\theta\). Distribusi radial digunakan untuk merangkum energi berdasarkan skala/frekuensi, sedangkan distribusi angular digunakan untuk merangkum energi berdasarkan orientasi/directionality.

Secara konseptual, energi pada suatu rentang radial dapat dituliskan sebagai:

\[
E_r(r_1,r_2)=\sum_{r_1^2\le u^2+v^2\le r_2^2}|F(u,v)|^2,
\]

sementara energi angular dapat diringkas sebagai:

\[
E_\theta(\theta_1,\theta_2)=
\sum_{\theta_1\le \theta(u,v)\le \theta_2}|F(u,v)|^2.
\]

Literatur texture analysis menunjukkan bahwa radial spectrum dapat berkaitan dengan periodisitas/skala tekstur, sedangkan angular spectrum dapat menggambarkan directionality dan orientasi pola [SPEC-01][SPEC-02].

Inilah dasar teoritis penggunaan istilah **frekuensi-angular** pada penelitian, bukan bukti bahwa operator AF2 telah terbukti efektif pada kopi.

### 2.8.4 Pemrosesan Frekuensi untuk Object Detection

Pemrosesan frekuensi telah digunakan pada object detection dalam beberapa bentuk. FE-YOLO melakukan Fourier enhancement pada citra sebelum YOLO [PRE-03]. Pendekatan wavelet dan Fourier lain memasukkan informasi frekuensi ke feature space detector [AGR-01][AGR-02][WAVE-01][FREQ-03]. Pada fine-grained object detection, Xu et al. mengeksplorasi integrasi representasi frekuensi untuk membedakan kategori yang memiliki perbedaan visual subtil [FG-01].

Dari literatur tersebut dapat ditarik satu posisi yang aman: informasi frekuensi dan arah merupakan ruang representasi yang secara teknis dapat membawa informasi komplementer untuk detection dan texture discrimination. Efektivitas **preprocessing input parameter-free berbasis frekuensi-angular pada fine-grained coffee defect detection** masih harus dibuktikan melalui eksperimen penelitian ini.

---

## 2.9 Penelitian Terkait

Tabel 2.1 merangkum penelitian yang paling relevan terhadap tiga jalur utama tesis: (1) deteksi cacat kopi, (2) preprocessing sebelum object detector, dan (3) pemrosesan frekuensi untuk fine-grained/detection tasks.

### Tabel 2.1. Perbandingan studi relevan deteksi cacat kopi, preprocessing citra, dan pemrosesan frekuensi

| No | Penulis & Tahun | Indeks | Fokus Penelitian | Metode / Model | Kontribusi dan Pengisian Gap Penelitian |
|---:|---|---|---|---|---|
| 1 | Hong et al. (2026) [COF-01] | TBD - verify | Deteksi cacat biji kopi dan peningkatan fine-grained feature extraction | Improved YOLOv10 + DSConv + SPPF-Attention + PConv | Menunjukkan YOLO dapat bekerja sangat baik pada coffee detection, tetapi subtle/visually similar defects masih menjadi motivasi peningkatan representasi. Penelitian yang diusulkan tidak menambah modul internal serupa, melainkan menguji preprocessing input. |
| 2 | Bahy & Rifai (2026) [COF-02] | TBD - verify | Deteksi cacat kopi berbasis SNI dengan taxonomy besar | Lightweight YOLOv5s; 20 kelas | Memberikan bukti bahwa taxonomy 20 kelas menghasilkan heterogenitas performa antarkelas. Mendukung evaluasi class-wise dan lower-tail. |
| 3 | Jundullah et al. (2026) [COF-05] | TBD - verify | Multi-class coffee defect and contaminant detection | YOLOv8s; 20 kelas | Menunjukkan overall performance dapat menyembunyikan kelas yang sulit dan visually similar. Mendukung problem fine-grained discrimination. |
| 4 | Hebert & Alamsyah (2026) [COF-04] | TBD - verify | Deteksi cacat kopi berbasis kategori SCA | YOLOv12; 15 kelas | Menunjukkan beberapa subtle defects memiliki AP sangat rendah dibanding kelas yang visualnya khas. Menguatkan kebutuhan analisis difficult classes. |
| 5 | Kesiman et al. (2023) [COF-07] | TBD - verify | Benchmark klasifikasi cacat kopi berbasis SNI | MobileNet dan InceptionResNetV2; 3 vs 17 kelas | Menunjukkan peningkatan granularitas taxonomy secara drastis meningkatkan kesulitan diskriminasi. Digunakan sebagai diagnostic evidence, bukan bukti object detection. |
| 6 | Syauqi et al. (2025) [PRE-04] | IEEE conference - verify index | Defect detection pada white pepper dengan image enhancement sebelum YOLO | CLAHE-based composite preprocessing + YOLOv8m | Memberi analog komoditas berbentuk biji bahwa fixed preprocessing sebelum detector dapat meningkatkan downstream detection; hanya 2 kelas dan bukan coffee. |
| 7 | Chen et al. (2024) [PRE-05] | Computers and Electronics in Agriculture - verify quartile | Deteksi retak biji jagung berbasis soft X-ray | Image enhancement + optimized YOLOv8 | Memisahkan kontribusi image enhancement dari optimasi detector; mendukung preprocessing sebagai treatment eksperimental yang dapat diuji secara terkontrol. |
| 8 | Liu et al. (2022), IA-YOLO [PRE-01] | AAAI - verify index wording | Object detection pada adverse weather melalui task-driven image preprocessing | Differentiable image processing + YOLOv3 | Menunjukkan preprocessing dapat diarahkan untuk utility deteksi, bukan hanya kualitas visual. Learned/adaptive dan berbeda dari AF2 parameter-free. |
| 9 | Li et al. (2025), FE-YOLO [PRE-03] | Digital Signal Processing - verify quartile | Low-light detection dengan Fourier-domain enhancement | FFT + learned FENet/FPB + IFFT + YOLO | Pembanding metodologis terdekat untuk input-space Fourier enhancement. Berbeda karena learned, low-light oriented, dan tidak memakai angular spectral selection. |
| 10 | Xu et al. (2025) [FG-01] | TBD - verify | Fine-grained object detection dengan language dan frequency representations | LFDet + AFAB/AFAB-2 | Memberi bridge bahwa frequency representation dapat digunakan untuk fine-grained detection; domain aircraft sehingga transfer ke kopi belum terbukti. |
| 11 | **Penelitian yang Diusulkan** | - | Analisis preprocessing frekuensi-angular untuk deteksi fine-grained cacat biji kopi | **AF2 parameter-free preprocessing + YOLO26** | Mengisi gap dengan mengevaluasi input-space frequency-angular preprocessing pada coffee defect detection secara matched terhadap native YOLO26, disertai analisis aggregate, lower-tail, classification/localization diagnostics, dan efficiency trade-off. |

### Positioning summary

Pola penelitian terdahulu dapat disederhanakan menjadi:

```text
coffee defect detection
    ↓
YOLO / CNN / Transformer improvements inside the model
    ↓
finer taxonomy exposes class-wise discrimination difficulty
```

sementara literatur preprocessing dan frekuensi menunjukkan:

```text
image preprocessing can affect downstream detection
    +
frequency / directional representations can encode complementary visual information
```

Penelitian ini menguji titik pertemuan kedua jalur tersebut melalui:

```text
RGB image
    ↓
parameter-free frequency-angular preprocessing
    ↓
YOLO26
    ↓
fine-grained coffee-defect detection
```

The effectiveness of this transfer is an empirical research question and must not be assumed from prior non-coffee studies.