# BAB II — TINJAUAN PUSTAKA

Status: citation-ready rewrite in progress following the adopted USU/campus proposal pattern. Sections 2.1–2.8 have been rewritten from verified primary/source-level evidence; Section 2.9 remains working prose until its related-work table audit is closed. The final textbook/page-level audit for canonical Fourier foundations also remains open even though the active equations are grounded in primary method papers.

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

Preprocessing citra pada konteks object detection adalah transformasi terhadap citra **sebelum** citra tersebut diterima detector. Transformasi dapat bertujuan memperbaiki kontras, menekan noise, mempertahankan detail, atau mengubah representasi sinyal agar informasi tertentu lebih mudah dimanfaatkan oleh model. Posisi preprocessing perlu dibedakan dari modifikasi backbone, neck, atau detection head karena perlakuan dilakukan pada ruang input dan dapat dievaluasi sebagai treatment tersendiri terhadap detector yang sama.

Literatur menunjukkan sedikitnya dua strategi besar. Strategi pertama menggunakan operator yang ditentukan tanpa jaringan enhancement yang dipelajari, misalnya kombinasi contrast enhancement, denoising, sharpening, atau transform-domain processing. Pada white-pepper defect detection, Syauqi et al. membandingkan YOLOv8m yang dilatih pada citra asli dengan citra yang telah melalui pipeline preprocessing. Walaupun paper menyebutnya *CLAHE-based preprocessing*, pipeline aktualnya bersifat komposit: gamma correction, CLAHE, bilinear blending, non-local means denoising, dan unsharp masking [PRE-04]. Kedua kondisi menggunakan 50 epoch, batch size 16, dan learning rate 0.0001; mAP50–95 yang dilaporkan meningkat dari 79% menjadi 82% pada kondisi enhanced [PRE-04]. Karena task tersebut hanya memiliki dua kelas, hasilnya digunakan sebagai bukti bahwa preprocessing input dapat memengaruhi downstream detection pada komoditas berbentuk biji, bukan sebagai bukti bahwa preprocessing yang sama akan menyelesaikan fine-grained coffee detection.

Contoh lain diberikan Chen et al. pada deteksi retak internal biji jagung menggunakan soft X-ray. Pipeline mereka menggabungkan wavelet-threshold denoising, image standardization, bilateral filtering, dan Laplacian sharpening sebelum YOLOv8 [PRE-05]. Paper tersebut menarik karena memisahkan kontribusi image enhancement dari optimasi arsitektur: optimized YOLOv8 dilaporkan meningkatkan AP sebesar 3.1 percentage points terhadap model asli, sedangkan penerapan image enhancement memberi tambahan 1.8 percentage points dalam setup mereka [PRE-05]. Sekali lagi, domain soft X-ray maize crack berbeda dari RGB green coffee sehingga angka tersebut hanya menjadi precedent desain eksperimen, bukan estimasi gain yang diharapkan pada tesis ini.

Transform-domain preprocessing juga telah digunakan sebagai operator pra-deteksi. Tu et al. mengusulkan WCTE pada low-contrast tablet defect detection dengan alur Haar-wavelet decomposition, frequency-band treatment, reconstruction, quadtree-guided CLAHE, dan bilinear fusion sebelum YOLOv11 [PRE-06]. Mereka melaporkan peningkatan overall mAP sebesar 2.5 percentage points dan penurunan false-detection rate sebesar 9% pada setup tersebut [PRE-06]. Studi ini memperluas ruang preprocessing dari transformasi intensitas murni menuju pemrosesan multiresolusi/transform-domain, tetapi target dan mekanismenya tetap berbeda dari frequency-angular preprocessing yang diuji pada kopi.

Strategi kedua adalah **task-driven learned preprocessing**, yaitu transformasi citra dipelajari bersama detector agar tujuan enhancement selaras dengan objective deteksi. IA-YOLO secara eksplisit mengingatkan bahwa peningkatan kualitas visual citra tidak otomatis meningkatkan detection performance [PRE-01]. Dalam IA-YOLO, CNN-PP memprediksi parameter untuk differentiable image-processing filters—termasuk defog, white balance, gamma, contrast, tone, dan sharpen—kemudian citra hasil filtering diberikan ke YOLOv3. CNN-PP dan detector dilatih end-to-end menggunakan detection loss sehingga parameter preprocessing dipelajari untuk downstream detection, bukan hanya untuk menghasilkan citra yang tampak lebih baik bagi manusia [PRE-01].

DENet mengambil pendekatan task-driven yang berbeda. Input diuraikan dengan Laplacian pyramid menjadi satu komponen low-frequency dan beberapa high-frequency residual components. Low-frequency branch digunakan untuk menangani informasi global seperti illumination/contrast, sedangkan high-frequency components mempertahankan detail seperti edges dan textures. Enhanced image kemudian diberikan ke YOLOv3 dan keseluruhan DE-YOLO dilatih end-to-end menggunakan detection loss tanpa membutuhkan clean-image ground truth [PRE-02]. Dengan demikian, IA-YOLO dan DENet memberi precedent bahwa preprocessing dapat ditempatkan sebelum detector tetapi tetap dioptimalkan berdasarkan utility deteksi.

Perbedaan antara *visual enhancement* dan *detection-oriented enhancement* juga terlihat pada literatur low-light. Retinexformer mengembangkan enhancement berbasis Retinex dan Transformer dan turut mengevaluasi nilai praktis enhancement pada downstream low-light detection [PRE-07]. FE-YOLO melangkah lebih jauh dengan menggabungkan Fourier Enhanced Network dan YOLO dalam satu training objective: amplitude/phase diproses di domain Fourier, kemudian enhancement loss dan detection loss dioptimalkan bersama [PRE-03]. FE-YOLO juga menunjukkan bahwa beberapa enhancement terpisah yang memperbaiki brightness belum tentu menghasilkan deteksi terbaik karena noise atau artifacts yang diperkenalkan preprocessing dapat merugikan detector [PRE-03]. Temuan ini penting sebagai batas interpretasi: citra yang terlihat lebih kontras atau lebih tajam tidak boleh dianggap otomatis lebih informatif bagi YOLO.

Di sisi lain, tidak semua manipulasi input-space harus memiliki jaringan preprocessing yang dipelajari. Fourier Domain Adaptation (FDA) melakukan manipulasi amplitude pada domain Fourier dan inverse transform kembali ke ruang citra tanpa image-translation network; task aslinya adalah unsupervised domain adaptation untuk semantic segmentation, bukan object detection [PRE-08]. FDA karena itu hanya memberi precedent bahwa operasi Fourier non-learned dapat digunakan untuk mengubah statistik citra sambil mempertahankan struktur task, bukan bukti efektivitas AF2 pada kopi.

Berdasarkan spektrum pendekatan tersebut, preprocessing dalam tesis ini ditempatkan sebagai **parameter-free, input-space, content-adaptive spectral preprocessing**. AF2 tidak memiliki CNN-PP seperti IA-YOLO, enhancement network seperti DENet/FE-YOLO, atau parameter trainable lain pada frontend. Namun AF2 juga bukan sekadar filter global dengan konstanta tunggal, karena respons angularnya dihitung dari statistik spektral citra/patch yang sedang diproses. Secara konseptual:

\[
I' = \mathcal{P}_{FA}(I), \qquad \Theta_{\mathcal{P}_{FA}}^{\mathrm{trainable}} = \varnothing,
\]

kemudian

\[
\hat{Y}=\operatorname{YOLO26}(I').
\]

Posisi ini menghasilkan pertanyaan eksperimen yang lebih bersih: apakah transformasi input tanpa parameter trainable tambahan dapat memperbaiki utility citra bagi YOLO26 pada fine-grained coffee-defect detection? Literatur preprocessing di atas membuat pertanyaan tersebut **layak diuji**, tetapi tidak menetapkan jawabannya. Dasar matematis mengapa preprocessing yang diusulkan disebut *frekuensi-angular* dibahas pada Subbab 2.8.

---

## 2.8 Representasi Citra pada Domain Frekuensi

### 2.8.1 Discrete Fourier Transform dan Fast Fourier Transform

Citra dapat dipandang sebagai sinyal dua dimensi pada domain spasial dan direpresentasikan kembali sebagai kombinasi komponen frekuensi melalui *Discrete Fourier Transform* (DFT). Yang dan Soatto menggunakan formulasi DFT dua dimensi pada citra RGB dalam Fourier Domain Adaptation (FDA), dan juga menegaskan bahwa transformasi tersebut dapat dihitung secara efisien menggunakan FFT [PRE-08]. Untuk satu kanal citra diskrit \(f(x,y)\) berukuran \(M\times N\), bentuk umum DFT dapat dituliskan sebagai:

\[
F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)
\exp\left[-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right],
\qquad j^2=-1.
\]

Koefisien \(F(u,v)\) bersifat kompleks dan menyatakan respons citra pada indeks frekuensi \((u,v)\). Rekonstruksi kembali ke domain spasial dilakukan menggunakan inverse DFT:

\[
f(x,y)=\frac{1}{MN}\sum_{u=0}^{M-1}\sum_{v=0}^{N-1}F(u,v)
\exp\left[j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)\right].
\]

Dalam pipeline berbasis Fourier, transformasi maju dan balik memungkinkan suatu operasi dilakukan pada representasi spektral tanpa mengubah definisi koordinat spasial keluaran setelah citra direkonstruksi. FDA, misalnya, memodifikasi sebagian amplitude spectrum lalu menggunakan inverse Fourier transform untuk menghasilkan citra pada ruang spasial [PRE-08]. FE-YOLO juga memakai pola transform–process–reconstruct sebelum citra diteruskan ke detector [PRE-03].

DFT global dan DFT lokal tidak selalu mempunyai fungsi yang sama. Xu et al. pada LFDet secara khusus membandingkan pemrosesan global dengan *patch-wise DFT* untuk fine-grained aircraft detection. Menurut analisis dan ablation mereka, global frequency response mengabaikan variasi frekuensi menurut posisi dan kurang mampu mempertahankan detail lokal; karena itu AFAB membagi citra menjadi patch yang saling overlap, melakukan DFT pada setiap patch, lalu merekonstruksi hasilnya dengan patch-wise inverse DFT [FG-01]. Ini adalah temuan dan pilihan desain pada domain remote sensing mereka, bukan teorema bahwa patch-wise DFT selalu lebih baik pada semua domain. Dalam tesis ini, pilihan patch-wise processing diperlakukan sebagai mekanisme yang ditransfer dan kemudian diuji pada citra kopi.

Catatan sumber: persamaan aktif pada subbab ini telah dicocokkan dengan primary method papers [PRE-08][PRE-03]. Sebelum naskah tesis final, satu sumber textbook image/signal processing yang otoritatif (`THEORY-01` dan/atau `THEORY-02`) tetap akan dipasangkan untuk definisi fundamental DFT/FFT dan dinormalisasi menurut edisi/halaman yang benar.

### 2.8.2 Magnitudo/Amplitudo dan Fase Spektrum

Karena \(F(u,v)\) bernilai kompleks, koefisien Fourier dapat ditulis dalam bagian real dan imajiner:

\[
F(u,v)=R(u,v)+jI(u,v).
\]

FE-YOLO menuliskan amplitude dan phase dari koefisien tersebut sebagai [PRE-03]:

\[
A(u,v)=\sqrt{R^2(u,v)+I^2(u,v)},
\]

\[
\phi(u,v)=\operatorname{atan2}(I(u,v),R(u,v)).
\]

Dengan demikian, representasi Fourier bukan hanya pemisahan “frekuensi rendah” dan “frekuensi tinggi”, tetapi menyimpan magnitude/amplitude serta phase pada seluruh koordinat spektral. Dua komponen tersebut kemudian dapat diperlakukan berbeda oleh suatu metode. Dalam FDA, low-frequency amplitude dari source image diganti dengan amplitude dari target image sementara source phase dipertahankan sebelum inverse transform [PRE-08]. Pada FE-YOLO, amplitude dan phase diproses dalam Fourier Enhanced Network dan dikendalikan melalui amplitude-difference loss serta phase-similarity loss sebelum hasil enhancement digunakan oleh YOLO [PRE-03].

Kedua studi tersebut menunjukkan bahwa amplitude dan phase dapat dimanipulasi secara terkontrol untuk tujuan downstream yang berbeda. Namun interpretasinya harus dibatasi sesuai paper. FDA membahas domain adaptation untuk semantic segmentation, sedangkan FE-YOLO membahas low-light object detection. Karena itu, tesis ini tidak menggunakan keduanya untuk menyatakan bahwa amplitude atau phase telah terbukti menjadi bottleneck pada cacat kopi.

Pada parent method AFAB, Xu et al. juga memilih memodifikasi amplitude sambil mempertahankan original phase ketika merekonstruksi patch. Penulis menggunakan strategi tersebut agar intensitas respons frekuensi dapat diremodel tanpa mengganti distribusi spasial yang direpresentasikan phase pada formulasi mereka [FG-01]. Konsep inilah yang menghubungkan teori amplitude/phase dengan pemrosesan angular pada subbagian berikutnya.

### 2.8.3 Representasi Radial dan Angular pada Spektrum Fourier

Koordinat frekuensi dua dimensi dapat dianalisis dalam bentuk polar. Jika pusat spektrum dinyatakan sebagai \((u_c,v_c)\), maka secara konseptual radius dan sudut suatu koefisien dapat ditulis sebagai:

\[
r(u,v)=\sqrt{(u-u_c)^2+(v-v_c)^2},
\]

\[
\theta(u,v)=\operatorname{atan2}(v-v_c,u-u_c).
\]

Pemisahan ini memungkinkan spektrum diringkas dari dua sudut pandang yang berbeda. Distribusi radial mengelompokkan energi berdasarkan jarak dari pusat spektrum, sedangkan distribusi angular mengelompokkan energi berdasarkan arah. Untuk suatu magnitude spectrum \(A(u,v)\), contoh ringkasan energi dapat dituliskan secara konseptual sebagai:

\[
E_r(r_1,r_2)=
\sum_{r_1\le r(u,v)<r_2} A^2(u,v),
\]

\[
E_\theta(\theta_1,\theta_2)=
\sum_{\theta_1\le\theta(u,v)<\theta_2} A^2(u,v).
\]

Cao et al. menggunakan radial distribution dan angular distribution dari spectrum energy untuk menganalisis texture pada high-spatial-resolution imagery [SPEC-01]. Dalam eksperimen mereka, perubahan radial spectrum digunakan untuk membaca perubahan frekuensi/periodisitas dan skala texture, sedangkan angular spectrum merepresentasikan directionality texture; puncak distribusi angular berkaitan dengan arah dominan pola spektral [SPEC-01]. Paper tersebut menjadi basis teoritis bahwa “frekuensi” dan “arah” adalah dua dimensi analisis yang berbeda. Ia tidak membuktikan bahwa cacat kopi tertentu harus mempunyai signature angular tertentu.

Parent method Xu et al. kemudian mengoperasionalkan ide arah spektral langsung pada fine-grained detector. Untuk patch \(P_i\), mereka mendefinisikan angular density distribution dengan menjumlahkan amplitude sepanjang radius pada setiap sudut [FG-01]:

\[
D_i^P(\theta)
=
\sum_r A_i^P(r\cos\theta,r\sin\theta),
\qquad \theta\in[0,360^\circ).
\]

Pada AFAB-2, distribusi tersebut dinormalisasi, information entropy dihitung untuk memperoleh threshold adaptif per patch, dan arah dengan density rendah disupresi. Adjusted amplitude kemudian dipasangkan dengan original phase dan dikembalikan ke ruang spasial melalui inverse DFT [FG-01]. Penulis menginterpretasikan density angular tinggi sebagai arah yang mengandung struktur edge/texture lebih jelas pada data mereka, sementara respons rendah lebih mungkin memuat informasi yang kurang relevan atau noise. Karena evidence tersebut berasal dari aircraft remote sensing, interpretasi ini tidak boleh langsung diperlakukan sebagai karakteristik fisik cacat biji kopi.

Dengan demikian, istilah **frekuensi-angular** pada tesis ini memiliki definisi teknis yang spesifik: *frekuensi* merujuk pada representasi citra melalui respons spektral lokal, sedangkan *angular* merujuk pada distribusi amplitude menurut arah pada domain Fourier. “Angular” tidak merujuk pada rotasi bounding box atau oriented object detection.

### 2.8.4 Pemrosesan Frekuensi untuk Computer Vision dan Object Detection

Literatur frequency-aware vision dapat dibedakan menurut **lokasi operasi**. Pada input/data space, FDA memodifikasi low-frequency amplitude sebelum citra direkonstruksi, walaupun task akhirnya adalah semantic segmentation [PRE-08]. FE-YOLO menempatkan Fourier Enhanced Network sebelum YOLO dan memproses amplitude/phase secara learned untuk low-light detection [PRE-03]. Xu et al. menempatkan AFAB pada data space input dalam LFDet; patch-wise DFT, adaptive filtering, dan angular amplitude suppression menghasilkan ruang data yang telah diremodel sebelum backbone membentuk feature representation [FG-01]. Ketiga contoh ini relevan karena operasi spektralnya terjadi sebelum atau pada sumber data yang diterima feature extractor, tetapi tujuan, supervision, dan mekanismenya berbeda.

Pada feature space, pendekatan yang digunakan juga beragam. Fast Fourier Convolution (FFC) menghubungkan jalur local/spatial dan global/spectral sehingga informasi lokal dan konteks non-local dapat diproses secara komplementer di dalam jaringan [FREQ-01]. FDADNet menggunakan transformasi domain frekuensi bersama jalur spasial untuk low-contrast surface-defect detection pada wood-based panels; paper tersebut secara eksplisit memanfaatkan perbedaan sifat spatial detail dan global frequency representation [FREQ-02]. Frequency Dynamic Convolution (FDConv) memodulasi convolution secara adaptif pada domain frekuensi untuk dense prediction tasks [FREQ-03]. Wavelet-based operators seperti WTConv memisahkan komponen melalui transformasi wavelet di dalam feature-processing pipeline [WAVE-01]. Metode-metode ini tidak boleh disatukan menjadi satu kategori “high-frequency enhancement”, karena masing-masing bekerja pada representasi, lokasi, dan objective yang berbeda.

Dalam rangkaian literatur tersebut, Xu et al. merupakan **parent mechanism yang paling dekat** dengan metode tesis karena AFAB-2 memang menggunakan patch-wise Fourier response dan angular amplitude suppression untuk fine-grained object detection [FG-01]. Namun bahkan pada paper parent, kontribusi AFAB-2 harus dipisahkan dari keseluruhan LFDet. Ablation mereka menunjukkan bahwa AFAB-1 dan AFAB-2 adalah subkomponen yang berbeda, integrasi keduanya tidak selalu aditif pada baseline, dan full LFDet memperoleh kontribusi tambahan dari CGFI dan FTIF [FG-01]. Dengan kata lain, peningkatan full LFDet tidak boleh diklaim sebagai gain AFAB-2 saja.

Posisi penelitian ini adalah mentransfer prinsip **local frequency response + adaptive angular amplitude selection** ke sebuah frontend yang berdiri sendiri sebelum YOLO26. Secara konseptual:

\[
I
\xrightarrow{\text{overlapping patches}}
\{P_i\}
\xrightarrow{\mathcal{F}}
\{A_i,\phi_i\}
\xrightarrow{D_i(\theta),\;\tau_i}
\{\widetilde{A}_i,\phi_i\}
\xrightarrow{\mathcal{F}^{-1}}
R_{FA}
\xrightarrow{\text{residual image reconstruction}}
I'
\xrightarrow{\text{YOLO26}}
\hat Y.
\]

Frontend tersebut tidak menambahkan parameter trainable:

\[
\Theta_{AF2}^{\mathrm{trainable}}=\varnothing.
\]

Walaupun garis mekanismenya berasal dari AFAB-2, implementasi tesis tidak boleh digambarkan sebagai salinan identik LFDet. Xu et al. mengembangkan AFAB sebagai salah satu branch dalam sistem LFDet pada aircraft detection, sedangkan penelitian ini mengevaluasi adaptasi frequency-angular sebagai preprocessing standalone untuk YOLO26 pada coffee defect detection. Keputusan coffee-specific seperti bentuk residual reconstruction, integration point, training protocol, dan parameter implementasi harus dijelaskan sebagai **adaptasi penelitian ini** pada Bab III, bukan diatribusikan kepada Xu et al.

Rangkaian evidence pada Subbab 2.8 karena itu hanya mendukung tiga proposisi terbatas: (1) representasi Fourier dapat dimanipulasi dan direkonstruksi kembali ke image space [PRE-08][PRE-03]; (2) radial/angular spectrum dapat dipakai untuk menganalisis skala dan directionality texture [SPEC-01]; dan (3) frequency-aware processing telah digunakan pada fine-grained detection dan berbagai dense/defect vision tasks [FG-01][FREQ-01][FREQ-02][FREQ-03]. Tidak satu pun proposisi tersebut membuktikan bahwa coffee defects pasti frequency-separable. Efektivitas transfer tersebut tetap merupakan pertanyaan empiris tesis.

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