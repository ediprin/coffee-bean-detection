# Official Citation Audit — Proposal

Status: **HARD GATE untuk daftar pustaka formal**

Dokumen ini menentukan apakah sebuah sumber boleh masuk ke `docs/thesis/proposal/DAFTAR_PUSTAKA.md`. Master workbook, `CANONICAL_SOURCE_KEYS.md`, catatan percakapan, Google Scholar, DBLP, Crossref, Semantic Scholar, ResearchGate, OpenAIRE, atau metadata sekunder lain hanya berfungsi sebagai **locator/corroboration**, bukan sebagai otoritas final apabila halaman resmi penerbit/proceedings/standard body atau primary PDF tersedia.

Pemetaan sitasi author–year yang benar-benar muncul pada BAB I–III terhadap key canonical dan status gate disimpan pada `CITATION_CROSSWALK.md`.

## Aturan status

- **FINAL — OFFICIAL VERIFIED**: metadata bibliografis utama telah diverifikasi pada sumber resmi penerbit/proceedings/standard body.
- **FINAL — PRIMARY PUBLISHER PDF VERIFIED**: PDF primer yang berasal dari penerbit/proceedings resmi telah diperiksa dan memuat metadata bibliografis yang diperlukan.
- **FINAL — PRIMARY PREPRINT VERIFIED**: paper primer pada repositori preprint resmi telah diperiksa. Jika dipakai di daftar pustaka, status preprint harus ditulis apa adanya.
- **OFFICIAL PUBLISHER METADATA VERIFIED**: metadata resmi penerbit telah dikunci untuk sumber teori/buku; klaim halaman spesifik tetap memerlukan halaman sumber bila diperlukan.
- **PRIMARY VERIFIED / PUBLISHER METADATA PENDING**: full text primer dan identitas paper telah diverifikasi, tetapi sebagian metadata final publisher masih belum tertutup.
- **PENDING — DO NOT CITE AS FINAL**: metadata primer/resmi belum memadai.

## Larangan

1. Jangan membuat DOI dari ingatan.
2. Jangan mengisi volume/issue/pages dari snippet sekunder bila source primer/resmi belum diperiksa.
3. Jangan mengubah `et al.` menjadi daftar author lengkap berdasarkan tebakan.
4. Jangan menyebut Q1/Q2/SINTA/Scopus tanpa audit indeks yang relevan.
5. Jangan menyamakan metadata aggregator dengan metadata publisher.
6. Klaim metodologis harus dibaca dari full text primer; metadata saja tidak membuktikan mekanisme.
7. Jika master workbook bertentangan dengan publisher/primary paper, **publisher/primary paper menang** dan discrepancy harus dicatat.

---

## A. Sumber yang sudah lolos gate primer/resmi

| Key | Status | Sumber yang dikunci | Metadata aman / batas penggunaan |
|---|---|---|---|
| STD-01 | FINAL — OFFICIAL VERIFIED | BSN Pesta Online | Badan Standardisasi Nasional. **SNI 2907:2008 — Biji kopi**. Status berlaku. |
| COF-17 | FINAL — OFFICIAL VERIFIED | MDPI Applied Sciences | Mauricio García, John E. Candelo-Becerra, Fredy E. Hoyos. **Quality and Defect Inspection of Green Coffee Beans Using a Computer Vision System**. *Applied Sciences*, 9(19), 4195 (2019). DOI `10.3390/app9194195`. |
| COF-01 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | **Automated detection of defective coffee beans based on improved YOLOv10 framework**. *Current Research in Food Science* (2026). DOI `10.1016/j.crfs.2026.101461`. |
| COF-02 | FINAL — OFFICIAL VERIFIED | IJoICT official article/PDF | Nanda Aptana Irsyadul Bahy; Achmad Pratama Rifai. **Real-Time Coffee Bean Defect Detection Based on SNI 01-2907-2008 Standards Using Lightweight YOLOv5s Architecture**. *IJoICT*, 12(1), 29–42 (2026). DOI `10.21108/ijoict.v12i1.10584`. |
| COF-03 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE Xplore-downloaded primary PDF | Melyna Nura Samudra; Ema Rachmawati. **Deep Learning-Based Defect Detection in Arabica Green Coffee Beans Using LSKNet**. ICoDSA 2025, pp. 692–697. Backend source tetap valid, tetapi saat ini tidak disitasi pada naskah formal. |
| COF-04 | FINAL — OFFICIAL VERIFIED | INOVTEK Polbeng official page | Hocwin Hebert; Derry Alamsyah. **Detection of Coffee Bean Defects in Speciality Coffee Association Standards using YOLOv12**. *INOVTEK Polbeng - Seri Informatika*, 11(1), 85–95 (2026). DOI `10.35314/47yqwd13`. |
| COF-05 | FINAL — OFFICIAL VERIFIED | Brilliance official page | Sayid Muhammad Jundullah; Hafizh Al Kautsar Aidilof; Fadlisyah. **YOLOv8-Based Multi-Class Detection of Coffee Bean Defects and Contaminants for Automated Quality Grading**. *Brilliance: Research of Artificial Intelligence*, 6(2), 313–322 (2026). DOI `10.47709/brilliance.v6i2.8612`. |
| COF-06 | FINAL — OFFICIAL VERIFIED | Nature / Scientific Reports | **Comparative analysis of YOLO models for green coffee bean detection and defect classification**. *Scientific Reports*, 14, 28946 (2024). DOI `10.1038/s41598-024-78598-7`. |
| COF-07 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | Primary conference PDF + IEEE record | Made Windu Antara Kesiman; Ismail Sulaiman; I Made Dendi Maysanjaya; Kadek Teguh Dermawan. **Benchmarking A New Dataset for Coffee Bean Defects Classification Based on SNI 01-2907-2008**. ICITRI 2023. DOI `10.1109/ICITRI59340.2023.10249345`. |
| COF-08 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | **Implementing a deep learning model for defect classification in Thai Arabica green coffee beans**. *Smart Agricultural Technology*, 9, 100680 (2024). DOI `10.1016/j.atech.2024.100680`. |
| COF-10 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Emanuelle Morais de Oliveira et al. **A computer vision system for coffee beans classification based on computational intelligence techniques**. *Journal of Food Engineering*, 171, 22–27 (2016). DOI `10.1016/j.jfoodeng.2015.10.009`. |
| COF-12 | FINAL — OFFICIAL VERIFIED | PLOS official article/PDF | Yujie Jiao et al. **Swin-HSSAM: A green coffee bean grading method by Swin transformer**. *PLOS ONE*, 20(5), e0322198 (2025). DOI `10.1371/journal.pone.0322198`. |
| COF-13 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Xingran Hu; Jun He; Xinyu Guo; Sunyan Hong; Jing Yu. **Siamese networks for few-shot coffee bean defect detection**. *LWT*, 235, 118631 (2025). DOI `10.1016/j.lwt.2025.118631`. |
| COF-18 | FINAL — OFFICIAL VERIFIED | Tech Science Press / Journal on Artificial Intelligence | Tesfaye Adisu Tarekegn; Taye Girma Debelee. **KN-YOLOv8: A Lightweight Deep Learning Model for Real-Time Coffee Bean Defect Detection**. *Journal on Artificial Intelligence*, 7(1), 585–613 (2025). DOI `10.32604/jai.2025.067333`. Official article reports a custom dataset of 562 images; detailed instance statistics used in methodology must follow the primary full text. |
| FG-01 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Xueru Xu; Zhong Chen; Yuxin Hu; Guoyou Wang. **More signals matter to detection: Integrating language knowledge and frequency representations for boosting fine-grained aircraft recognition**. *Neural Networks*, 187, 107402 (2025). DOI `10.1016/j.neunet.2025.107402`. |
| FG-02 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE TCSVT publisher-format PDF | Xingxing Xie et al. **Learning Discriminative Representation for Fine-Grained Object Detection in Remote Sensing Images**. *IEEE TCSVT*, 35(8), 8197–8208 (2025). DOI `10.1109/TCSVT.2025.3544741`. |
| PRE-01 | FINAL — OFFICIAL VERIFIED | AAAI official proceedings/PDF | Wenyu Liu et al. **Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions**. AAAI 2022. DOI `10.1609/aaai.v36i2.20072`. |
| PRE-02 | FINAL — OFFICIAL VERIFIED | CVF ACCV 2022 Open Access | Qingpao Qin; Kan Chang; Mengyuan Huang; Guiqing Li. **DENet: Detection-driven Enhancement Network for Object Detection under Adverse Weather Conditions**. ACCV 2022, 2813–2829. |
| PRE-03 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Yang Li; Xianguo Li; Michael Lin. **FE-YOLO: Fourier enhancement YOLO for end-to-end object detection in low-light conditions**. *Digital Signal Processing*, 166, 105355 (2025). DOI `10.1016/j.dsp.2025.105355`. |
| PRE-04 | FINAL — PRIMARY PUBLISHER PDF VERIFIED | IEEE Xplore-downloaded PDF | Faturrahman Syauqi et al. **Edge AI-Based Defect Detection in White Pepper (Piper nigrum L.) Using CLAHE-Based Pre-processing and YOLO**. ICONS-IoT 2025, pp. 18–23. DOI `10.1109/ICONS-IOT65216.2025.11211242`. Metode adalah pipeline komposit; jangan mengatribusikan hasil kepada CLAHE saja. |
| PRE-05 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Siyu Chen et al. **Soft X-ray image recognition and classification of maize seed cracks based on image enhancement and optimized YOLOv8 model**. *Computers and Electronics in Agriculture*, 216, 108475 (2024). DOI `10.1016/j.compag.2023.108475`. |
| PRE-08 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Yanchao Yang; Stefano Soatto. **FDA: Fourier Domain Adaptation for Semantic Segmentation**. CVPR 2020. |
| THEORY-01 | OFFICIAL PUBLISHER METADATA VERIFIED | Pearson | Rafael C. Gonzalez; Richard E. Woods. **Digital Image Processing**, 4th ed., Global Edition (2018). Digunakan sebagai landasan teori DFT/FFT/amplitudo/fase; nomor halaman formula tidak ditulis tanpa halaman sumber yang tersedia. |
| SPEC-01 | FINAL — OFFICIAL VERIFIED | Publisher official record | Cao et al. **Frequency Spectrum-Based Optimal Texture Window Size Selection for High Spatial Resolution Remote Sensing Image Analysis**. *Journal of Spectroscopy* (2019), 4970376. DOI `10.1155/2019/4970376`. |
| SPEC-02 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Jianguo Zhang; Tieniu Tan. **Affine invariant classification and retrieval of texture images**. *Pattern Recognition*, 36(3), 657–664 (2003). DOI `10.1016/S0031-3203(02)00099-7`. |
| FREQ-01 | FINAL — OFFICIAL VERIFIED | NeurIPS official proceedings | Lu Chi; Borui Jiang; Yadong Mu. **Fast Fourier Convolution**. NeurIPS 2020. |
| FREQ-02 | FINAL — OFFICIAL VERIFIED | MDPI Processes | Hongli Li et al. **FDADNet: Detection of Surface Defects in Wood-Based Panels Based on Frequency Domain Transformation and Adaptive Dynamic Downsampling**. *Processes*, 12(10), 2134 (2024). DOI `10.3390/pr12102134`. |
| FREQ-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Linwei Chen; Lin Gu; Liang Li; Chenggang Yan; Ying Fu. **Frequency Dynamic Convolution for Dense Image Prediction**. CVPR 2025. |
| DET-01 | FINAL — PRIMARY PREPRINT VERIFIED | arXiv primary paper | Glenn Jocher et al. **Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models**. arXiv:2606.03748 (2026). DOI `10.48550/arXiv.2606.03748`. |
| DET-02 | FINAL — OFFICIAL VERIFIED | NeurIPS official proceedings | Shaoqing Ren; Kaiming He; Ross Girshick; Jian Sun. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**. NeurIPS 2015. |
| DET-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Joseph Redmon; Santosh Divvala; Ross Girshick; Ali Farhadi. **You Only Look Once: Unified, Real-Time Object Detection**. CVPR 2016. |
| DET-04 | FINAL — OFFICIAL VERIFIED | CVF WACV 2025 Open Access | Shuo Wang; Chunlong Xia; Feng Lv; Yifeng Shi. **RT-DETRv3: Real-Time End-to-End Object Detection with Hierarchical Dense Positive Supervision**. WACV 2025, pp. 1628–1636. Digunakan sebagai dasar evaluasi lintas arsitektur yang bersifat opsional. |
| DIAG-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Chengjian Feng et al. **TOOD: Task-Aligned One-Stage Object Detection**. ICCV 2021. |
| DIAG-02 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Yue Wu et al. **Rethinking Classification and Localization for Object Detection**. CVPR 2020. |
| DIAG-03 | FINAL — OFFICIAL VERIFIED | ECCV/Springer proceedings | Borui Jiang et al. **Acquisition of Localization Confidence for Accurate Object Detection**. ECCV 2018. |
| EVAL-01 | FINAL — OFFICIAL VERIFIED | Springer Nature | Tsung-Yi Lin et al. **Microsoft COCO: Common Objects in Context**. ECCV 2014, 740–755. Backend source valid, tetapi saat ini tidak disitasi pada naskah formal. |
| XAI-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Ramprasaath R. Selvaraju et al. **Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization**. ICCV 2017, 618–626. DOI `10.1109/ICCV.2017.74`. |
| XAI-03 | FINAL — PRIMARY PREPRINT VERIFIED | arXiv primary paper | Mohammed Bany Muhammad; Mohammed Yeasin. **Eigen-CAM: Class Activation Map using Principal Components** (2020). Jika digunakan di bibliography, status preprint harus ditulis transparan. |

### Metadata discrepancies yang ditemukan

**COF-10 / de Oliveira et al. (2016):** DOI resmi adalah `10.1016/j.jfoodeng.2015.10.009`; DOI workbook lama yang berakhiran `.030` tidak boleh dipakai.

**PRE-05 / Chen et al. (2024):** tahun artikel adalah 2024, sedangkan DOI resmi mengandung string `2023`: `10.1016/j.compag.2023.108475`. DOI tidak boleh diubah.

---

## B. Sumber yang masih belum tertutup sepenuhnya

| Key | Status | Bukti yang sudah ada | Keputusan aman |
|---|---|---|---|
| XAI-02 | PRIMARY VERIFIED / OPTIONAL CITATION | Primary author preprint memuat identitas dan metodologi Grad-CAM++. | Tidak disitasi eksplisit pada proposal formal saat ini; tidak masuk daftar pustaka. |

---

## C. Gate pembangunan `DAFTAR_PUSTAKA.md`

Untuk snapshot proposal saat ini, set formal telah disinkronkan melalui `CITATION_CROSSWALK.md` dan `BIDIRECTIONAL_CITATION_AUDIT.md`.

Kondisi yang berlaku:

1. hanya sumber yang benar-benar disitasi pada BAB I–III yang masuk `DAFTAR_PUSTAKA.md`;
2. sumber backend yang telah diverifikasi tetapi tidak disitasi, seperti `COF-03`, `EVAL-01`, dan `XAI-02`, tetap boleh disimpan pada audit repository tetapi tidak masuk bibliography formal;
3. metadata baru tetap harus berasal dari sumber resmi/primer;
4. klaim metodologis sensitif tetap memerlukan full text primer meskipun metadata bibliografis telah lolos gate;
5. perubahan naskah harus memicu audit ulang crosswalk dan audit dua arah.

Jika sebuah sumber tidak dapat diverifikasi, pilihannya hanya menghapus/mengganti klaim yang bergantung padanya atau menandai proposal belum citation-ready; metadata tidak boleh ditebak.