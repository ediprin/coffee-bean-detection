# Official Citation Audit — Proposal

Status: **HARD GATE untuk daftar pustaka formal**

Dokumen ini menentukan apakah sebuah sumber boleh masuk ke `docs/thesis/proposal/DAFTAR_PUSTAKA.md`. Master workbook, `CANONICAL_SOURCE_KEYS.md`, catatan percakapan, Google Scholar, DBLP, Crossref, Semantic Scholar, ResearchGate, OpenAIRE, atau metadata sekunder lain hanya berfungsi sebagai **locator/corroboration**, bukan sebagai otoritas final apabila halaman resmi penerbit/proceedings/standard body atau primary PDF tersedia.

## Aturan status

- **FINAL — OFFICIAL VERIFIED**: metadata bibliografis utama telah diverifikasi pada sumber resmi/primer. Boleh dipakai untuk membangun entri APA formal.
- **FINAL — PRIMARY + OFFICIAL LANDING VERIFIED**: primary PDF telah diperiksa dan official publisher/proceedings landing juga telah dikonfirmasi. Boleh dipakai, tetapi metadata final tetap mengikuti primary PDF + official landing, bukan aggregator.
- **CORROBORATED — OFFICIAL METADATA PENDING**: identitas paper/DOI kuat, tetapi official publisher record belum dapat dibaca lengkap. Belum dipromosikan ke daftar pustaka final.
- **PENDING — DO NOT CITE AS FINAL**: metadata resmi belum diverifikasi.

## Larangan

1. Jangan membuat DOI dari ingatan.
2. Jangan mengisi volume/issue/pages dari snippet sekunder bila sumber resmi belum diperiksa.
3. Jangan mengubah `et al.` pada sitasi teks menjadi daftar author lengkap berdasarkan tebakan.
4. Jangan menyebut Q1/Q2/SINTA/Scopus tanpa audit indeks yang relevan.
5. Jangan memasukkan sumber `PENDING` atau `CORROBORATED` ke `DAFTAR_PUSTAKA.md` sebagai entri final.
6. Klaim metodologis tetap harus dibaca dari full text paper primer; metadata resmi saja tidak cukup untuk membuktikan mekanisme.
7. Jika master workbook bertentangan dengan publisher resmi, **publisher/primary paper menang** dan discrepancy wajib dicatat.

---

## A. Sumber yang sudah lolos official-source gate

| Key | Status | Sumber resmi/primer yang dikunci | Metadata aman |
|---|---|---|---|
| STD-01 | FINAL — OFFICIAL VERIFIED | BSN Pesta Online | Badan Standardisasi Nasional. **SNI 2907:2008 — Biji kopi**. Status berlaku. |
| COF-17 | FINAL — OFFICIAL VERIFIED | MDPI Applied Sciences | Mauricio García, John E. Candelo-Becerra, Fredy E. Hoyos. **Quality and Defect Inspection of Green Coffee Beans Using a Computer Vision System**. *Applied Sciences*, 9(19), 4195 (2019). DOI `10.3390/app9194195`. |
| COF-01 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | **Automated detection of defective coffee beans based on improved YOLOv10 framework**. *Current Research in Food Science* (2026). DOI `10.1016/j.crfs.2026.101461`. |
| COF-02 | FINAL — OFFICIAL VERIFIED | IJoICT official article + official PDF | Nanda Aptana Irsyadul Bahy; Achmad Pratama Rifai. **Real-Time Coffee Bean Defect Detection Based on SNI 01-2907-2008 Standards Using Lightweight YOLOv5s Architecture**. *IJoICT*, 12(1), 29–42 (2026). DOI `10.21108/ijoict.v12i1.10584`. |
| COF-04 | FINAL — OFFICIAL VERIFIED | INOVTEK Polbeng official journal page | Hocwin Hebert; Derry Alamsyah. **Detection of Coffee Bean Defects in Speciality Coffee Association Standards using YOLOv12**. *INOVTEK Polbeng - Seri Informatika*, 11(1), 85–95 (2026). DOI `10.35314/47yqwd13`. |
| COF-05 | FINAL — OFFICIAL VERIFIED | Brilliance official journal page | Sayid Muhammad Jundullah; Hafizh Al Kautsar Aidilof; Fadlisyah. **YOLOv8-Based Multi-Class Detection of Coffee Bean Defects and Contaminants for Automated Quality Grading**. *Brilliance: Research of Artificial Intelligence*, 6(2), 313–322 (2026). DOI `10.47709/brilliance.v6i2.8612`. |
| COF-06 | FINAL — OFFICIAL VERIFIED | Nature / Scientific Reports | **Comparative analysis of YOLO models for green coffee bean detection and defect classification**. *Scientific Reports*, 14, 28946 (2024). DOI `10.1038/s41598-024-78598-7`. |
| COF-07 | FINAL — PRIMARY + OFFICIAL LANDING VERIFIED | Primary conference PDF + IEEE Xplore official landing | Made Windu Antara Kesiman; Ismail Sulaiman; I Made Dendi Maysanjaya; Kadek Teguh Dermawan. **Benchmarking A New Dataset for Coffee Bean Defects Classification Based on SNI 01-2907-2008**. ICITRI 2023. DOI `10.1109/ICITRI59340.2023.10249345`. |
| COF-08 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | **Implementing a deep learning model for defect classification in Thai Arabica green coffee beans**. *Smart Agricultural Technology*, 9, 100680 (2024). DOI `10.1016/j.atech.2024.100680`. |
| COF-10 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Emanuelle Morais de Oliveira; Dimas Samid Leme; Bruno Henrique Groenner Barbosa; Mirian Pereira Rodarte; Rosemary Gualberto Fonseca Alvarenga Pereira. **A computer vision system for coffee beans classification based on computational intelligence techniques**. *Journal of Food Engineering*, 171, 22–27 (2016). DOI `10.1016/j.jfoodeng.2015.10.009`. |
| COF-12 | FINAL — OFFICIAL VERIFIED | PLOS official article/PDF | Yujie Jiao et al. **Swin-HSSAM: A green coffee bean grading method by Swin transformer**. *PLOS ONE*, 20(5), e0322198 (2025). DOI `10.1371/journal.pone.0322198`. |
| COF-13 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Xingran Hu; Jun He; Xinyu Guo; Sunyan Hong; Jing Yu. **Siamese networks for few-shot coffee bean defect detection**. *LWT*, 235, 118631 (2025). DOI `10.1016/j.lwt.2025.118631`. |
| FG-01 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Xueru Xu; Zhong Chen; Yuxin Hu; Guoyou Wang. **More signals matter to detection: Integrating language knowledge and frequency representations for boosting fine-grained aircraft recognition**. *Neural Networks*, 187, 107402 (2025). DOI `10.1016/j.neunet.2025.107402`. |
| PRE-01 | FINAL — OFFICIAL VERIFIED | AAAI official proceedings/PDF | Wenyu Liu et al. **Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions**. AAAI 2022. DOI `10.1609/aaai.v36i2.20072`. |
| PRE-02 | FINAL — OFFICIAL VERIFIED | CVF ACCV 2022 Open Access | Qingpao Qin; Kan Chang; Mengyuan Huang; Guiqing Li. **DENet: Detection-driven Enhancement Network for Object Detection under Adverse Weather Conditions**. ACCV 2022, 2813–2829. |
| PRE-03 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Yang Li; Xianguo Li; Michael Lin. **FE-YOLO: Fourier enhancement YOLO for end-to-end object detection in low-light conditions**. *Digital Signal Processing*, 166, 105355 (2025). DOI `10.1016/j.dsp.2025.105355`. |
| PRE-05 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Siyu Chen; Yixuan Li; Yidong Zhang; Yifan Yang; Xiangxue Zhang. **Soft X-ray image recognition and classification of maize seed cracks based on image enhancement and optimized YOLOv8 model**. *Computers and Electronics in Agriculture*, 216, 108475 (2024). DOI `10.1016/j.compag.2023.108475`. |
| PRE-08 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Yanchao Yang; Stefano Soatto. **FDA: Fourier Domain Adaptation for Semantic Segmentation**. CVPR 2020. |
| SPEC-01 | FINAL — OFFICIAL VERIFIED | Wiley / Journal of Spectroscopy | Cao et al. **Frequency Spectrum-Based Optimal Texture Window Size Selection for High Spatial Resolution Remote Sensing Image Analysis**. *Journal of Spectroscopy* (2019), article 4970376. DOI `10.1155/2019/4970376`. |
| SPEC-02 | FINAL — OFFICIAL VERIFIED | ScienceDirect / Elsevier | Jianguo Zhang; Tieniu Tan. **Affine invariant classification and retrieval of texture images**. *Pattern Recognition*, 36(3), 657–664 (2003). DOI `10.1016/S0031-3203(02)00099-7`. |
| FREQ-01 | FINAL — OFFICIAL VERIFIED | NeurIPS official proceedings | Lu Chi; Borui Jiang; Yadong Mu. **Fast Fourier Convolution**. NeurIPS 2020. |
| FREQ-02 | FINAL — OFFICIAL VERIFIED | MDPI Processes | Hongli Li et al. **FDADNet: Detection of Surface Defects in Wood-Based Panels Based on Frequency Domain Transformation and Adaptive Dynamic Downsampling**. *Processes*, 12(10), 2134 (2024). DOI `10.3390/pr12102134`. |
| FREQ-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Linwei Chen; Lin Gu; Liang Li; Chenggang Yan; Ying Fu. **Frequency Dynamic Convolution for Dense Image Prediction**. CVPR 2025. |
| DET-01 | FINAL — PRIMARY PREPRINT VERIFIED | arXiv primary preprint | Glenn Jocher; Jing Qiu; Mengyu Liu; Shuai Lyu; Fatih Cagatay Akyon; Muhammet Esat Kalfaoglu. **Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models**. arXiv:2606.03748 (2026). DOI `10.48550/arXiv.2606.03748`. |
| DET-02 | FINAL — OFFICIAL VERIFIED | NeurIPS official proceedings | Shaoqing Ren; Kaiming He; Ross Girshick; Jian Sun. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**. NeurIPS 2015. |
| DET-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Joseph Redmon; Santosh Divvala; Ross Girshick; Ali Farhadi. **You Only Look Once: Unified, Real-Time Object Detection**. CVPR 2016. |
| DIAG-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Chengjian Feng; Yujie Zhong; Yu Gao; Matthew R. Scott; Weilin Huang. **TOOD: Task-Aligned One-Stage Object Detection**. ICCV 2021. |
| DIAG-02 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Yue Wu et al. **Rethinking Classification and Localization for Object Detection**. CVPR 2020. |
| DIAG-03 | FINAL — OFFICIAL VERIFIED | ECCV/Springer official proceedings | Borui Jiang; Ruixuan Luo; Jiayuan Mao; Tete Xiao; Yuning Jiang. **Acquisition of Localization Confidence for Accurate Object Detection**. ECCV 2018. |
| EVAL-01 | FINAL — OFFICIAL VERIFIED | Springer Nature | Tsung-Yi Lin et al. **Microsoft COCO: Common Objects in Context**. ECCV 2014, 740–755. DOI `10.1007/978-3-319-10602-1_48`. |
| XAI-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Ramprasaath R. Selvaraju et al. **Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization**. ICCV 2017, 618–626. DOI `10.1109/ICCV.2017.74`. |

### Metadata discrepancy yang ditemukan

**COF-10 / de Oliveira et al. (2016):** master workbook saat ini mencantumkan DOI `10.1016/j.jfoodeng.2015.10.030`. Halaman resmi ScienceDirect menunjukkan DOI yang benar adalah:

`10.1016/j.jfoodeng.2015.10.009`

DOI workbook yang lama **tidak boleh dipakai** dalam daftar pustaka final.

**PRE-05 / Chen et al. (2024):** tahun sitasi adalah 2024 karena artikel berada pada *Computers and Electronics in Agriculture*, volume 216 (January 2024), tetapi DOI resminya mengandung tahun 2023: `10.1016/j.compag.2023.108475`. Ini bukan kontradiksi dan DOI tidak boleh "dikoreksi" menjadi 2024.

---

## B. Sumber yang belum lolos final gate

| Key | Status | Bukti yang sudah ada | Yang masih kurang |
|---|---|---|---|
| COF-03 | CORROBORATED — OFFICIAL METADATA PENDING | Primary project PDF; DOI `10.1109/ICoDSA67155.2025.11157423`; IEEE official landing dapat diakses tetapi metadata halaman terblokir JS; conference record konsisten. | Kunci metadata final dari IEEE Xplore export/record resmi sebelum APA final. |
| FG-02 | CORROBORATED — OFFICIAL METADATA PENDING | DOI `10.1109/TCSVT.2025.3544741`; volume 35(8), 8197–8208 dikorroborasi; publisher IEEE. | Baca official IEEE Xplore record/export sebelum APA final. |
| PRE-04 | CORROBORATED — OFFICIAL METADATA PENDING | Conference official session confirms title/presenter; author repository gives DOI `10.1109/ICONS-IoT65216.2025.11211242`; SINTA/secondary records consistent. | IEEE Xplore official bibliographic record. |
| XAI-02 | CORROBORATED — OFFICIAL METADATA PENDING | Primary arXiv `1710.11063`; DOI `10.1109/WACV.2018.00097`; authors/pages corroborated. | IEEE Xplore official bibliographic export/record. |
| XAI-03 | CORROBORATED — OFFICIAL METADATA PENDING | Primary arXiv `2008.00299`; DOI `10.1109/IJCNN48605.2020.9206626`; authors/pages corroborated. | IEEE Xplore official bibliographic export/record. |

Sumber lain yang muncul pada artefak formal tetapi belum tercantum pada tabel A otomatis berstatus **PENDING** sampai diaudit.

---

## C. Gate pembangunan `DAFTAR_PUSTAKA.md`

`DAFTAR_PUSTAKA.md` **belum boleh dianggap final** sampai langkah berikut selesai:

1. ekstrak seluruh sitasi author-year yang benar-benar muncul di `BAB_I_PENDAHULUAN.md`, `BAB_II_TINJAUAN_PUSTAKA.md`, dan `BAB_III_METODOLOGI_PENELITIAN.md`;
2. cocokkan setiap sitasi dengan satu key canonical;
3. pastikan setiap key memiliki status `FINAL`;
4. kunci author spelling, year, exact title, venue, volume/issue/article number/pages, dan DOI dari sumber primer/resmi;
5. bentuk entri APA;
6. lakukan audit dua arah: **cited → bibliography** dan **bibliography → cited**.

Jika sebuah sumber tidak dapat diverifikasi secara resmi, pilihannya hanya dua: **hapus/ganti klaim yang bergantung padanya** atau **tetap tandai pending dan jangan menganggap proposal citation-ready**.
