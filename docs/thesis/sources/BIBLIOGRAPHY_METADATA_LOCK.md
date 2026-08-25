# Bibliography Metadata Lock — Proposal

Status: **ACTIVE HARD GATE — metadata APA dikunci hanya dari sumber resmi/primer**

Dokumen ini adalah lapisan terakhir sebelum `docs/thesis/proposal/DAFTAR_PUSTAKA.md` dibangun. Setiap baris mewakili sumber yang benar-benar masih disitasi pada BAB I–III formal. Metadata tidak boleh dilengkapi dari ingatan, Google Scholar, DBLP, Crossref, Semantic Scholar, atau workbook jika source publisher/proceedings/standard body/primary paper tersedia.

## Aturan

1. `LOCKED` berarti field yang tertulis telah diperiksa pada source resmi/primer.
2. `PARTIAL` berarti identitas sumber aman, tetapi satu atau lebih field APA masih belum dikunci; field tersebut harus dibiarkan kosong, bukan ditebak.
3. Untuk conference paper, gunakan **satu konvensi source** secara konsisten. Jangan mencampur year/pages dari CVF Open Access dengan DOI/pages versi Springer/IEEE jika metadata tersebut berasal dari edisi yang berbeda.
4. Untuk preprint, tulis sebagai preprint. Jangan menyamarkan arXiv menjadi artikel jurnal/proceedings.
5. `DAFTAR_PUSTAKA.md` hanya boleh memuat sumber dengan metadata yang cukup untuk membentuk entri APA tanpa tebakan.

---

## A. Coffee / domain sources

### STD-01 — LOCKED
- Author/organization: Badan Standardisasi Nasional
- Year: 2008
- Title: *SNI 2907:2008 Biji kopi*
- Type: Standar Nasional Indonesia
- Authority: BSN / Pesta Online
- Catatan: bentuk `SNI 01-2907-2008` boleh muncul hanya bila merupakan bagian persis dari judul paper terdahulu; nama standar pada narasi proposal dinormalisasi menjadi `SNI 2907:2008`.

### COF-17 — LOCKED
- Authors: Mauricio García; John E. Candelo-Becerra; Fredy E. Hoyos
- Year: 2019
- Title: *Quality and Defect Inspection of Green Coffee Beans Using a Computer Vision System*
- Journal: *Applied Sciences*
- Volume/issue: 9(19)
- Article number: 4195
- DOI: `10.3390/app9194195`
- Authority: MDPI official article page

### COF-01 — LOCKED
- Authors: Sunyan Hong; Dengji Zhang; Haiyang Chi; Jun He; Xiudong Guo; Hui Fu; Lu Wang
- Year: 2026
- Title: *Automated detection of defective coffee beans based on improved YOLOv10 framework*
- Journal: *Current Research in Food Science*
- Volume: 13
- Article number: 101461
- DOI: `10.1016/j.crfs.2026.101461`
- Authority: Elsevier/ScienceDirect; author list cross-checked on PubMed/PMC record

### COF-02 — LOCKED
- Authors: Nanda Aptana Irsyadul Bahy; Achmad Pratama Rifai
- Year: 2026
- Title: *Real-Time Coffee Bean Defect Detection Based on SNI 01-2907-2008 Standards Using Lightweight YOLOv5s Architecture*
- Journal: *IJoICT*
- Volume/issue: 12(1)
- Pages: 29–42
- DOI: `10.21108/ijoict.v12i1.10584`
- Authority: official journal article/PDF

### COF-04 — LOCKED
- Authors: Hocwin Hebert; Derry Alamsyah
- Year: 2026
- Title: *Detection of Coffee Bean Defects in Speciality Coffee Association Standards using YOLOv12*
- Journal: *INOVTEK Polbeng - Seri Informatika*
- Volume/issue: 11(1)
- Pages: 85–95
- DOI: `10.35314/47yqwd13`
- Authority: official journal article page

### COF-05 — LOCKED
- Authors: Sayid Muhammad Jundullah; Hafizh Al Kautsar Aidilof; Fadlisyah
- Year: 2026
- Title: *YOLOv8-Based Multi-Class Detection of Coffee Bean Defects and Contaminants for Automated Quality Grading*
- Journal: *Brilliance: Research of Artificial Intelligence*
- Volume/issue: 6(2)
- Pages: 313–322
- DOI: `10.47709/brilliance.v6i2.8612`
- Authority: official journal article page

### COF-06 — LOCKED
- Authors: Hira Lal Gope; Hidekazu Fukai; Fahim Mahafuz Ruhad; Shohag Barman
- Year: 2024
- Title: *Comparative analysis of YOLO models for green coffee bean detection and defect classification*
- Journal: *Scientific Reports*
- Volume: 14
- Article number: 28946
- DOI: `10.1038/s41598-024-78598-7`
- Authority: Nature / Scientific Reports official article page

### COF-07 — LOCKED
- Authors: Made Windu Antara Kesiman; Ismail Sulaiman; I Made Dendi Maysanjaya; Kadek Teguh Dermawan
- Year: 2023
- Title: *Benchmarking A New Dataset for Coffee Bean Defects Classification Based on SNI 01-2907-2008*
- Proceedings: *2023 International Conference on Information Technology Research and Innovation (ICITRI)*
- Pages: 75–80
- DOI: `10.1109/ICITRI59340.2023.10249345`
- Authority: primary IEEE publisher-format paper / IEEE-linked record

### COF-08 — LOCKED
- Authors: Sujitra Arwatchananukul; Dan Xu; Phasit Charoenkwan; Sai Aung Moon; Rattapon Saengrayap
- Year: 2024
- Title: *Implementing a deep learning model for defect classification in Thai Arabica green coffee beans*
- Journal: *Smart Agricultural Technology*
- Volume: 9
- Article number: 100680
- DOI: `10.1016/j.atech.2024.100680`
- Authority: Elsevier/ScienceDirect + primary publisher PDF

### COF-10 — LOCKED
- Authors: Emanuelle Morais de Oliveira; Dimas Samid Leme; Bruno Henrique Groenner Barbosa; Mirian Pereira Rodarte; Rosemary Gualberto Fonseca Alvarenga Pereira
- Year: 2016
- Title: *A computer vision system for coffee beans classification based on computational intelligence techniques*
- Journal: *Journal of Food Engineering*
- Volume: 171
- Pages: 22–27
- DOI: `10.1016/j.jfoodeng.2015.10.009`
- Authority: Elsevier/ScienceDirect
- **Discrepancy closed:** DOI workbook lama yang berakhiran `.030` salah dan dilarang dipakai.

### COF-12 — LOCKED
- Authors: Yujie Jiao; Yuqing Zhao; Aoying Jia; Tianyun Wang; Jiashun Li; Kaiming Xiang; Hangyu Deng; Maochang He; Rui Jiang; Yue Zhang
- Year: 2025
- Title: *Swin-HSSAM: A green coffee bean grading method by Swin transformer*
- Journal: *PLOS ONE*
- Volume/issue: 20(5)
- Article number: e0322198
- DOI: `10.1371/journal.pone.0322198`
- Authority: PLOS official article page/PDF

### COF-13 — LOCKED
- Authors: Xingran Hu; Jun He; Xinyu Guo; Sunyan Hong; Jing Yu
- Year: 2025
- Title: *Siamese networks for few-shot coffee bean defect detection*
- Journal: *LWT*
- Volume: 235
- Article number: 118631
- DOI: `10.1016/j.lwt.2025.118631`
- Authority: Elsevier/ScienceDirect

---

## B. Fine-grained / preprocessing / frequency sources

### FG-01 — LOCKED
- Authors: Xueru Xu; Zhong Chen; Yuxin Hu; Guoyou Wang
- Year: 2025
- Title: *More signals matter to detection: Integrating language knowledge and frequency representations for boosting fine-grained aircraft recognition*
- Journal: *Neural Networks*
- Volume: 187
- Article number: 107402
- DOI: `10.1016/j.neunet.2025.107402`
- Authority: Elsevier/ScienceDirect

### FG-02 — LOCKED
- Authors: Xingxing Xie; Gong Cheng; Wenbo Li; Chunbo Lang; Peng Zhang; Yanqing Yao; Junwei Han
- Year: 2025
- Title: *Learning Discriminative Representation for Fine-Grained Object Detection in Remote Sensing Images*
- Journal: *IEEE Transactions on Circuits and Systems for Video Technology*
- Volume/issue: 35(8)
- Pages: 8197–8208
- DOI: `10.1109/TCSVT.2025.3544741`
- Authority: IEEE publisher-format primary PDF

### PRE-01 — LOCKED
- Authors: Wenyu Liu; Gaofeng Ren; Runsheng Yu; Shi Guo; Jianke Zhu; Lei Zhang
- Year: 2022
- Title: *Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions*
- Proceedings: *Proceedings of the AAAI Conference on Artificial Intelligence*
- Volume/issue: 36(2)
- Pages: 1792–1800
- DOI: `10.1609/aaai.v36i2.20072`
- Authority: AAAI official proceedings/PDF

### PRE-02 — LOCKED WITH CONVENTION NOTE
- Authors: Qingpao Qin; Kan Chang; Mengyuan Huang; Guiqing Li
- Citation convention selected for proposal: **CVF ACCV 2022 Open Access convention**
- Year: 2022
- Title: *DENet: Detection-driven Enhancement Network for Object Detection under Adverse Weather Conditions*
- Proceedings: *Proceedings of the Asian Conference on Computer Vision (ACCV)*
- Pages: 2813–2829
- Authority: CVF ACCV 2022 official Open Access page/PDF
- **Do not attach Springer DOI to this CVF citation line.** Springer publishes the LNCS chapter online in 2023 at pp. 491–507 with DOI `10.1007/978-3-031-26313-2_30`. That is a different bibliographic representation of the same ACCV 2022 paper. Proposal author-year currently uses `Qin et al. (2022)`, so the CVF convention is retained for internal consistency.

### PRE-03 — LOCKED
- Authors: Yang Li; Xianguo Li; Michael Lin
- Year: 2025
- Title: *FE-YOLO: Fourier enhancement YOLO for end-to-end object detection in low-light conditions*
- Journal: *Digital Signal Processing*
- Volume: 166
- Article number: 105355
- DOI: `10.1016/j.dsp.2025.105355`
- Authority: Elsevier/ScienceDirect + primary publisher PDF

### PRE-04 — PARTIAL
- Authors: Faturrahman Syauqi; Maulisa Oktiana; Kahlil Muchtar; Al Bahri; Safrizal Razali
- Year: 2025
- Title: *Edge AI-Based Defect Detection in White Pepper (Piper nigrum L.) Using CLAHE-Based Pre-processing and YOLO*
- Proceedings: *2025 IEEE International Conference on Networking, Intelligent Systems, and IoT (ICONS-IoT)*
- DOI: `10.1109/ICONS-IOT65216.2025.11211242`
- Authority: IEEE Xplore-downloaded primary PDF
- Pending field before final APA: exact proceedings page range must be read from the official record/PDF if APA template requires it.
- Method guardrail: paper uses a **composite preprocessing pipeline**; do not describe it as CLAHE alone.

### PRE-05 — LOCKED
- Authors: Siyu Chen; Yixuan Li; Yidong Zhang; Yifan Yang; Xiangxue Zhang
- Year: 2024
- Title: *Soft X-ray image recognition and classification of maize seed cracks based on image enhancement and optimized YOLOv8 model*
- Journal: *Computers and Electronics in Agriculture*
- Volume: 216
- Article number: 108475
- DOI: `10.1016/j.compag.2023.108475`
- Authority: Elsevier/ScienceDirect + primary publisher PDF
- **Discrepancy closed:** article year is 2024 although DOI contains string `2023`.

### PRE-08 — LOCKED
- Authors: Yanchao Yang; Stefano Soatto
- Year: 2020
- Title: *FDA: Fourier Domain Adaptation for Semantic Segmentation*
- Proceedings: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*
- Pages: 4085–4095
- Authority: CVF Open Access official page/PDF

### SPEC-01 — LOCKED
- Authors: Min Cao; Dongping Ming; Lu Xu; Ju Fang; Lin Liu; Xiao Ling; Weizhi Ma
- Year: 2019
- Title: *Frequency Spectrum-Based Optimal Texture Window Size Selection for High Spatial Resolution Remote Sensing Image Analysis*
- Journal: *Journal of Spectroscopy*
- Volume: 2019
- Article ID: 4970376
- DOI: `10.1155/2019/4970376`
- Authority: publisher record + primary publisher-format PDF

### SPEC-02 — LOCKED
- Authors: Jianguo Zhang; Tieniu Tan
- Year: 2003
- Title: *Affine invariant classification and retrieval of texture images*
- Journal: *Pattern Recognition*
- Volume/issue: 36(3)
- Pages: **657–664**
- DOI: `10.1016/S0031-3203(02)00099-7`
- Authority: Elsevier/ScienceDirect
- **Discrepancy closed:** HAL copy/secondary header that reports pp. 215–223 is not used; official ScienceDirect pagination 657–664 wins.

### FREQ-01 — LOCKED
- Authors: Lu Chi; Borui Jiang; Yadong Mu
- Year: 2020
- Title: *Fast Fourier Convolution*
- Proceedings: *Advances in Neural Information Processing Systems*
- Volume: 33
- Pages: 4479–4488
- Authority: NeurIPS official proceedings; page range additionally visible in the reference list of the official CVPR 2025 FDConv paper

### FREQ-02 — LOCKED
- Authors: Hongli Li; Zhiqi Yi; Zhibin Wang; Ying Wang; Liang Ge; Wei Cao; Liye Mei; Wei Yang; Qin Sun
- Year: 2024
- Title: *FDADNet: Detection of Surface Defects in Wood-Based Panels Based on Frequency Domain Transformation and Adaptive Dynamic Downsampling*
- Journal: *Processes*
- Volume/issue: 12(10)
- Article number: 2134
- DOI: `10.3390/pr12102134`
- Authority: MDPI official article page

### FREQ-03 — LOCKED
- Authors: Linwei Chen; Lin Gu; Liang Li; Chenggang Yan; Ying Fu
- Year: 2025
- Title: *Frequency Dynamic Convolution for Dense Image Prediction*
- Proceedings: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*
- Pages: 30178–30188
- Authority: CVF Open Access official page/PDF

---

## C. Detector / diagnosis / evaluation sources

### DET-01 — LOCKED AS PREPRINT
- Authors: Glenn Jocher; Jing Qiu; Mengyu Liu; Shuai Lyu; Fatih Cagatay Akyon; Muhammet Esat Kalfaoglu
- Year: 2026
- Title: *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*
- Source: arXiv preprint
- Identifier: arXiv:2606.03748
- DOI: `10.48550/arXiv.2606.03748`
- Authority: primary arXiv paper
- Guardrail: bibliography must state preprint/arXiv transparently.

### DET-02 — LOCKED
- Authors: Shaoqing Ren; Kaiming He; Ross Girshick; Jian Sun
- Year: 2015
- Title: *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*
- Proceedings: *Advances in Neural Information Processing Systems*
- Volume: 28
- Pages: 91–99
- Authority: NeurIPS official proceedings

### DET-03 — LOCKED
- Authors: Joseph Redmon; Santosh Divvala; Ross Girshick; Ali Farhadi
- Year: 2016
- Title: *You Only Look Once: Unified, Real-Time Object Detection*
- Proceedings: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*
- Pages: 779–788
- Authority: CVF Open Access official page/PDF
- DOI: **not inserted here until IEEE official DOI metadata is separately locked**. The page range and bibliographic identity are already sufficient for a conference APA entry without guessing a DOI.

### DIAG-01 — LOCKED
- Authors: Chengjian Feng; Yujie Zhong; Yu Gao; Matthew R. Scott; Weilin Huang
- Year: 2021
- Title: *TOOD: Task-Aligned One-Stage Object Detection*
- Proceedings: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*
- Pages: 3510–3519
- Authority: CVF Open Access official page/PDF

### DIAG-02 — LOCKED
- Authors: Yue Wu; Yinpeng Chen; Lu Yuan; Zicheng Liu; Lijuan Wang; Hongzhi Li; Yun Fu
- Year: 2020
- Title: *Rethinking Classification and Localization for Object Detection*
- Proceedings: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*
- Pages: 10186–10195
- Authority: CVF Open Access official page/PDF
- Guardrail: official CVF first author is **Yue Wu**.

### DIAG-03 — LOCKED
- Authors: Borui Jiang; Ruixuan Luo; Jiayuan Mao; Tete Xiao; Yuning Jiang
- Year: 2018
- Title: *Acquisition of Localization Confidence for Accurate Object Detection*
- Book/proceedings: *Computer Vision – ECCV 2018*
- LNCS volume: 11218
- Pages: **816–832**
- DOI: `10.1007/978-3-030-01264-9_48`
- Authority: Springer official chapter page
- **Discrepancy closed:** some CVF-style/reference-list citations report conference pagination 784–799. For bibliography, the Springer chapter representation is selected; do not mix 784–799 with the Springer DOI.

### EVAL-01 — LOCKED
- Authors: Tsung-Yi Lin; Michael Maire; Serge Belongie; James Hays; Pietro Perona; Deva Ramanan; Piotr Dollár; C. Lawrence Zitnick
- Year: 2014
- Title: *Microsoft COCO: Common Objects in Context*
- Book/proceedings: *Computer Vision – ECCV 2014*
- LNCS volume: 8693
- Pages: 740–755
- DOI: `10.1007/978-3-319-10602-1_48`
- Authority: Springer official chapter record

---

## D. Activation-visualization sources

### XAI-01 — LOCKED
- Authors: Ramprasaath R. Selvaraju; Michael Cogswell; Abhishek Das; Ramakrishna Vedantam; Devi Parikh; Dhruv Batra
- Year: 2017
- Title: *Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization*
- Proceedings: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*
- Pages: 618–626
- DOI: `10.1109/ICCV.2017.74`
- Authority: CVF Open Access + IEEE DOI record

### XAI-03 — LOCKED AS PRIMARY PREPRINT
- Authors: Mohammed Bany Muhammad; Mohammed Yeasin
- Year: 2020
- Title: *Eigen-CAM: Class Activation Map using Principal Components*
- Source: arXiv primary preprint
- Identifier: arXiv:2008.00299
- Authority: primary arXiv paper
- Guardrail: until the IEEE IJCNN publisher metadata is separately locked, bibliography must cite the preprint transparently and must not invent conference pages/DOI.

---

## E. Current readiness

Unique cited-source set: **34 sources**.

- `LOCKED`: 33
- `PARTIAL`: 1 (`PRE-04`, exact proceedings page range pending if required by the final APA template)
- Backend-only and intentionally excluded from formal cited set: `COF-03` Samudra & Rachmawati (2025), `XAI-02` Grad-CAM++.

### Bibliography gate

`DAFTAR_PUSTAKA.md` is **not yet declared final**. Before generation:

1. close `PRE-04` page-range field from official IEEE record/PDF or explicitly use a page-less conference entry if the institutional APA convention permits it;
2. format the 34 locked records into APA consistently;
3. preserve `DET-01` and `XAI-03` as preprints unless official publisher versions are subsequently locked;
4. run bidirectional audit: `BAB I–III citation → bibliography entry` and `bibliography entry → actual citation in BAB I–III`;
5. compare final author-year strings against the formal manuscript so no year mismatch is introduced during bibliography formatting.

No metadata may be filled by inference during the formatting step.