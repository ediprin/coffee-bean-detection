# APA Metadata Lock — Proposal

Status: **PRIMARY/OFFICIAL METADATA LOCK — NOT THE FORMAL BIBLIOGRAPHY**

Dokumen ini menyimpan hanya metadata bibliografis yang sudah dikunci dari sumber resmi penerbit/proceedings/standard body atau primary paper. `DAFTAR_PUSTAKA.md` tidak boleh mengisi field yang tidak ada di sini dengan ingatan, autocomplete, atau metadata sekunder yang belum diverifikasi.

## Aturan keras

1. Publisher/proceedings/standard-body record atau primary paper menang atas workbook dan aggregator.
2. Jangan menebak DOI, author list, volume, issue, article number, atau page range.
3. Jika versi preprint dan version of record berbeda, daftar pustaka harus menyebut versi yang benar-benar dipakai.
4. Field yang belum dikunci boleh dihilangkan; salah lebih berbahaya daripada tidak lengkap.
5. Setiap discrepancy dicatat eksplisit di bagian akhir.

---

## A. Coffee / domain

| Key | Metadata yang dikunci | Authority |
|---|---|---|
| STD-01 | Badan Standardisasi Nasional. (2008). *SNI 2907:2008 — Biji kopi*. | BSN Pesta Online |
| COF-17 | García, Mauricio; Candelo-Becerra, John E.; Hoyos, Fredy E. (2019). `Quality and Defect Inspection of Green Coffee Beans Using a Computer Vision System`. *Applied Sciences*, 9(19), 4195. DOI `10.3390/app9194195`. | MDPI official article |
| COF-01 | Hong, Sunyan; Zhang, Dengji; Chi, Haiyang; He, Jun; Guo, Xiudong; Fu, Hui; Wang, Lu. (2026). `Automated detection of defective coffee beans based on improved YOLOv10 framework`. *Current Research in Food Science*, 13, 101461. DOI `10.1016/j.crfs.2026.101461`. | Elsevier version of record; author list corroborated by PubMed/PMC for same DOI |
| COF-02 | Bahy, Nanda Aptana Irsyadul; Rifai, Achmad Pratama. (2026). `Real-Time Coffee Bean Defect Detection Based on SNI 01-2907-2008 Standards Using Lightweight YOLOv5s Architecture`. *International Journal on ICT*, 12(1), 29–42. DOI `10.21108/ijoict.v12i1.10584`. | IJoICT official article/PDF |
| COF-03 | Samudra, Melyna Nura; Rachmawati, Ema. (2025). `Deep Learning-Based Defect Detection in Arabica Green Coffee Beans Using LSKNet`. *2025 International Conference on Data Science and Its Applications (ICoDSA)*, 692–697. DOI `10.1109/ICoDSA67155.2025.11157423`. | Primary PDF downloaded from IEEE Xplore + DOI record |
| COF-04 | Hebert, Hocwin; Alamsyah, Derry. (2026). `Detection of Coffee Bean Defects in Speciality Coffee Association Standards using YOLOv12`. *INOVTEK Polbeng - Seri Informatika*, 11(1), 85–95. DOI `10.35314/47yqwd13`. | Official journal page |
| COF-05 | Jundullah, Sayid Muhammad; Aidilof, Hafizh Al Kautsar; Fadlisyah. (2026). `YOLOv8-Based Multi-Class Detection of Coffee Bean Defects and Contaminants for Automated Quality Grading`. *Brilliance: Research of Artificial Intelligence*, 6(2), 313–322. DOI `10.47709/brilliance.v6i2.8612`. | Official journal page |
| COF-06 | Gope, Hira Lal; Fukai, Hidekazu; Ruhad, Fahim Mahafuz; Barman, Shohag. (2024). `Comparative analysis of YOLO models for green coffee bean detection and defect classification`. *Scientific Reports*, 14, 28946. DOI `10.1038/s41598-024-78598-7`. | Nature / Scientific Reports |
| COF-07 | Kesiman, Made Windu Antara; Sulaiman, Ismail; Maysanjaya, I Made Dendi; Dermawan, Kadek Teguh. (2023). `Benchmarking A New Dataset for Coffee Bean Defects Classification Based on SNI 01-2907-2008`. *2023 International Conference on Information Technology Research and Innovation (ICITRI)*, 75–80. DOI `10.1109/ICITRI59340.2023.10249345`. | Primary IEEE paper establishes first page 75; proceedings TOC/J-GLOBAL corroborate 75–80 and next paper begins p.81 |
| COF-08 | Arwatchananukul, Sujitra; Xu, Dan; Charoenkwan, Phasit; Moon, Sai Aung; Saengrayap, Rattapon. (2024). `Implementing a deep learning model for defect classification in Thai Arabica green coffee beans`. *Smart Agricultural Technology*, 9, 100680. DOI `10.1016/j.atech.2024.100680`. | Elsevier / ScienceDirect |
| COF-10 | de Oliveira, Emanuelle Morais; Leme, Dimas Samid; Barbosa, Bruno Henrique Groenner; Rodarte, Mirian Pereira; Pereira, Rosemary Gualberto Fonseca Alvarenga. (2016). `A computer vision system for coffee beans classification based on computational intelligence techniques`. *Journal of Food Engineering*, 171, 22–27. DOI `10.1016/j.jfoodeng.2015.10.009`. | Elsevier / ScienceDirect |
| COF-12 | Jiao, Yujie; Zhao, Yuqing; Jia, Aoying; Wang, Tianyun; Li, Jiashun; Xiang, Kaiming; Deng, Hangyu; He, Maochang; Jiang, Rui; Zhang, Yue. (2025). `Swin-HSSAM: A green coffee bean grading method by Swin transformer`. *PLOS ONE*, 20(5), e0322198. DOI `10.1371/journal.pone.0322198`. | PLOS official article |
| COF-13 | Hu, Xingran; He, Jun; Guo, Xinyu; Hong, Sunyan; Yu, Jing. (2025). `Siamese networks for few-shot coffee bean defect detection`. *LWT*, 235, 118631. DOI `10.1016/j.lwt.2025.118631`. | Elsevier / ScienceDirect |

---

## B. Detection / fine-grained / evaluation

| Key | Metadata yang dikunci | Authority |
|---|---|---|
| DET-01 | Jocher, Glenn; Qiu, Jing; Liu, Mengyu; Lyu, Shuai; Akyon, Fatih Cagatay; Kalfaoglu, Muhammet Esat. (2026). `Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models`. arXiv:2606.03748. DOI `10.48550/arXiv.2606.03748`. | Primary arXiv record. **Do not invent a journal/conference venue.** |
| DET-02 | Ren, Shaoqing; He, Kaiming; Girshick, Ross; Sun, Jian. (2015). `Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks`. *Advances in Neural Information Processing Systems*, 28, 91–99. | NeurIPS official proceedings establishes paper identity/volume; Springer official ECCV chapter bibliography independently gives pp. 91–99. |
| DET-03 | Redmon, Joseph; Divvala, Santosh; Girshick, Ross; Farhadi, Ali. (2016). `You Only Look Once: Unified, Real-Time Object Detection`. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 779–788. | CVF Open Access |
| DIAG-01 | Feng, Chengjian; Zhong, Yujie; Gao, Yu; Scott, Matthew R.; Huang, Weilin. (2021). `TOOD: Task-Aligned One-Stage Object Detection`. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 3510–3519. | CVF Open Access |
| DIAG-02 | Wu, Yue; Chen, Yinpeng; Yuan, Lu; Liu, Zicheng; Wang, Lijuan; Li, Hongzhi; Fu, Yun. (2020). `Rethinking Classification and Localization for Object Detection`. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 10186–10195. | CVF Open Access |
| DIAG-03 | Jiang, Borui; Luo, Ruixuan; Mao, Jiayuan; Xiao, Tete; Jiang, Yuning. (2018). `Acquisition of Localization Confidence for Accurate Object Detection`. In V. Ferrari, M. Hebert, C. Sminchisescu, & Y. Weiss (Eds.), *Computer Vision – ECCV 2018* (LNCS 11218, pp. 816–832). Springer. DOI `10.1007/978-3-030-01264-9_48`. | **Direct Springer chapter page** explicitly gives pp. 816–832, LNCS 11218, editors and DOI. |
| FG-01 | Xu, Xueru; Chen, Zhong; Hu, Yuxin; Wang, Guoyou. (2025). `More signals matter to detection: Integrating language knowledge and frequency representations for boosting fine-grained aircraft recognition`. *Neural Networks*, 187, 107402. DOI `10.1016/j.neunet.2025.107402`. | Elsevier / ScienceDirect |
| FG-02 | Xie, Xingxing; Cheng, Gong; Li, Wenbo; Lang, Chunbo; Zhang, Peng; Yao, Yanqing; Han, Junwei. (2025). `Learning Discriminative Representation for Fine-Grained Object Detection in Remote Sensing Images`. *IEEE Transactions on Circuits and Systems for Video Technology*, 35(8), 8197–8208. DOI `10.1109/TCSVT.2025.3544741`. | IEEE publisher-format primary PDF |
| EVAL-01 | Lin, Tsung-Yi; Maire, Michael; Belongie, Serge; Hays, James; Perona, Pietro; Ramanan, Deva; Dollár, Piotr; Zitnick, C. Lawrence. (2014). `Microsoft COCO: Common Objects in Context`. In *Computer Vision – ECCV 2014* (LNCS 8693, pp. 740–755). DOI `10.1007/978-3-319-10602-1_48`. | Springer published chapter. **Use this published 8-author list, not the earlier 10-author arXiv list.** |

---

## C. Preprocessing / frequency

| Key | Metadata yang dikunci | Authority |
|---|---|---|
| PRE-01 | Liu, Wenyu; Ren, Gaofeng; Yu, Runsheng; Guo, Shi; Zhu, Jianke; Zhang, Lei. (2022). `Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions`. *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(2), 1792–1800. DOI `10.1609/aaai.v36i2.20072`. | AAAI official article |
| PRE-02 | Qin, Qingpao; Chang, Kan; Huang, Mengyuan; Li, Guiqing. (2022). `DENet: Detection-driven Enhancement Network for Object Detection under Adverse Weather Conditions`. *Asian Conference on Computer Vision (ACCV 2022)*, 2813–2829. | CVF ACCV Open Access |
| PRE-03 | Li, Yang; Li, Xianguo; Lin, Michael. (2025). `FE-YOLO: Fourier enhancement YOLO for end-to-end object detection in low-light conditions`. *Digital Signal Processing*, 166, 105355. DOI `10.1016/j.dsp.2025.105355`. | Elsevier / ScienceDirect |
| PRE-04 | Syauqi, Faturrahman; Oktiana, Maulisa; Muchtar, Kahlil; Bahri, Al; Razali, Safrizal. (2025). `Edge AI-Based Defect Detection in White Pepper (Piper nigrum L.) Using CLAHE-Based Pre-processing and YOLO`. *2025 IEEE International Conference on Networking, Intelligent Systems, and IoT (ICONS-IoT)*. DOI `10.1109/ICONS-IOT65216.2025.11211242`. | Primary IEEE publisher PDF. **Method is a composite pipeline; do not call it CLAHE-only.** |
| PRE-05 | Chen, Siyu; Li, Yixuan; Zhang, Yidong; Yang, Yifan; Zhang, Xiangxue. (2024). `Soft X-ray image recognition and classification of maize seed cracks based on image enhancement and optimized YOLOv8 model`. *Computers and Electronics in Agriculture*, 216, 108475. DOI `10.1016/j.compag.2023.108475`. | Elsevier / ScienceDirect. Publication year is 2024 although DOI string contains 2023. |
| PRE-08 | Yang, Yanchao; Soatto, Stefano. (2020). `FDA: Fourier Domain Adaptation for Semantic Segmentation`. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 4085–4095. | CVF Open Access |
| SPEC-01 | Cao, Min; Ming, Dongping; Xu, Lu; Fang, Ju; Liu, Lin; Ling, Xiao; Ma, Weizhi. (2019). `Frequency Spectrum-Based Optimal Texture Window Size Selection for High Spatial Resolution Remote Sensing Image Analysis`. *Journal of Spectroscopy*, 2019, 4970376. DOI `10.1155/2019/4970376`. | Official publisher record |
| SPEC-02 | Zhang, Jianguo; Tan, Tieniu. (2003). `Affine invariant classification and retrieval of texture images`. *Pattern Recognition*, 36(3), 657–664. DOI `10.1016/S0031-3203(02)00099-7`. | Elsevier / ScienceDirect |
| FREQ-01 | Chi, Lu; Jiang, Borui; Mu, Yadong. (2020). `Fast Fourier Convolution`. *Advances in Neural Information Processing Systems*, 33. | NeurIPS official proceedings. Official record does **not expose a page range in the accessible metadata; do not fabricate one.** |
| FREQ-02 | Li, Hongli; Yi, Zhiqi; Wang, Zhibin; Wang, Ying; Ge, Liang; Cao, Wei; Mei, Liye; Yang, Wei; Sun, Qin. (2024). `FDADNet: Detection of Surface Defects in Wood-Based Panels Based on Frequency Domain Transformation and Adaptive Dynamic Downsampling`. *Processes*, 12(10), 2134. DOI `10.3390/pr12102134`. | MDPI official article |
| FREQ-03 | Chen, Linwei; Gu, Lin; Li, Liang; Yan, Chenggang; Fu, Ying. (2025). `Frequency Dynamic Convolution for Dense Image Prediction`. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 30178–30188. | CVF Open Access |

---

## D. Activation visualization / XAI

| Key | Metadata yang dikunci | Authority |
|---|---|---|
| XAI-01 | Selvaraju, Ramprasaath R.; Cogswell, Michael; Das, Abhishek; Vedantam, Ramakrishna; Parikh, Devi; Batra, Dhruv. (2017). `Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization`. *Proceedings of the IEEE International Conference on Computer Vision*, 618–626. DOI `10.1109/ICCV.2017.74`. | CVF Open Access |
| XAI-03 | Muhammad, Mohammed Bany; Yeasin, Mohammed. (2020). `Eigen-CAM: Class Activation Map using Principal Components`. Primary paper: arXiv:2008.00299. Published-conference metadata is corroborated as *2020 International Joint Conference on Neural Networks (IJCNN)*, 1–7, DOI `10.1109/IJCNN48605.2020.9206626`, but direct IEEE landing was not retrievable by the crawler. | **For zero-risk proposal bibliography, cite the primary arXiv version unless the IEEE record is directly locked later.** Primary PDF independently verifies authors/title/mechanism. |

`XAI-02` / Grad-CAM++ is intentionally omitted because it is no longer explicitly cited in formal BAB II/BAB III.

---

## Discrepancy log

1. **COF-10 DOI:** an older workbook used `10.1016/j.jfoodeng.2015.10.030`; Elsevier gives `10.1016/j.jfoodeng.2015.10.009`. Only `.009` is valid here.
2. **PRE-05 year:** article is 2024 (volume 216); DOI contains `2023`. Do not infer publication year from DOI text.
3. **EVAL-01 authors:** Springer published version has 8 authors; arXiv 1405.0312 exposes an earlier 10-author list. Use the Springer 8-author version when citing the Springer chapter.
4. **DIAG-03 pagination:** CVF author-created conference copy is indexed as pp. 784–799, but the **Springer version of record associated with DOI `10.1007/978-3-030-01264-9_48` explicitly states pp. 816–832**. The formal bibliography will use the Springer version-of-record pagination 816–832.
5. **FREQ-01 pagination:** accessible NeurIPS official metadata does not expose a page range. No page range will be invented.
6. **XAI-03 publication version:** primary paper is available and sufficient for the methodological claim. IEEE conference metadata is corroborated, but direct IEEE landing remains unclosed; cite arXiv transparently if bibliography is built before that gate is closed.

---

## Remaining gate before `DAFTAR_PUSTAKA.md`

The major bibliographic conflicts are now resolved. Before rendering the formal APA bibliography:

1. extract the **actual current author–year citations** from BAB I–III one final time;
2. ensure every cited work has an entry in this lock and no uncited work is carried into the bibliography;
3. use the arXiv version for Eigen-CAM unless the IEEE version-of-record is directly verified;
4. render `DAFTAR_PUSTAKA.md` from this lock only;
5. run cited → bibliography and bibliography → cited audits.
