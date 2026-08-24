# Official Citation Audit — Proposal

Status: **HARD GATE untuk daftar pustaka formal**

Dokumen ini menentukan apakah sebuah sumber boleh masuk ke `docs/thesis/proposal/DAFTAR_PUSTAKA.md`. Master workbook, `CANONICAL_SOURCE_KEYS.md`, catatan percakapan, Google Scholar, DBLP, Crossref, Semantic Scholar, ResearchGate, atau metadata sekunder lain hanya berfungsi sebagai **locator/corroboration**, bukan sebagai otoritas final apabila halaman resmi penerbit/proceedings/standard body tersedia.

## Aturan status

- **FINAL — OFFICIAL VERIFIED**: metadata bibliografis utama telah diverifikasi pada sumber resmi/primer (penerbit, proceedings resmi, badan standar, atau arXiv primer untuk preprint yang memang belum diterbitkan). Boleh dipakai untuk membangun entri APA formal.
- **CORROBORATED — OFFICIAL LANDING PENDING**: identitas paper/DOI telah diperkuat oleh primary preprint atau metadata yang konsisten, tetapi halaman resmi penerbit/proceedings belum ditutup auditnya. Belum boleh dipromosikan sebagai entri final.
- **PENDING — DO NOT CITE AS FINAL**: metadata resmi belum diverifikasi. Jangan menebak author, title, DOI, volume, issue, pages, quartile, atau indexing.

## Larangan

1. Jangan membuat DOI dari ingatan.
2. Jangan mengisi volume/issue/pages dari snippet sekunder bila sumber resmi belum diperiksa.
3. Jangan mengubah `et al.` pada sitasi teks menjadi daftar author lengkap berdasarkan tebakan.
4. Jangan menyebut Q1/Q2/SINTA/Scopus tanpa audit indeks yang relevan.
5. Jangan memasukkan sumber `PENDING` ke `DAFTAR_PUSTAKA.md` sebagai entri final.
6. Klaim metodologis tetap harus dibaca dari full text paper primer; metadata resmi saja tidak cukup untuk membuktikan mekanisme.

---

## A. Sumber yang sudah lolos official-source gate

| Key | Status | Sumber resmi yang diverifikasi | Metadata aman yang telah dikunci |
|---|---|---|---|
| STD-01 | FINAL — OFFICIAL VERIFIED | BSN Pesta Online — `https://pesta.bsn.go.id/produk/detail/7404-sni29072008` | Badan Standardisasi Nasional. **SNI 2907:2008 — Biji kopi**. Status: Berlaku. |
| COF-01 | FINAL — OFFICIAL VERIFIED | ScienceDirect/Elsevier — `https://www.sciencedirect.com/science/article/pii/S2665927126001619` | **Automated detection of defective coffee beans based on improved YOLOv10 framework**. DOI `10.1016/j.crfs.2026.101461`. Tahun 2026. |
| COF-06 | FINAL — OFFICIAL VERIFIED | Nature / Scientific Reports official article page | **Comparative analysis of YOLO models for green coffee bean detection and defect classification**. Scientific Reports 14 (2024), article 28946. DOI `10.1038/s41598-024-78598-7`. |
| COF-08 | FINAL — OFFICIAL VERIFIED | ScienceDirect/Elsevier official article page | **Implementing a deep learning model for defect classification in Thai Arabica green coffee beans**. Smart Agricultural Technology 9 (2024), article 100680. DOI `10.1016/j.atech.2024.100680`. |
| COF-12 | FINAL — OFFICIAL VERIFIED | PLOS official article page | **Swin-HSSAM: A green coffee bean grading method by Swin transformer**. PLOS ONE 20(5) (2025), e0322198. DOI `10.1371/journal.pone.0322198`. |
| FG-01 | FINAL — OFFICIAL VERIFIED | ScienceDirect/Elsevier — `https://www.sciencedirect.com/science/article/pii/S0893608025002813` | Xueru Xu, Zhong Chen, Yuxin Hu, Guoyou Wang. **More signals matter to detection: Integrating language knowledge and frequency representations for boosting fine-grained aircraft recognition**. Neural Networks 187 (2025), 107402. DOI `10.1016/j.neunet.2025.107402`. |
| PRE-01 | FINAL — OFFICIAL VERIFIED | AAAI official proceedings/PDF — `https://ojs.aaai.org/index.php/AAAI/article/download/20072/19831` | Wenyu Liu et al. **Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions**. AAAI-22 (2022). DOI `10.1609/aaai.v36i2.20072`. |
| DET-02 | FINAL — OFFICIAL VERIFIED | NeurIPS official proceedings | Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**. NeurIPS 2015. |
| DET-03 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. **You Only Look Once: Unified, Real-Time Object Detection**. CVPR 2016. |
| DIAG-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Chengjian Feng, Yujie Zhong, Yu Gao, Matthew R. Scott, Weilin Huang. **TOOD: Task-Aligned One-Stage Object Detection**. ICCV 2021. |
| DIAG-02 | FINAL — OFFICIAL VERIFIED | CVF Open Access | Yue Wu et al. **Rethinking Classification and Localization for Object Detection**. CVPR 2020. |
| DIAG-03 | FINAL — OFFICIAL VERIFIED | ECCV/Springer/CVF official proceedings record | Borui Jiang, Ruixuan Luo, Jiayuan Mao, Tete Xiao, Yuning Jiang. **Acquisition of Localization Confidence for Accurate Object Detection**. ECCV 2018. |
| EVAL-01 | FINAL — OFFICIAL VERIFIED | Springer Nature — `https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48` | Tsung-Yi Lin et al. **Microsoft COCO: Common Objects in Context**. ECCV 2014, pp. 740–755. DOI `10.1007/978-3-319-10602-1_48`. |
| DET-01 | FINAL — PRIMARY PREPRINT VERIFIED | arXiv — `https://arxiv.org/abs/2606.03748` | Glenn Jocher, Jing Qiu, Mengyu Liu, Shuai Lyu, Fatih Cagatay Akyon, Muhammet Esat Kalfaoglu. **Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models**. arXiv:2606.03748 (submitted 2026-06-02). arXiv DOI `10.48550/arXiv.2606.03748`. |
| XAI-01 | FINAL — OFFICIAL VERIFIED | CVF Open Access — `https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf` | Ramprasaath R. Selvaraju et al. **Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization**. ICCV 2017, pp. 618–626. DOI `10.1109/ICCV.2017.74`. |

Catatan: `FINAL` di tabel ini hanya mengunci metadata bibliografis yang telah diperiksa. Klaim teknis di BAB II/BAB III tetap harus mengikuti isi full text primer dan tidak boleh diperluas dari abstract/metadata.

---

## B. Sumber yang sudah teridentifikasi tetapi official-source gate belum ditutup

| Key | Status | Yang sudah diketahui | Yang masih wajib diverifikasi sebelum daftar pustaka formal |
|---|---|---|---|
| XAI-02 | CORROBORATED — OFFICIAL LANDING PENDING | Primary arXiv `1710.11063` mengidentifikasi **Grad-CAM++** dan author Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N. Balasubramanian; DOI conference dilaporkan `10.1109/WACV.2018.00097`. | Buka/arsipkan IEEE Xplore official record dan kunci title conference version, author spelling, pages, date. |
| XAI-03 | CORROBORATED — OFFICIAL LANDING PENDING | Primary arXiv `2008.00299`; DOI conference dilaporkan `10.1109/IJCNN48605.2020.9206626`; publisher IEEE. | Buka/arsipkan IEEE Xplore official record dan kunci metadata conference final. |
| COF-03 | CORROBORATED — OFFICIAL LANDING PENDING | Paper LSKNet + oriented detector telah teridentifikasi; DOI locator tersedia di backend. | Verifikasi IEEE/proceedings official landing dan metadata final. |
| FG-02 | PENDING — DO NOT CITE AS FINAL | Xie et al. 2025 fine-grained object detection / DRNet teridentifikasi di backend. | Official IEEE publisher page + full metadata + full-text locator. |
| PRE-02 | PENDING — DO NOT CITE AS FINAL | Qin et al. / DENet / DE-YOLO teridentifikasi di backend. | Official ACCV/Springer proceedings metadata dan PDF. |
| PRE-03 | PENDING — DO NOT CITE AS FINAL | FE-YOLO teridentifikasi di backend. | Official ScienceDirect article metadata, DOI, author list, volume/article number. |
| PRE-04 | PENDING — DO NOT CITE AS FINAL | Syauqi et al. 2025 white-pepper preprocessing teridentifikasi. | Official IEEE/conference record dan full metadata. |
| PRE-05 | PENDING — DO NOT CITE AS FINAL | Chen et al. 2024 maize-seed preprocessing + YOLOv8 teridentifikasi. | Official Elsevier page, DOI, full author/title/volume/pages. |
| SPEC-01 | PENDING — DO NOT CITE AS FINAL | Cao et al. 2019 radial/angular Fourier texture analysis teridentifikasi. | Official publisher/proceedings page dan exact title/DOI. |
| SPEC-02 | PENDING — DO NOT CITE AS FINAL | Zhang & Tan 2003 orientation-spectrum texture study teridentifikasi. | Official publisher page dan exact bibliographic metadata. |
| FREQ-01 | PENDING — DO NOT CITE AS FINAL | Fast Fourier Convolution teridentifikasi. | Official NeurIPS proceedings metadata. |
| FREQ-02 | PENDING — DO NOT CITE AS FINAL | FDADNet teridentifikasi. | Official publisher page + exact metadata. |
| FREQ-03 | PENDING — DO NOT CITE AS FINAL | Frequency Dynamic Convolution teridentifikasi. | Official CVF page + exact author/year/pages. |
| COF-13 | PENDING — DO NOT CITE AS FINAL | Hu et al. 2025 Siamese coffee-defect study teridentifikasi. | Official ScienceDirect page + exact metadata/full text. |
| COF-10 | PENDING — DO NOT CITE AS FINAL | de Oliveira et al. 2016 coffee computer-vision study teridentifikasi. | Official publisher page + exact metadata. |
| COF-17 | PENDING — DO NOT CITE AS FINAL | García et al. 2019 coffee inspection study teridentifikasi. | Official publisher/proceedings page + exact metadata. |

Sumber lain yang muncul pada artefak formal tetapi belum tercantum pada tabel ini otomatis berstatus **PENDING** sampai diaudit.

---

## C. Gate pembangunan `DAFTAR_PUSTAKA.md`

`DAFTAR_PUSTAKA.md` **belum boleh dibangun sebagai daftar pustaka final** sampai langkah berikut selesai:

1. ekstrak seluruh sitasi author-year yang benar-benar muncul di `BAB_I_PENDAHULUAN.md`, `BAB_II_TINJAUAN_PUSTAKA.md`, dan `BAB_III_METODOLOGI_PENELITIAN.md`;
2. cocokkan setiap sitasi dengan satu key canonical;
3. tutup official-source gate untuk setiap sumber yang tetap digunakan;
4. cocokkan author spelling, year, exact title, venue, volume/issue/article number/pages, dan DOI dengan sumber resmi;
5. baru bentuk entri APA dan lakukan audit dua arah: **cited → bibliography** dan **bibliography → cited**.

Jika sebuah sumber tidak dapat diverifikasi secara resmi, pilihannya hanya dua: **hapus/ganti klaim yang bergantung padanya** atau **tetap tandai pending dan jangan menganggap proposal citation-ready**.
