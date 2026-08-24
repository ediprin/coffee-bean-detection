# Coffee Evidence Matrix

Purpose: provide the coffee-domain evidence base for proposal drafting. This file is intentionally problem-driven. It does not treat every coffee paper as equally relevant, and it does not infer a frequency-domain bottleneck from coffee-domain difficulty.

Citation keys in this file are internal proposal keys. Convert them to the final bibliography style only when prose is finalized.

## Core rule

Coffee papers establish the **problem**, scope, taxonomy difficulty, class-wise heterogeneity, and current solution patterns. They do **not** by themselves prove that AF2 or any frequency-domain preprocessing is suitable.

## Core coffee-domain evidence

| Key | Paper | Task / classes | What the paper directly supports | Important result / observation | How proposal may use it | Do not overclaim |
|---|---|---|---|---|---|---|
| COF-01 | Hong et al. (2026), *Automated Detection of Defective Coffee Beans Based on Improved YOLOv10 Framework* | Object detection; 7 defect categories | Modern YOLO is viable for coffee; visually similar/subtle defects remain a challenge; targeted architectural modification + ablation is a defensible research pattern | Improved YOLOv10 reports strong aggregate detection performance while the paper still motivates better fine-grained feature extraction | Main narrative pivot; problem framing; methodological pattern | Do not transfer DSConv/SPPF-Attention/PConv gains to AF2 |
| COF-02 | Bahy & Rifai (2026), *Real-Time Coffee Bean Defect Detection Based on SNI 01-2907-2008 Standards Using Lightweight YOLOv5s Architecture* | Object detection; 20 SNI-aligned classes | Multi-class SNI detection exhibits class-wise heterogeneity; lightweight deployment is relevant | Some subtle classes have materially lower AP than easier classes; paper also reports efficiency | Direct evidence that aggregate performance can hide difficult classes | Do not infer that low AP is caused by missing frequency information |
| COF-03 | Samudra & Rachmawati (2025), *Deep Learning-Based Defect Detection in Arabica Green Coffee Beans Using LSKNet* | Object detection; black, partially black, small coffee husk | Even a narrow 3-class problem can contain visually similar categories | Authors identify black vs partially-black confusion due to visual similarity; LSKNet-S mAP@0.5 = 0.879 in their setup | Direct coffee evidence for visual-similarity difficulty | Do not generalize the 3-class result to 20-class detection or claim LSKNet validates AF2 |
| COF-04 | Hebert & Alamsyah (2026), *Detection of Coffee Bean Defects in Speciality Coffee Association Standards using YOLOv12* | Object detection; 15 SCA-style classes | Fine/subtle defects can be much harder than visually distinctive categories | Reported per-class AP includes Cherry Pods 0.89, Floater 0, Fungus Damage 0.18, Slight Insect Damage 0.15; authors discuss tiny marks, texture masking and normal-bean resemblance | Strong direct evidence for lower-tail / difficult-class analysis | Do not use these AP values as evidence of a frequency bottleneck |
| COF-05 | Jundullah et al. (2026), *YOLOv8-Based Multi-Class Detection of Coffee Bean Defects and Contaminants for Automated Quality Grading* | Object detection; 20 classes | Large taxonomy exposes strong per-class disparity and visually-similar-class difficulty | Overall mAP50 reported around 0.75; visually distinctive classes are easier than black/sour variants; per-class metrics available | Direct detection evidence for the fine-grained discrimination problem | Do not characterize all YOLO variants as inherently inadequate from this single study |
| COF-06 | Gope et al. (2024), *Comparative Analysis of YOLO Models for Green Coffee Bean Detection and Defect Classification* | Object detection; 4 classes | YOLO-family detectors are practical for green-coffee detection | Near-saturated results in a relatively small taxonomy | Establish YOLO viability and show why few-class success should not be equated with fine-grained success | Do not generalize 4-class performance to 15–20 classes |
| COF-07 | Kesiman et al. (2023), *Benchmarking a New Dataset for Coffee Bean Defects Classification Based on SNI 01-2907-2008* | Classification benchmark; 3 coarse classes vs 17 defect classes | Increasing label granularity materially increases discrimination difficulty | MobileNet: 92.52% (3-class) vs 39.82% (17-class); InceptionResNetV2: 91.29% vs 53.35% | Strong diagnostic evidence that fine taxonomy is harder than coarse taxonomy | Classification evidence only; never report these values as object-detection performance |
| COF-08 | Arwatchananukul et al. (2024), *Implementing a Deep Learning Model for Defect Classification in Thai Arabica Green Coffee Beans* | Fine-grained classification; 17 defect types | Controlled validation can hide degradation on unseen data | 5-fold CV accuracy 98.78–99.84%; unseen-data accuracy 88.63% | Support generalization caution and need for class-wise analysis | Do not transfer classification behavior directly to one-stage detection |
| COF-09 | Lei et al. (2025), *A Coffee Bean Defect Detection Algorithm with Decoupled Classification and Localization* | Coffee defect recognition; separate classification/localization stages | Classification and localization can require distinct treatment/representations | ResNet34 classification accuracy 98.42%; FMYOLO mAP 96.20 in the paper's two-step workflow | Conceptual support for diagnosing classification vs localization separately | Do not present the paper's two-step architecture as our method |
| COF-12 | Jiao et al. (2025), *Swin-HSSAM: A Green Coffee Bean Grading Method by Swin Transformer* | Grading/classification; 4 top-level groups with 9 defect subdivisions inside defective group | Recent coffee work explicitly improves discriminative representation through multistage fusion and selective attention | Swin-HSSAM reports mAP 98.51%, accuracy 96.34%, F1 96.35%; authors note future need for wider defect coverage | Show that current coffee research mainly attacks representation **inside the model** | Do not use this as object-detection evidence |
| COF-13 | Hu et al. (2025), *Siamese Networks for Few-Shot Coffee Bean Defect Detection* | Similarity-based/few-shot recognition; sound + 6 defect types | Authors explicitly identify limited samples and subtle visual differences as central problems | Siamese accuracy 94.95%, recall 96.06%, precision 93.93%; conventional CNN 74.35%; per-defect F1 varies | Strong direct coffee evidence that subtle categories are a representation/discrimination problem | Despite the title, the evaluated task is not bounding-box detection |
| COF-16 | Gope et al. (2025), *Automated Defective Green Coffee Bean Image Classification Using Deep Learning for Quality Enhancement and Market Competitiveness* | Cross-family benchmark; 6 classes | Modern cross-family coffee benchmarking can perform strongly on a limited taxonomy, while authors still identify small/subtle defects and generalization limits | 4,367 images; optimized YOLOv10-N reports P=0.992, R=0.984, F1=0.987, mAP=0.995 | Additional evidence for the few-class / broader-taxonomy contrast | Do not treat six-class performance as proof of 15–20 class generalization |

## Supporting coffee papers not yet promoted to core prose

These remain useful landscape references, but numerical claims should not be inserted into proposal prose unless the original full text is re-opened and verified during drafting.

- Chang & Liu (2024), *Multiscale Defect Extraction Neural Network for Green Coffee Bean Defects Detection* — Q1 support for multiscale internal-representation improvement.
- Thai et al. (2024), *Coffee Bean Defects Automatic Classification Realtime Application Adopting Deep Learning* — Q1 support for defect classification and practical deployment.
- Adiwijaya et al. (2024), *Coffee Defects Detection Based on Green Bean Images Using YOLO Architecture* — supports viability of YOLO in a small defect taxonomy.
- Adiwijaya et al. (2024), *Real Time Detection of Coffee Bean Defects Using YOLO Method and SAHI Framework* — supports small-object / slicing-oriented detection context.
- Muchtar et al. (2025), *Edge AI-Based Detection for Defective Coffee Beans Using Deep Learning and Streamlit Framework* — deployment-oriented coffee evidence.

## Cross-paper synthesis allowed for proposal

The following synthesis is supported by the set above:

1. Coffee defect recognition has moved from handcrafted features to CNNs, Transformers and YOLO-family detectors.
2. High aggregate performance on small/coarse taxonomies does not demonstrate that detailed multi-class coffee-defect discrimination is solved.
3. As taxonomies become more granular, class-wise heterogeneity, visually similar classes and unseen-data degradation become more visible.
4. Current coffee studies predominantly improve representation **inside the model** through backbone changes, attention, multiscale fusion, metric learning or detector architecture modification.
5. Therefore, input-space preprocessing is a legitimate alternative solution space to investigate, but its effectiveness must be established experimentally.

## Synthesis that is NOT allowed

Do not write any of the following as established facts:

- "Coffee defects are difficult because the detector lacks frequency information."
- "High-frequency features are the proven bottleneck of coffee-defect detection."
- "AF2 is already validated by Hong / Jiao / Hu / Jundullah."
- "YOLO cannot localize coffee beans well."
- "A high aggregate mAP means all defect classes are solved."

## Proposal narrative role

Use this matrix to build the domain-side chain:

```text
manual / subjective coffee inspection
        ↓
computer vision and deep learning
        ↓
YOLO is viable for coffee detection
        ↓
few/coarse-class tasks can be very strong
        ↓
finer taxonomies expose difficult and visually similar classes
        ↓
problem is increasingly one of fine-grained discrimination / representation
        ↓
current coffee solutions mostly alter internal model representation
        ↓
input-space preprocessing becomes a researchable alternative
```

This file should be updated before changing the central problem statement in the proposal.