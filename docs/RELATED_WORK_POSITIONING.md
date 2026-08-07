# Related Work And Research Positioning

Document ID: `RW-COFFEE-YOLO`

Current version: `v1.0.1`

Effective date: `2026-08-02`

Status: evidence map frozen for protocol design. This document does not
authorize training or test evaluation.

Supersedes: none (initial frozen version).

### Versioning policy

- `PATCH` (`v1.0.x`): source-link, indexing-status, wording, or factual
  correction that does not change the research position.
- `MINOR` (`v1.x.0`): adds newly verified studies or changes the evidence
  ranking without changing the central research question.
- `MAJOR` (`vx.0.0`): changes the target task, primary methodological anchor,
  or contribution boundary.
- Never silently overwrite an earlier decision. Update `Current version`, add
  a dated entry to the version history below, and preserve the reason and
  evidence that triggered the change. Git history is the immutable diff-level
  record; this section is the human-readable research-decision record.

### Version history

| Version | Date | Status | Change |
|---|---|---|---|
| `v1.0.0` | 2026-08-02 | Frozen | Initial evidence map; publication-status hierarchy fixed with Hong as the primary architecture anchor, Gope as YOLO-family baseline anchor, Ji as mechanism/ablation anchor, Lei as classification-localization conceptual anchor, and Bahy plus Jundullah as SNI problem anchors. |
| `v1.0.1` | 2026-08-02 | Frozen | Status-only update after the Hong full-transfer implementation passed its no-training architecture gate; the evidence hierarchy and research position are unchanged. |

## Research Scope

The target problem is end-to-end detection and classification of 21 physical
coffee-bean defect and contaminant classes aligned with SNI terminology. The
current development evidence comes from the corrected, parent-grouped Faruq-v3
source. The target is not generic coffee quality classification, binary
good/bad recognition, or dense-grain counting by itself.

The current research question is:

> Can the classification side of an end-to-end YOLO26 detector be improved for
> fine-grained SNI classes after proposal accessibility and localization have
> been shown to be sufficient?

This wording intentionally separates the target from three adjacent problems:

1. choosing a newer YOLO version;
2. detecting more objects in a dense 300-gram scene; and
3. classifying an already cropped single bean with a separate network.

## Evidence Base And Precedence

The local `docs` directory contains 67 PDFs as of this freeze. The older
literature workbook has a 17 July 2026 cutoff and its local-PDF inventory is no
longer complete. It remains useful as an index, but claims in this document are
resolved from the paper PDF and repository evidence.

For research decisions, use this order:

1. raw evaluation report and checkpoint metadata;
2. frozen repository protocol;
3. verified result document;
4. paper full text, including dataset and ablation sections;
5. literature workbook;
6. abstract, README, or chat summary.

Cross-paper headline metrics are not treated as directly comparable when
taxonomies, splits, image identities, object density, or evaluation metrics
differ.

## Publication Status And Citation Hierarchy

Status verified on 2026-08-02. Journal reputation and methodological relevance
are treated as separate axes. Scopus/SJR or Web of Science quartiles are not
interchangeable with SINTA accreditation, and a reputable publisher does not
make every intervention in an article causally established.

| Priority | Study | Verified publication status | Role in this thesis | Boundary |
|---:|---|---|---|---|
| 1 | Hong et al. (2026) | *Current Research in Food Science*, Elsevier; Scopus/SJR Q1 (2024 SJR 1.408), CiteScore 10.7, Impact Factor 7.0 | Primary coffee-specific architecture-transfer reference | Seven merged defect classes, cumulative rather than full-factorial ablation, and no direct SNI-21 evidence |
| 2 | Gope et al. (2024) | *Scientific Reports*, Nature Portfolio; indexed by Scopus and Web of Science | Primary YOLO-family and strong pretrained-baseline reference | Four coarse classes and several architecture/hyperparameter changes bundled together |
| 3 | Ji et al. (2024) | *Journal of Food Processing and Preservation*, Wiley; indexed by Scopus and Web of Science; 2024 CiteScore quartile Q1, while the JCR category quartile is lower | Mechanism and ablation reference for WIoUv3, ECA, and Atn-C3Ghost | Four classes; much of the intervention addresses localization/small-object behavior |
| 4 | Lei et al. (2025) | Springer CCIS conference chapter | Conceptual reference for separating classification and localization bottlenecks | Two-stage routing, not the target single end-to-end detector |
| 5 | Bahy and Rifai (2026) | IJoICT; current SINTA 3 | Primary SNI-taxonomy and problem-scope reference | 13,863 boxes originate from only 107 parent images |
| 6 | Jundullah et al. (2026) | *Brilliance: Research of Artificial Intelligence*; current journal accreditation SINTA 3 | Supporting evidence for 20-class SNI confusion under a standard YOLO head | Random/augmented image protocol and weaker publication venue than the international anchors |
| 7 | Kurniawan et al. (2026) | *Jurnal Teknik Pertanian Lampung*; current SINTA 2 | National deployment, model-scale, and blind-test reference | Binary defective/non-defective target cannot establish fine-grained class discrimination |
| 8 | Tarekegn and Debelee (2025) | Tech Science Press, *Journal on Artificial Intelligence*; Scopus status was not verified from an authoritative index during this freeze | Broad lightweight/fusion comparator | Many simultaneous interventions and no verified high-rank venue basis for making it the primary anchor |
| 9 | Chen and Widiyanto (2026) | *Brilliance: Research of Artificial Intelligence*; current journal accreditation SINTA 3 | Closest published YOLO26 version-level coffee reference | Post-roast, five defect classes plus normal, and mainly transfer-learning/grid-search evidence |

Verified status sources:

- [Current Research in Food Science journal metrics](https://www.sciencedirect.com/journal/current-research-in-food-science)
- [Current Research in Food Science SJR profile](https://www.scimagojr.com/journalsearch.php?q=21101022831&tip=sid)
- [Scientific Reports indexing and metrics](https://www.nature.com/srep/about)
- [Journal of Food Processing and Preservation index record](https://portal.issn.org/resource/issn/1745-4549)
- [IJoICT SINTA profile](https://sinta.kemdiktisaintek.go.id/journals/profile/10709)
- [Brilliance SINTA profile](https://sinta.kemdiktisaintek.go.id/journals/profile/11122)
- [Jurnal Teknik Pertanian Lampung SINTA profile](https://sinta.kemdiktisaintek.go.id/journals/profile/3044)

### Frozen citation decision

No single article supplies the complete foundation. Use the following package
consistently in proposals, protocols, and thesis text:

1. **Hong** as the nearest and strongest primary architecture reference;
2. **Gope** as the strongest YOLO-family/baseline reference;
3. **Ji** as the controlled mechanism-and-ablation reference;
4. **Lei** as the classification-versus-localization conceptual reference; and
5. **Bahy plus Jundullah** as the SNI taxonomy and fine-grained problem
   references.

The resulting position is: Hong is the nearest architecture prior art, not a
universal SOTA result and not the sole foundation. Its seven-class result does
not resolve the present 21-class SNI problem. Publication prestige strengthens
confidence in the article's relevance, but it does not override mismatched
taxonomies, split design, metric choice, or missing mechanism controls.

## Current Empirical Diagnosis

The Faruq-v3 YOLO26n baseline is the development reference. It uses 1,665 train
images, 294 validation images, 21 classes, and no available test images in the
development archive. The baseline protocol is frozen in
`FARUQ_V3_BASELINE_PROTOCOL.md`.

The corrected operational audit established two simultaneous facts:

- lowering the confidence threshold and applying class-agnostic suppression
  raised proposal accessibility from 63.69% to 96.39% and correct-decision F1
  from 45.97% to 56.18%;
- conditional top-1 class accuracy at the selected operating point remained
  57.40%, leaving 42.60% classification-error headroom.

The decision is therefore
`PASS_POSTPROCESSING_CLASSIFICATION_UNRESOLVED`, not “postprocessing solves the
detector.” See `FARUQ_V3_OPERATIONAL_AUDIT_RESULT_2026-08-02.md`.

A P2 feature level did not materially improve proposal accessibility, and the
first-order and bilinear ROI refiners both failed the frozen quick-10 gate.
Those negative results rule out repeating the same candidate-level ROI design;
they do not prove that every classification-side modification must fail.

## Related-Work Map

### 1. YOLO baselines and version comparisons

| Study | Task and evidence | Main intervention | Critical limitation | Use here |
|---|---|---|---|---|
| Gope et al. (2024) | Four green-coffee defect classes; 5,044 images; random 80/10/10 split | Compared YOLOv3/4/5/7/8 and a custom YOLOv8n; changed depth/width, FPN, anchors, and hyperparameters | Several factors changed together; coarse taxonomy; parent identity grouping was not demonstrated | Historical coffee-YOLO baseline, not causal architectural evidence |
| Chen and Widiyanto (2026) | Post-roast coffee defects with YOLO26n | Transfer learning and grid-search tuning | Primarily a model-configuration study; not a fine-grained mechanism | Closest version-level baseline reference |
| Kurniawan et al. (2026) | Binary defective/non-defective conveyor detection | Compared YOLOv11 model scales | Defects are merged into one class | Deployment and efficiency reference only |

Sources:

- [Gope et al. (2024)](<Gope et al. - 2024 - Comparative analysis of YOLO models for green coffee bean detection and defect classification.pdf>)
- [Chen and Widiyanto (2026)](<Chen and Widiyanto - 2026 - Analysis of YOLO26 Model Performance with Transfer Learning in Detecting Coffee Bean Defects  Brill.pdf>)
- [Kurniawan et al. (2026)](<Kurniawan et al. - 2026 - Non-Destructive Detection of Coffee Bean Defects using Machine Vision and the YOLOv11 Algorithm.pdf>)

These studies justify a strong pretrained YOLO baseline. They do not establish
that changing the YOLO generation is a sufficient contribution.

### 2. Coffee-specific architectural modifications

| Study | Architecture | Modification | Evidence quality and caveat | Disposition |
|---|---|---|---|---|
| Ji et al. (2024) | YOLOv8 | Atn-C3Ghost feature extraction and WIoUv3 localization loss | Coffee-specific ablations, but the loss primarily targets localization | Secondary comparator when localization is the bottleneck |
| Hong et al. (2026) | YOLOv10 | Distribution-shift DSConv in backbone/neck, SPPF-Attention, and PConv detection heads | Five-fold results, but ablation is cumulative rather than a full factorial; seven classes merge full/partial black and sour | Primary architecture-transfer comparator, subject to the separate Hong protocol |
| Tarekegn and Debelee (2025) | KN-YOLOv8 | KN modules, MDC downsampling, CSP fusion, SPPF, anchor-free decoupled head, preprocessing and class handling | 562 source images and 19,228 instances with a random 80/10/10 split; many interventions are bundled | Broad coffee-specific comparator, not the first implementation target |
| Samudra and Rachmawati (2025) | LSKNet-based detector | Large selective kernels for contextual feature extraction | Small study and limited protocol detail | Backbone/context reference |

Sources:

- [Ji et al. (2024)](<Ji et al. - 2024 - Coffee Green Bean Defect Detection Method Based on an Improved YOLOv8 Model.pdf>)
- [Hong et al. (2026)](<Hong et al. - 2026 - Automated detection of defective coffee beans based on improved YOLOv10 framework.pdf>)
- [Tarekegn and Debelee (2025)](<Tarekegn and Debelee - 2025 - KN-YOLOv8 A Lightweight Deep Learning Model for Real-Time Coffee Bean Defect Detection.pdf>)
- [Samudra and Rachmawati (2025)](<Samudra and Rachmawati - 2025 - Deep Learning-Based Defect Detection in Arabica Green Coffee Beans Using LSKNet.pdf>)

Hong is the closest direct reference for a modified end-to-end coffee detector,
but its complete module bundle is not assumed to solve the current
classification diagnosis.

### 3. Classification-localization decomposition

| Study | Decomposition | Result type | Limitation for this thesis |
|---|---|---|---|
| Lei et al. (2025) | ResNet34 first classifies global defects; FMYOLO then localizes local defects in the residual category | Two-stage global/local recognition | It is not a single end-to-end detector and requires routing between models |
| Saputra et al. (2025) | YOLOv11 finds coffee beans; EfficientNetV2 classifies cropped beans | Detector plus external classifier | Added latency and crop dependency; source-group independence is unclear |

Sources:

- [Lei et al. (2025)](<Lei et al. - 2025 - A Coffee Bean Defect Detection Algorithm with Decoupled Classification and Localization.pdf>)
- [Saputra et al. (2025)](<Saputra et al. - 2025 - Integration of YOLOv11 and Convolutional Neural Network in a Deep Learning Approach for Coffee Bean.pdf>)

These papers support the conceptual distinction between localization and
classification. Their two-stage implementations are not adopted because the
target contribution remains a single end-to-end detector, and the repository's
own ROI-refinement screening was negative.

### 4. SNI-scale taxonomies and fine-grained confusion

| Study | Data and taxonomy | Reported pattern | Relevance |
|---|---|---|---|
| Bahy and Rifai (2026) | 107 images, 13,863 boxes, 20 SNI categories | Distinct shapes are strong; slight insect damage is substantially weaker | Closest paper by SNI taxonomic scope; also demonstrates the danger of many boxes from few parent images |
| Jundullah et al. (2026) | 2,000 augmented images, 20 physical defect/contaminant classes | Black/sour variants and subtle color/texture classes remain confused | Direct evidence that localization plus a standard YOLO head does not remove fine-grained confusion |

Sources:

- [Bahy and Rifai (2026)](<Bahy and Rifai - 2026 - Real-Time Coffee Bean Defect Detection Based on SNI 01-2907-2008 Standards Using Lightweight YOLOv5s.pdf>)
- [Jundullah et al. (2026)](<Jundullah et al. - 2026 - YOLOv8-Based Multi-Class Detection of Coffee Bean Defects and Contaminants for Automated Quality Gra.pdf>)

These are closer to the present problem definition than a seven-class detector,
even when their detector architecture is older.

### 5. Dense scenes, small objects, overlap, and postprocessing

| Study | Mechanism | Appropriate use | Why it is not the current primary answer |
|---|---|---|---|
| Adiwijaya et al. (2024) | SAHI slicing with YOLO | Tiny-object inference in large scenes | Slicing changes scale/accessibility, not class representation |
| Kasman and Mokobombang (2025) | MFNMS | Recover overlapping detections suppressed by ordinary NMS | Postprocessing cannot recover a wrong class score for an already localized bean |
| Chen et al. (2025), rice | High-resolution head, lightweight architecture, ODConv, MLCA, SIoU | Hundreds of densely bonded grains and counting | Different target: instance accessibility/counting rather than 21-way defect discrimination |
| Chang et al. (2026), flowers | Synthetic augmentation and Second NMS | Duplicate/overlap suppression | Cross-crop operational analogy, not coffee fine-grained evidence |

Sources:

- [Adiwijaya et al. (2024)](<Adiwijaya et al. - 2024 - Real Time Detection of Coffee Bean Defects Using YOLO Method and SAHI (Slicing Aided Hyper Inference.pdf>)
- [Kasman and Mokobombang (2025)](<Kasman and Novy Mokobombang - 2025 - Optimization of the YOLOv4 Algorithm with MFNMS for Defect Detection in Arabica Coffee Beans.pdf>)
- [Chen et al. (2025)](<Chen et al. - 2025 - A lightweight detection model for rice grain with dense bonding distribution based on YOLOv5s.pdf>)
- [Chang et al. (2026)](<Chang et al. - 2026 - SynthAug and SNMS Enhancing flower detection via data augmentation and improved NMS.pdf>)

The 300-gram dense-scene question remains a separate benchmark track. A method
that improves Faruq-v3 does not automatically solve density, and a dense-grain
method does not automatically solve class confusion.

### 6. Detector-family alternatives

RF-DETR and Mask R-CNN studies demonstrate that coffee inspection need not be
restricted to YOLO. They are useful family-level comparators, but replacing
YOLO with a transformer or two-stage segmenter is not itself the identified
fine-grained contribution.

Sources:

- [Sabar et al. (2026)](<Sabar et al. - 2026 - Defect Detection System in Coffee Beans Using Roboflow-Detection Transformer (RF-DTER) Algorithm.pdf>)
- [Talunga et al. (2024)](<Talunga et al. - 2024 - Detection of Coffee Bean Defects on Conveyor Machines Using the Mask-RCNN Algorithm.pdf>)

## Recurring Methodological Weaknesses

Across the corpus, the most important recurring weaknesses are:

1. coarse or merged labels that remove the hardest boundaries;
2. random image splits without demonstrated parent/source grouping;
3. augmentation before the split chronology is made clear;
4. many boxes but few independent source images;
5. global mAP or accuracy without lower-tail and class-pair analysis;
6. cumulative module bundles without factorial or mechanism controls;
7. sparse validation used to support claims about dense deployment;
8. no conditional analysis separating proposal, localization, suppression,
   and class decision errors;
9. no external acquisition-domain validation; and
10. latency numbers measured on incomparable hardware or preprocessing stacks.

These weaknesses are stronger research opportunities than merely replacing
YOLOv11 with YOLO26.

## Research Position

### Nearest prior art by dimension

- **YOLO version baseline:** Chen and Widiyanto (YOLO26).
- **Coffee-specific architecture:** Hong and KN-YOLOv8.
- **Classification-localization diagnosis:** Lei et al.
- **SNI-scale taxonomy:** Bahy and Rifai; Jundullah et al.
- **Dense deployment:** Chen et al. on bonded rice, SAHI, and MFNMS.

No single paper covers the combination of:

1. a 21-class SNI taxonomy;
2. parent-grouped development data;
3. an end-to-end YOLO26 detector;
4. conditional proposal-versus-classification diagnostics;
5. lower-tail class metrics; and
6. factorial and capacity/mechanism controls.

### Defensible gap statement

> Existing coffee-defect detectors frequently improve global detection metrics
> through backbone, neck, loss, or postprocessing changes, often on coarse or
> merged taxonomies. They rarely isolate whether residual errors in a broad SNI
> taxonomy arise before localization or in the conditional class decision. The
> present study evaluates classification-side architectural transfer only after
> proposal accessibility is controlled, using grouped source identities,
> lower-tail class metrics, and explicit mechanism controls.

### Contribution boundary

The potential contribution is not “Hong modules on YOLO26.” It is the
controlled identification and correction of a conditional classification
bottleneck in an end-to-end SNI detector. A Hong-derived module becomes part of
the contribution only if it passes the frozen mechanism and multi-seed gates.

## Explicit Non-Claims

Until independent evidence exists, do not claim:

- state of the art from cross-paper headline comparisons;
- robustness on a real 300-gram dense scene;
- that attention maps localize the causal defect;
- that an improvement from one seed is stable;
- that more boxes imply more independent data;
- that Hong's complete method has been faithfully reproduced by substituting
  ordinary depthwise convolution; or
- that test performance is available.

## Immediate Decision

The static architecture work under `HONG_YOLO26_REPRODUCTION_PROTOCOL.md`
v1.2.0 is complete and recorded in `HONG_YOLO26_IMPLEMENTATION_AUDIT.md`.
The next conditional action is review of that static report followed by the
single `HF_seed42` validation screen. No component ablation, extra seed, or
test access is authorized before the full-transfer fail-fast decision.
