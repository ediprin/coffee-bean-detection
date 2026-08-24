# 01 — Hong-Centered Literature Method

## 1. Why Hong is the pivot

Hong et al. (2026), *Automated detection of defective coffee beans based on improved YOLOv10 framework*, is used as a methodological pivot because it demonstrates a recognizable research pattern for coffee-YOLO work:

```text
coffee-domain literature
    -> identify unresolved coffee problems
    -> import candidate mechanisms from adjacent CV/agriculture literature
    -> integrate those mechanisms into a detector
    -> validate transfer on coffee through experiments and ablation
```

The thesis should imitate this **reasoning pattern**, not copy Hong's modules.

## 2. How to treat Hong's references

Every Hong reference must be assigned a function. Do not treat the citation list as if all references provide the same type of evidence.

### A. Coffee-domain problem evidence

Use coffee papers to establish:

- historical transition from handcrafted features to deep learning;
- viability of CNN/YOLO for coffee inspection;
- defect taxonomies and number of classes;
- visually similar or subtle defect classes;
- class-wise performance heterogeneity;
- limits of current datasets, protocols, robustness, and deployment.

### B. Adjacent-domain mechanism evidence

Hong also uses papers outside coffee to motivate mechanisms such as lightweight convolution, attention, multiscale feature extraction, and partial convolution.

These papers establish only:

> mechanism X worked or was motivated in domain Y.

They do **not** establish:

> mechanism X is proven to solve coffee-defect detection.

Transfer to coffee must be validated experimentally.

This same discipline applies to AF2.

## 3. Coffee research pattern to preserve

The current literature map suggests the following broad progression:

```text
handcrafted color / morphology / texture
        ↓
CNN feature learning
        ↓
YOLO / object detection for coffee defects
        ↓
more granular defect taxonomies
        ↓
attention / multiscale / transformer / specialized-convolution / metric-learning responses
```

The proposal should present this as a literature progression, not as an assumption that every later model is universally better.

## 4. High-value coffee anchors and what they support

The following papers are high-priority anchors for proposal generation. Exact numerical claims must be checked against their full-text PDFs before insertion into the final thesis document.

### Hong et al. (2026) — improved YOLOv10

Use for:

- direct coffee-YOLO precedent;
- reasoning pattern: baseline -> targeted modification -> ablation;
- author-identified challenges including subtle visually similar defects;
- evidence that coffee-YOLO research often modifies internal feature extraction/fusion rather than using spectral input preprocessing.

Do not use Hong alone to define the entire coffee research landscape.

### Bahy & Rifai (2026) — lightweight YOLOv5s under SNI

Use for:

- high-class-count/SNI coffee detection;
- class-wise heterogeneity;
- evidence that aggregate performance can conceal difficult classes.

### Jundullah et al. (2026) — multi-class YOLOv8 coffee defects/contaminants

Use for:

- 20-class detection context;
- direct object-detection evidence that visually distinctive classes and visually similar coffee-defect classes can behave differently.

### Samudra & Rachmawati (2025) — LSKNet + oriented detection

Use for:

- direct coffee evidence of confusion between visually related defect classes such as black versus partially black;
- evidence that subtle local appearance differences matter even with a small number of classes.

### Hebert & Alamsyah (2026) — SCA coffee defects / YOLOv12

Use for:

- class-wise detection difficulty;
- small/subtle defect cues such as slight insect damage, fungus, and floater-like appearance.

### Kesiman et al. (2023) — SNI fine-grained benchmark

Use for:

- direct evidence that coarse coffee categorization and fine-grained defect categorization are not equivalent problems;
- justification for treating granular defect recognition as a difficult representation problem.

### Arwatchananukul et al. (2024) — 17 defect classes

Use for:

- fine-grained coffee classification;
- controlled-versus-unseen performance gap;
- caution against interpreting high cross-validation scores as complete practical generalization.

### Jiao et al. — Swin/multistage attention coffee grading

Use for:

- evidence that recent coffee research explicitly seeks more discriminative multistage representations;
- model-internal feature recalibration/fusion as a common solution family.

### Hu et al. — Siamese/few-shot coffee defect recognition

Use for:

- explicit subtle visual differences between coffee-defect categories;
- metric/similarity learning as another response to fine-grained discrimination difficulty.

### Gope et al. — YOLO-family coffee comparison

Use for:

- YOLO-family viability in green-coffee tasks;
- few-/moderate-class baseline context.

Do not extrapolate a low-class-count result to a 17–20-class fine-grained setting.

## 5. Cross-paper synthesis we are allowed to make

After verifying the full texts, the thesis may synthesize the following pattern if the cited sources jointly support it:

```text
Few/coarse classes can achieve very strong aggregate performance,
while more granular taxonomies expose larger class-wise disparity,
visually similar categories, difficult lower-tail classes,
and weaker unseen-data behavior.
```

This is a **cross-paper synthesis**. It must be supported by multiple coffee papers and should not be attributed to one author unless that author explicitly makes the same claim.

## 6. Where preprocessing/frequency papers enter

Only after the coffee problem is established should the proposal introduce the second literature layer:

- IA-YOLO: task-oriented preprocessing before YOLO;
- DENet/DE-YOLO: low/high-frequency enhancement before YOLO;
- FE-YOLO: learned Fourier amplitude/phase enhancement before YOLO;
- white-pepper CLAHE-based YOLO: seed-like agricultural preprocessing precedent;
- maize-seed enhancement + YOLOv8: preprocessing contribution in a seed-defect task;
- frequency-angular/fine-grained parent literature: candidate directional-spectral mechanism outside coffee.

These papers justify a **solution space**, not a proven coffee solution.

## 7. Required logic for future writing

Every proposal section that motivates AF2 should preserve this sequence:

```text
1. What does coffee literature say is difficult?
2. What approaches has coffee literature already used?
3. What remains insufficiently explored in the reviewed coffee corpus?
4. What does adjacent literature show is technically possible?
5. What specific hypothesis does this thesis therefore test?
```

If a paragraph cannot answer which of these functions it serves, it probably does not belong in the main argument.
