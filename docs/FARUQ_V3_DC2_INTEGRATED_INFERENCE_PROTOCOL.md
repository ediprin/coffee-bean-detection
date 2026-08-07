# Faruq-v3 DC2d Integrated Inference Screening Protocol

## Status

This protocol is a **breadth-search validation screen**. It is deliberately **not claimed as a literal or end-to-end reproduction of DC2**.

Reference: Zheng et al. (2025), *Detector With Classifier²: An End-to-End Multi-Stream Feature Aggregation Network for Fine-Grained Object Detection in Remote Sensing Images*, IEEE Transactions on Image Processing.

## Paper-derived basis

The transfer uses only mechanisms explicitly supported by the paper:

1. Section III-C states that at testing time the detector first produces object regions, the predicted regions are cropped, and the fine-grained classification branch refines the detector category prediction.
2. Eqs. (6)-(8) define local features from predicted image crops, global features cropped from the detector stream, and their aggregation.
3. Eq. (9) defines the paper's joint end-to-end loss. **Eq. (9) is not implemented in DC2d.**
4. IFE Eqs. (2)-(3) are also not implemented because the Faruq-v3 residual audit already indicates high proposal accessibility and the current breadth question is whether secondary raw-pixel re-reading improves fine-grained classification.

## Frozen adaptation

- Detector: frozen YOLO26 checkpoint.
- Split: `val` only.
- Locked holdout: the development directory must not contain `test`.
- NMS: class-agnostic (`agnostic_nms=True`) to approximate a single parent-category detector while preserving the existing detector checkpoint.
- Every post-NMS predicted box is sent to the secondary classifier, including predictions that are false positives. GT matching is **not** used to decide classifier access.
- Local stream: raw RGB crop from each predicted box, resized to the resolution inherited from the passing DC2b screen.
- Global stream: P3/P4/P5 predicted-box descriptors from frozen YOLO26, using the already-screened DC2c MSFA adaptation.
- Classifier: the passing DC2c `MSFA` checkpoint.
- Box geometry: frozen and identical before/after refinement.
- Detection confidence: frozen and identical before/after refinement.
- Fine-grained class ID: replaced by the secondary classifier output.

The detector confidence is intentionally preserved. The paper states that the fine-grained classifier refines category predictions but does not specify a product rule between detector confidence and classifier softmax confidence; inventing such a rule would confound the screening question.

## Evaluation

The same prediction boxes and scores are evaluated twice:

- `NATIVE`: native YOLO26 fine-grained class IDs.
- `DC2_INTEGRATED`: secondary classifier class IDs.

Metrics are class-aware AP at IoU thresholds 0.50:0.95 using a 101-point precision envelope:

- mAP50-95 (primary)
- mAP50
- bottom-three class AP50-95 mean
- worst-class AP50-95

Because boxes and confidence scores are held fixed, the paired delta isolates the effect of class reassignment at detection level.

## Frozen seed-42 gate

DC2d is PASS only if all conditions hold:

1. mAP50-95 gain vs `NATIVE` >= **+0.50 pp**.
2. Bottom-three AP50-95 is **not lower** than `NATIVE`.
3. Worst-class AP50-95 drop is no worse than **-1.00 pp**.

Additional static contracts:

- exactly the same number of predictions before/after refinement;
- box geometry and detector confidence preserved;
- all 21 classes present in validation GT;
- `test` split absent;
- DC2c must already be PASS before this escalation can execute.

## Decision semantics

- PASS: the integrated inference mechanism is worth considering for a later joint/end-to-end transfer experiment.
- FAIL: stop DC2 escalation and return to breadth search. Do not tune on validation confusion pairs.

## Explicit non-claims

DC2d does **not** claim:

- literal reproduction of DC2;
- joint optimization of detector and classifier;
- implementation of Eq. (9);
- implementation of IFE;
- a coarse-category retrained YOLO26 detector;
- test-set performance.
