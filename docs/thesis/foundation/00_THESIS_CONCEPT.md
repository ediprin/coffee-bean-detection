# 00 — Frozen Thesis Concept

## 1. Core reasoning chain

The thesis must follow this order of reasoning:

```text
Coffee quality-control problem
        ↓
Automated coffee-bean defect recognition/detection
        ↓
YOLO and deep models can perform well on coarse/few-class settings
        ↓
Performance becomes heterogeneous as defect taxonomy becomes more granular
        ↓
Repeated coffee-literature evidence of visually similar/subtle classes,
class-wise disparity, difficult tail classes, and controlled-to-unseen degradation
        ↓
The defensible bottleneck is fine-grained visual discrimination / representation,
not a general claim that localization is the main problem
        ↓
Most coffee studies respond by modifying the internal model:
backbone, convolution, attention, multiscale fusion, transformer, metric learning
        ↓
Adjacent detection literature shows another solution space:
process/enhance the input before the detector
        ↓
Frequency-domain literature shows that spectral decomposition/manipulation can
separate or emphasize different visual components
        ↓
Frequency-angular processing is therefore a plausible candidate mechanism,
NOT a coffee-domain fact
        ↓
Research hypothesis:
parameter-free frequency-angular preprocessing may improve fine-grained
coffee-defect discrimination in YOLO26 without degrading localization
        ↓
Controlled experiment decides whether the hypothesis is true
```

## 2. What the coffee literature establishes

The literature can support statements such as:

- Coffee-defect tasks become harder when the taxonomy is made more granular.
- Some defect categories are visually similar and have substantially lower class-wise performance than visually distinctive categories.
- Strong aggregate accuracy/mAP does not imply that every defect class is solved.
- Recent coffee research often improves representation inside the model using attention, feature fusion, specialized convolution, transformer designs, or metric-learning approaches.

The literature **does not currently establish** that insufficient frequency representation is the proven cause of coffee-defect errors.

Therefore, never write:

> Coffee-defect detection fails because frequency information is not represented adequately.

Instead write:

> Coffee literature identifies fine-grained visual discrimination as a recurring difficulty. Frequency-domain preprocessing is investigated in this thesis as a candidate mechanism for improving the usefulness of discriminative visual cues.

## 3. Position of AF2

AF2 is treated as **input-space preprocessing**, not as a YOLO neck/head/backbone module.

Conceptual pipeline:

```text
I
 -> local patches
 -> FFT
 -> angular/directional spectral analysis
 -> spectral weighting/filtering
 -> IFFT
 -> residual image enhancement
 -> I'
 -> YOLO26
```

Conceptually:

```text
I' = I + I ⊙ Norm(R_AF2)
```

AF2 is parameter-free in the sense that the preprocessing operator adds no learned neural-network parameters. This does **not** mean that AF2 is computationally free or lightweight in latency terms.

## 4. Main thesis comparison

The cleanest core comparison is:

```text
B0: Raw RGB -> YOLO26
M1: AF2(RGB) -> YOLO26
```

Both arms must use matched conditions: same official YOLO26 pretrained initialization, same split, same training budget, same optimizer/training settings, same augmentation policy, and paired seeds.

A CLAHE control may later be added as a classical preprocessing comparator, but it is not required to justify proposal feasibility.

## 5. Main interpretation target

The thesis should not stop at overall mAP. It should test whether any AF2 gain is primarily associated with:

- better raw localization/proposal accessibility, or
- better class discrimination/ranking after localization is already available.

Current working hypothesis:

```text
raw localization accessibility approximately stable
while
classification/discrimination improves
```

This is a hypothesis and pilot-supported interpretation, not a completed final result.

## 6. Scope discipline

The proposal should avoid unnecessary module stacking. Do not make the thesis about AF2 + attention + new neck + new loss + new head simultaneously.

The contribution should remain interpretable:

> **Analysis and optimization of parameter-free frequency-angular image preprocessing for fine-grained coffee-defect detection with YOLO26.**

The research contribution is established by controlled evidence, not by the number of modules added.
