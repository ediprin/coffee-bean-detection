# Faruq-v3 AF2 Complementary Mechanisms — Seed-42 Screening Protocol

Date frozen: 2026-08-28  
Branch: `codex/af2-complementary-mechanisms`  
Status: **IMPLEMENTED — STATIC AUDIT REQUIRED BEFORE TRAINING**  
Evaluation: Faruq-v3 grouped development validation only  
Locked test: **closed**

## Research question

Can one of three mechanisms complement the already-confirmed AF2 frontend
without repeating the failed input-cue fusions, post-hoc box/score separation,
global residual calibration, or unbalanced auxiliary-loss directions?

AF2 remains the thesis method and the common parent. This study does not add
WAV1 or another standalone preprocessing family to the thesis design. It is a
single controlled search for a compatible AF2 extension.

## Evidence that constrains the design

The design is frozen after the following completed diagnostics:

1. AF2's three-seed mechanism attribution was classification-dominant, but the
   raw box/score factorial showed that full AF2 performance still depends on
   their interaction.
2. Direct AF2 + WAV cue fusion failed its frozen seed-42 gate.
3. AF2 residual gating, RGB channel calibration, recovered-cue calibration,
   ontology marginalization, DG/FG regularization, feature-frequency adapters,
   strong-model parent residuals, and class-selective DLRBC did not establish a
   superior AF2 extension.
4. Validation-only quality rescoring did not expose sufficient evidence for an
   IoU-quality loss and cannot be used to authorize one.

Consequently, the two inference-time candidates act on a **shared P3 feature
before both native box and class branches**, while the third candidate is a
train-only balanced contrastive objective.

## Frozen arms

All arms begin from the exact historical `AF2_seed42` checkpoint and continue
for the same 30-epoch schedule.

| Arm | Added mechanism | Inference change |
|---|---|---|
| `AF2CTRL` | No added information; matched AF2 continuation | none |
| `AF2FS1` | Spatially varying low/high-frequency selector on shared P3 | +194 parameters |
| `AF2SFS1` | Spatial-context/frequency-detail selector on shared P3 | +770 parameters |
| `AF2BHCL1` | Balanced leaf/family supervised contrastive loss on assigned positive class logits | none |

### AF2FS1

For shared P3 feature `F`, define a fixed local decomposition:

```text
L = AvgPool3x3(F)
H = F - L
```

A learned `1x1` selector produces two spatial weights, and a depthwise
zero-initialized output projects the selected feature into a residual:

```text
F' = F + Z(softmax(S(F))[0] * L + softmax(S(F))[1] * H).
```

The transfer is inspired by the spatially variant frequency selection idea in
FADC, but does not reproduce FADC's adaptive dilation or replace native YOLO
convolutions:

- Chen et al., *Frequency-Adaptive Dilated Convolution for Semantic
  Segmentation*, CVPR 2024,
  <https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Frequency-Adaptive_Dilated_Convolution_for_Semantic_Segmentation_CVPR_2024_paper.html>.

### AF2SFS1

This arm contrasts a learnable depthwise spatial-context path with the fixed
local high-frequency residual and selects them spatially before a
zero-initialized residual projection. It is a compact transfer of the
space/frequency selection principle, not a reproduction of the paper's SAR
network or fractional Gabor transformer:

- Li et al., *Unleashing Channel Potential: Space-Frequency Selection
  Convolution for SAR Object Detection*, CVPR 2024,
  <https://openaccess.thecvf.com/content/CVPR2024/html/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.html>.

### AF2BHCL1

This arm aggregates the positive class logits assigned to each ground-truth
object and applies class-balanced supervised contrastive losses at two levels:
the 21 leaf classes and the frozen SNI entity-family mapping. The two terms are
equally weighted, scaled by a fixed auxiliary gain of `0.05`, and applied only
to the one-to-many training branch. There is no inference module or parameter.

The conceptual basis is the explicit treatment of hierarchical imbalance and
classification/localization interference in:

- Chen et al., *Balanced Hierarchical Contrastive Learning with Decoupled
  Queries for Fine-grained Object Detection in Remote Sensing Images*, CVPR
  2026,
  <https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Balanced_Hierarchical_Contrastive_Learning_with_Decoupled_Queries_for_Fine-grained_Object_CVPR_2026_paper.html>.

The YOLO transfer does not claim to reproduce DETR queries. It uses native
YOLO positive assignments, leaves the native box objective unchanged, and adds
no ROI/crop path.

## Common training contract

- dataset: Faruq-v3 grouped development;
- seed: 42;
- parent: exact AF2 seed-42 checkpoint and SHA verified by static audit;
- epochs: 30;
- image size: 640;
- batch: 16;
- optimizer: Ultralytics `auto`;
- patience: 10;
- native YOLO26n P3/P4/P5 detector retained;
- validation must contain all 21 classes;
- test must not be present or accessed.

The zero-initialized residuals require `AF2FS1` and `AF2SFS1` to reproduce the
AF2 parent exactly before optimization. `AF2CTRL` and `AF2BHCL1` have exactly
the AF2 detector parameter count.

## Static gates

Training is authorized only when the static audit establishes all of:

1. all arm codes, AF2 configs, model YAMLs, and schedules are exact;
2. all four models reproduce the AF2 frontend input bitwise and detector
   output numerically within `1e-4` at initialization; this is the same CUDA
   tolerance already used by the frozen AF2R/AF2CAL wiring audits because
   separate equivalent GPU forwards can differ by a few ULPs;
3. shared P3 is identical at initialization;
4. activated FS and SFS change both raw boxes and raw scores;
5. all adapter and contrastive gradients are finite and nonzero where required;
6. there is no ROI, decoded-box dependency, or test access.

## Seed-42 decision

Every candidate is compared with `AF2CTRL`, not the historical pre-continuation
AF2 number. A candidate is retained through either route:

### Strict Macro route

- Macro gain at least `+0.5` percentage point;
- Bottom-3 not lower;
- Worst-class drop no more than `1.0` point.

### Lower-tail Pareto route

- Macro drop no more than `0.1` point;
- Bottom-3 gain at least `+0.5` point;
- Worst-class gain at least `+0.5` point.

If more than one candidate is retained, the descriptive winner is selected by
Macro, then Bottom-3, then Worst-class AP. A PASS freezes—but does not silently
execute—a separate paired seed-123/2026 confirmation protocol. A FAIL retains
the existing AF2 method and closes these mechanisms.

## Execution artifacts

- Static audit notebook:
  `notebooks/Faruq_V3_AF2_Complement_Static_Audit_Colab.ipynb`
- Arm notebooks:
  `Faruq_V3_AF2CTRL_Complement_Colab.ipynb`,
  `Faruq_V3_AF2FS1_Complement_Colab.ipynb`,
  `Faruq_V3_AF2SFS1_Complement_Colab.ipynb`, and
  `Faruq_V3_AF2BHCL1_Complement_Colab.ipynb`
- Decision notebook:
  `notebooks/Faruq_V3_AF2_Complement_Decision_Colab.ipynb`
- Drive root:
  `experiments/faruq-v3-af2-complement-v1`

All training notebooks write `last.pt` into Drive, support same-run resume, and
emit only a five-minute epoch status rather than a browser-heavy progress bar.
