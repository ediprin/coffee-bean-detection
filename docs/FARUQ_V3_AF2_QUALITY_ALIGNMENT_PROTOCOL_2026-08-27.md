# Faruq-v3 AF2 Confidence–IoU Alignment Audit Protocol

Status: **frozen before execution**. Date: 27 August 2026.

## Motivation

The completed box-score factorial established that AF2's gain is
classification-dominant, while its regression and classification outputs still
interact. YOLO26 already trains with Task-Aligned Assignment and soft target
scores; replacing its classification loss with another quality-aware loss
without measuring residual misalignment would therefore be speculative.

This audit asks whether AF2 confidence still fails to rank its fixed final
candidate set by same-class localization quality. It is the required gate
before any Varifocal/Quality-Focal-style training study.

## Fixed design

- Models: optimization-matched D0FT seed 42 and confirmed AF2 seed 42.
- Data: Faruq-v3 grouped validation only; all 21 classes must be present.
- Candidate set, boxes, and predicted classes are frozen to each model's native
  final `max_det=500` output at confidence 0.001.
- Quality target for each prediction is maximum IoU to a ground-truth instance
  of the predicted class, or zero if none exists.
- The oracle changes only global confidence ordering to that quality target.
  It cannot add proposals, alter boxes/classes, or train a model.
- Native metrics must exactly reproduce DD and AA from the completed factorial.
- Test is absent and training is forbidden.

Reported alignment metrics are Spearman confidence–quality correlation,
continuous ECE, and quality Brier error. Oracle headroom is reported for Macro,
Bottom-3, and Worst-class mAP50–95.

## Decision

`QUALITY_ALIGNMENT_HEADROOM_SUPPORTED` requires:

1. fixed-candidate AF2 oracle Macro gain at least 0.5 point;
2. AF2 oracle Bottom-3 gain at least 0.5 point;
3. AF2 oracle Worst-class does not decrease;
4. AF2 confidence–quality Spearman is below 0.95.

A PASS authorizes only a matched seed-42 quality-loss screen with an unchanged
AF2 frontend and architecture. It does not select VFL or QFL parameters and
does not authorize test access or extra seeds.

Methodological bases: Generalized Focal Loss (NeurIPS 2020), VarifocalNet
(CVPR 2021), and Task-aligned One-stage Object Detection (ICCV 2021). The
present audit measures the premise shared by these methods rather than assuming
their transfer to coffee defects.

## Artifacts

- Output root: `experiments/faruq-v3-af2-quality-alignment-v1`
- Summary: `af2_quality_alignment.json`
- Notebook: `notebooks/Faruq_V3_AF2_Quality_Alignment_Audit_Colab.ipynb`

