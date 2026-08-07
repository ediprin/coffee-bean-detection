# Faruq-v3 DRNet Interaction Verification Screening Protocol

Date frozen: 2026-08-07
Candidate: `DRIV1`
Branch: `agent/drnet-interaction-verification-screening`
Stage: breadth discovery

## Paper/code basis

The official DRNet implementation trains a coarse classifier using labels
converted from fine categories and trains a separate fine-grained classifier on
the original fine labels. At inference, it takes the coarse-class argmax,
obtains the fine-class subset assigned to that coarse class, zeroes fine scores
outside that subset, and only then performs detector post-processing/NMS.

This experiment transfers that **Interaction Verification** rule. It is not a
literal two-stage ORCNN reproduction.

## Frozen coffee coarse taxonomy

The coarse taxonomy is generated only from
`configs/sni21/structured_ontology_v1.yaml` using each class' `entity_family`.
The expected families are determined from the dataset's declared class order and
ontology, not from validation errors/confusions:

- coffee bean;
- dried coffee cherry;
- coffee husk;
- parchment;
- foreign matter.

Exact numeric group IDs follow first occurrence in `data.yaml`; the runner saves
both `class_to_group` and group membership in the result JSON.

## YOLO26 transfer

- Base fine representation is the existing DRF1 dense P3/P4/P5 dual-refinement
  branch; fine residual remains zero-initialized.
- Add a separate P3/P4/P5 coarse classifier.
- Coarse loss: cross entropy on positively assigned one-to-many samples only,
  because YOLO26 has no explicit proposal background class.
- Coarse loss weight: 1.0, matching the unit classification-loss weighting used
  by the official DRNet coarse head.
- Fine classification/localization remain native YOLO26 + DRF1 transfer.
- At inference only, coarse argmax restricts the allowed fine classes on every
  one-to-one dense location; disallowed fine logits are set to -80 before native
  YOLO26 inference/post-processing.
- Box branch is unchanged.

## Frozen training

Matched to DRF1:

- seed 42;
- 50 epochs;
- imgsz 640;
- batch 16;
- workers 2;
- patience 15;
- optimizer auto;
- close_mosaic 10;
- max_det 500;
- same D0 checkpoint initialization.

## Required controls

- D0FT is the optimization-matched baseline reference.
- DRF1 from the existing DRNet breadth screen is the **primary component
  control**, because DRIV1 = DRF1 + coarse supervision + verification.
- ACMC1 remains the selected-model reference.

The runner refuses to proceed unless the D0 checkpoint hash agrees across the
D0FT/ACMC1 control summary and DRF1 predecessor summary.

## Frozen seed-42 gate

Primary comparison: `DRIV1 - DRF1`.

All required:

1. incremental signal: at least one of Macro +0.20 pp, Bottom-3 +0.50 pp, or
   Worst +0.50 pp;
2. Macro drop vs DRF1 no worse than -0.20 pp;
3. Bottom-3 drop vs DRF1 no worse than -1.00 pp;
4. Worst drop vs DRF1 no worse than -2.00 pp;
5. Macro vs D0FT no worse than -0.20 pp.

`RETAIN` means only that the Interaction Verification mechanism remains in the
candidate pool. It does not authorize test access.

## Boundaries

- no locked test access;
- no validation-confusion-derived hierarchy;
- no CML in DRIV1, to isolate Interaction Verification over DRF1;
- no claim that `entity_family` is the only or optimal SNI hierarchy;
- no claim of literal DRNet/ORCNN reproduction;
- no post-result change to coarse grouping or thresholds in this screen.
