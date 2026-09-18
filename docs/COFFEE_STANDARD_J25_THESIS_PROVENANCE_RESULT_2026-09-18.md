# Coffee Standard J25 Thesis-Provenance Audit

Status: **PASS — PRIMARY-DATASET PROVENANCE SUPPORTED**  
Date: 2026-09-18  
Scope: dataset evidence only; no training and no model test evaluation

## Sources

- Sayid Muhammad Jundullah's Universitas Malikussaleh thesis, *YOLOv8-Based
  Multi-Class Detection of Coffee Bean Defects and Contaminants for Automated
  Quality Grading*.
- The author's QR-linked Drive artifact `data_aug_11.zip`.
- Embedded Roboflow metadata for workspace `tes-rcphs`, project
  `coffee-detection-with-standard`, version 8, CC BY 4.0.

## Reconciliation

| Evidence | Thesis | Author export | Result |
|---|---:|---:|---|
| Original images | 451 | `271 + 113 + 67` | exact |
| Train images | 271 raw | `813 = 271 × 3` | exact |
| Validation images | 113 | 113 | exact |
| Test images | 67 | 67 | exact |
| Train annotations | 3,720 raw | `11,160 = 3,720 × 3` | exact |
| Validation annotations | 1,606 | 1,606 | exact |
| Test annotations | 1,161 | 1,160 | one fewer in export |
| Classes | 25 | 25 | exact |

The archive README states that augmentation creates three versions of each
source image and lists exposure ±10% plus 0.1% salt-and-pepper noise. The
archive has 993 images and zero exact image-hash overlap across its supplied
train, validation, and test splits.

## Correction of the earlier identity audit

The earlier parent parser stripped `.rf.<hash>` and treated the remaining
basename as a unique source ID. That produced 267 components and was wrong:
the thesis establishes 451 raw source photographs, proving that these
basenames collide across distinct assets. Consequently:

- the J25 v1/v2 regrouping is retracted;
- the 253-image v2 artifact must not be used;
- the fixed-holdout feasibility failure based on inferred identities is
  retracted;
- the author-provided official split is restored;
- future identity auditing requires authoritative Roboflow asset IDs or author
  metadata, not filenames.

## Decision and limitations

`PASS_THESIS_LINEAGE_WITH_ONE_ANNOTATION_DISCREPANCY`

Dataset role: `PRIMARY_DATASET_PROVENANCE_SUPPORTED`.

This resolves provenance, not every statistical limitation. Validation and
test are small and some classes have few source images, so Macro mAP, per-class
AP, Bottom-3, uncertainty, and split limitations must be reported. The
one-annotation test discrepancy and the difference between thesis augmentation
text (which also mentions flips) and the v8 README remain disclosed.

A follow-up label audit found that the supplied validation omits class 1 and
the supplied test omits classes 1 and 14. Those official partitions therefore
cannot support 25-class Macro/Bottom-3/Worst claims. Before any model training,
the 271 augmentation triplets are recovered using basename plus exact label
content (never basename alone), visually gated, and combined with the 180
single-source images into a prospective class-complete source split. See
`COFFEE_STANDARD_J25_AF2_DIRECT_PROTOCOL_2026-09-18.md`.

Training remains separately gated until the baseline/model protocol is frozen.
This audit itself neither authorizes training nor opens the test set.
