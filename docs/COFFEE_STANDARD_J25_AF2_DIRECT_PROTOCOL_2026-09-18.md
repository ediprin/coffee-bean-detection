# Coffee Standard J25 Source-Split AF2 Direct Protocol

Status: **frozen before training**
Date: 2026-09-18

## Question

On a thesis-backed, source-independent J25 split, does the parameter-free AF2 angular
frequency frontend improve a matched YOLO26n detector trained directly from the
same official pretrained checkpoint?

## Dataset contract

- Author archive: `data_aug_11.zip`, SHA256
  `a4d8570e7de8d0ebba408d1d736256806e8f5ff1ebcf62cafd2fa13d149d613d`.
- Ontology: the original ordered 25 classes.
- Source identity is recovered by the conjunction of basename and exact label
  content. It yields exactly 271 train groups of exactly three augmentation
  siblings plus 180 single-source validation/test images: 451 identities total.
- Every recovered triplet must have normalized structural similarity at least
  0.98; observed minimum before freezing is 0.9844.
- A deterministic, label-only grouped assignment produces 315 train, 68
  validation, and 68 locked-test source representatives, with all 25 classes
  present in every split.
- Only train and validation representatives are extracted. Locked-test members
  remain as a frozen manifest inside the source archive.
- Basename alone is explicitly forbidden as an identity rule.

The provenance report must be
`PASS_THESIS_LINEAGE_WITH_ONE_ANNOTATION_DISCREPANCY`. The source-split builder
must pass every identity, similarity, ontology-coverage, and test-lock gate.

## Matched arms

| Arm | Detector | Input |
|---|---|---|
| `D0DIRECT` | YOLO26n P3 | native RGB |
| `AF2DIRECT` | identical YOLO26n P3 | retained AF2 frontend |

Both arms start directly from the same official `yolo26n.pt`, use identical
target-head RNG initialization, seed 42, 50 epochs maximum, image size 640,
batch 16, patience 15, optimizer `auto`, and the same augmentation schedule.
AF2 has no trainable parameters. Neither arm may inherit a coffee checkpoint.

## Seed-42 decision

Promotion occurs through either prospectively frozen route:

1. overall route: Macro gain at least +0.5 point, Bottom-3 not lower, and
   Worst-class drop no more than 1 point; or
2. lower-tail route: Macro drop no more than 0.2 point, Bottom-3 gain at least
   +1 point, and Worst-class gain at least +1 point.

Failure stops AF2 on J25 without extra seeds or test access. Passing authorizes
only paired seeds 123 and 2026. It is not a final superiority claim.

## Execution

The two arm notebooks deterministically reconstruct the same frozen source
split and can run in parallel Colab accounts.
Each writes a run contract, `last.pt`, `best.pt`, log, validation report, and
result JSON to the shared Drive output. A separate decision notebook performs
no training. The test split remains locked throughout this stage.
