# DefectosCafeVerde v7 Eligibility Audit (2026-09-17)

## Scope

This is a model-free audit of the paper-backed DefectosCafeVerde public
dataset. No training or detector test evaluation was executed.

- Paper: *Dual-Sided Green Coffee Bean Defect Inspection Using a Mechatronic
  System with AI-Powered Computer Vision*, Agriculture 2026, 16, 1796.
- Roboflow project: `redtraining/defectoscafeverde`.
- Frozen export: version 7, YOLOv8.
- Archive SHA256:
  `88057de9a6f85a1fba8a0af22bd1d35c28942661132806fe5a1feff98c459069`.

## Frozen-export findings

| Item | Result |
|---|---:|
| Exported images | 9,596 |
| Source parent IDs | 4,038 |
| Boxes | 11,647 |
| Classes | 12 |
| Train images / parents | 8,337 / 2,779 |
| Validation images / parents | 859 / 859 |
| Test images / parents | 400 / 400 |
| Invalid image or label records | 0 |
| Exact cross-split groups | 0 |
| Roboflow-parent cross-split groups | 0 |
| Whole-image dHash review candidates | 27,244 |

The train count is exactly three generated outputs for each of its 2,779
source parents. Validation and test contain one image per Roboflow parent.
Therefore, the frozen v7 export does not place Roboflow augmentation siblings
across splits.

The whole-image dHash candidates are not leakage proof. Manual inspection
shows visually distinct beans receiving similar hashes because the 320 x 320
canvas is dominated by the same pale background and the bean occupies only a
small central area.

## Paper/export discrepancies

The public v7 split is 8,337/859/400 after augmentation, corresponding to
2,779/859/400 source parents. It does not match the paper's stated
2,838/800/400 source split or 6,720/1,920/960 augmented split.

The 4,038 source parents are internally coherent, but the paper's Table 2
per-class image counts sum to 3,999. The frozen export instead has the
following unique-parent image counts:

| Class | Source images |
|---|---:|
| agrio | 333 |
| broca | 239 |
| caracolillo | 522 |
| concha | 320 |
| elefante | 331 |
| helado | 214 |
| negro | 198 |
| normal | 463 |
| oreja | 218 |
| partido | 227 |
| seca | 620 |
| triangulo | 353 |

These counts sum to 4,038 and differ materially from the paper for `negro`,
`partido`, and `seca`.

## Dual-sided physical-identity risk

The paper states that the acquisition system images both faces of each bean.
The export does not provide an explicit physical-bean identifier. Visual review
of sequential files such as `A0/A1`, `A2/A3`, and `A4/A5` is consistent with
opposite faces being stored as consecutive image numbers.

Under the explicit **inferred** grouping rule
`(class prefix, floor(sequence_number / 2))`:

- 4,038 image parents form 2,069 candidate physical-bean groups;
- 1,969 groups contain two sequential views;
- 920 of those pairs cross the published train/validation/test boundaries;
- the cross-split pairs comprise 587 train-validation, 259 train-test, and
  74 validation-test pairs.

For 1,879 sequential pairs with one object on both views, bounding-box shape
is more similar than random within-prefix pairs (median width delta 0.0391 vs
0.0547, height delta 0.0509 vs 0.0691, and absolute log-aspect delta 0.1518 vs
0.1913). This supports, but does not by itself prove, the physical-pairing
hypothesis.

## Decision

**FAIL_AS_PUBLISHED_SPLIT / REBUILD_PHYSICAL_BEAN_GROUPED_SPLIT**

The dataset is paper-backed, licensed, expert-labelled, and scientifically
usable. However, it must not be presented as leakage-safe under the supplied
split until the sequential dual-sided pairing is confirmed by the authors or
acquisition metadata. For thesis experiments, group the two views of each
physical bean before splitting, apply augmentation to train only, and freeze a
new manifest. If the pairing cannot be confirmed, use the dataset only as an
external diagnostic with the limitation stated explicitly.

Training remains unauthorized by this audit.
