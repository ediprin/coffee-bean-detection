# Controlled B/C generalization baseline V1

Registered before training and evaluation of this experiment. The purpose is to
separate source-group sensitivity from a trivial difference in training image
count. This is exploratory development, not independent test confirmation.

## Design
G0 is the completed official-fold-A RGB baseline (676/119), descriptive only.
The originally proposed full B->C and C->B comparisons are revised to use
common evaluation sets and equal image-count budgets. The official val B
subset has missing classes (8 and 13) and is not suitable for clean18 AP.
Use the same 795 previously audited B/C images, exact fingerprint and
Zenodo metadata. A is locked. Only labels and manifest determine the split.
No learned scores or pixel-based selection. Select 75 evaluation images per
group by greedily minimizing squared normalized per-class count deviations
from the proportional target, with deterministic SHA256 tie order. For B use
all 300 remaining images for training; for C select 300 of the remaining
345 by the same rule, leaving 45 unused. Freeze the four list digests using
the previous feature cache IDs and labels; verify against actual polygons
before training. No overlap between any of the four lists.

| Partition | Images | Instances | Min/max class instances |
|---|---:|---:|---:|
| B train | 300 | 976 | 41/48 |
| C train | 300 | 900 | 42/49 |
| B eval | 75 | 265 | 11/13 |
| C eval | 75 | 230 | 10/12 |

Train G1 on B only, G2 on C only. During training validate on the source's
internal eval set only. Evaluate each fixed last.pt at epoch 80 on BOTH eval
sets. Same official YOLO26s-Seg pretrained weights, 21 classes, Ultralytics
8.4.210, seed 0, batch16, 640, AdamW, same LR/augmentation as G0.
Patience=0 disables early stopping; no best-epoch selection for endpoint.
Same image count and nominal updates; object counts, composition, scale,
acquisition and specimen identity still differ. No reuse of G0 or pilot weights.

| Training | Eval B | Eval C |
|---|---|---|
| G1 B | B->B internal | B->C cross |
| G2 C | C->B cross | C->C internal |

Primary contrasts: on eval B, AP(G1)-AP(G2); on eval C, AP(G2)-AP(G1).
Report both signed contrasts, not just their average. Primary clean18 box AP,
secondary hard5 box AP, bottom3/worst AP, mask equivalents and per-class AP.
Positive contrasts suggest group-dependent transfer difficulty under this
design; not causal identification of specimen vs acquisition shift, nor
proof of a loss/assignment defect. Both directions need to be examined.
All 21 trained; clean18 excludes three foreign-object classes with reuse.
Within-group partitions may share physical specimens; no specimen IDs for
clean grouping. Prior audits already used B/C, so no independent confirmation.
No claims of improvement over G0 from these different splits. No hypothesis
tests treating individual repeated instances as independent.
Next: frozen classification/ranking/assignment diagnosis of G1/G2. Only
choose a modification if diagnostic evidence supports it. No architecture
modification or training on A here.

## Kaggle
Import notebook, attach existing auliya-robusta-sni, GPU + Internet, Run All.
Default trains both source groups in separate output folders. For shorter
sessions set GROUPS=["B"] or ["C"] and attach the previous full Notebook Output
for resume in a compatible environment. Keep the same code and split.
Report archive omits checkpoint bytes; preserve last/best for follow-up.
No actual GPU training or runtime test claimed by implementation.
