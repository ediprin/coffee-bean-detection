# Faruq-v3 BHCL Hierarchical Contrastive Breadth Screening Protocol

## Purpose

Screen the **Balanced Hierarchical Contrastive Learning (BHCL)** representation objective of Chen et al. (CVPR 2026) on the controlled SNI-21 coffee detector. This arm tests hierarchy-aware representation geometry, not DETR query decoupling.

The detector remains YOLO26. The native box branch and native inference graph are retained; BHCL adds a training-only projection/prototype objective on positive one-to-many Task-Aligned Assigner (TAL) locations.

## Paper-derived formulation

The paper projects matched classification queries into a normalized representation space. At each hierarchy level `l`, positives are queries sharing the same ancestor node. Pair contrastive probability follows Eq. (6)/(8), with temperature `tau`.

The hierarchy penalty from Eq. (7) is

`lambda_l = exp(1/(L+1-l)) / sum_l' exp(1/(L+1-l'))`.

For this two-level hierarchy (root excluded):

- level 1 coarse weight ≈ 0.37754;
- level 2 fine/leaf weight ≈ 0.62246.

BHCL modifies the denominator so each category contributes its **mean** exponential similarity rather than a raw sample-count-weighted sum. For class `c`, `I'_c = I_c ∪ {M(c)}`, and Eq. (8) divides by `|I'_c|` even when the anchor is removed from the inner sum. The own ancestor prototype is also included in the positive set `P'_l(i)`.

All non-root category nodes have class prototypes. Eq. (10) updates a prototype by

`M_c <- (1 - epsilon^(L-l)) M_c + epsilon^(L-l) mean(f_c)`.

For coarse/intermediate nodes, the paper states that the mean includes matched queries belonging to the node and all descendants.

Verified implementation hyperparameters from Sec. 4.2:

- `lambda_BHCL = 0.6`
- `tau = 0.1`
- `epsilon = 0.1`

The paper uses two augmented views (random flip + random shift), batch size 8, AdamW and 5e-5 learning rate in its DETR experiments.

## SNI-21 hierarchy

Only a strict tree from the predeclared SNI-21 ontology is used. No validation confusion matrix or residual-error pairs define the hierarchy.

Root (excluded from BHCL computation): `SNI21_object`

Level 1: `entity_family`

1. `coffee_bean`
2. `dried_coffee_cherry`
3. `coffee_husk`
4. `parchment`
5. `foreign_matter`

Level 2: the 21 existing leaf classes in the fixed SNI-21 class order.

Other ontology fields (`primary_condition`, `hole_count`, `surface_extent`, etc.) are **not** stacked as hierarchy levels because they are overlapping/orthogonal semantic attributes and do not form one strict tree.

## YOLO26 transfer boundary

### Retained

- 128D projection from P3/P4/P5, capacity-matched to the existing APCL arm (`1x1 Conv -> BN -> SiLU` per level);
- only native one-to-many foreground TAL positives enter BHCL;
- L2 normalization;
- two hierarchy levels and Eq. (7) penalties;
- Eq. (8) category-balanced denominator;
- prototypes for every non-root node;
- Eq. (10) level-dependent EMA;
- paper `lambda_BHCL=0.6`, `tau=0.1`, `epsilon=0.1`;
- no BHCL projection during inference.

### Not transferred

The paper's decoupled classification/localization DETR queries are architecture-specific and are not reproduced in YOLO26. The one-to-one branch remains native. YOLO26 already has separate native classification and box heads, and this experiment does not claim equivalence to the paper's decoder-level decoupling.

The paper applies BHCL at each DETR decoder layer. YOLO26 has no decoder-layer query stack; BHCL is applied once to the P3/P4/P5 projected positive representation.

The paper explicitly creates two augmented views. This breadth arm keeps the project-controlled detector augmentation schedule instead of doubling each image into a paired-view pipeline. Native TAL provides multiple positive locations for each matched GT, so same-category positive representations exist. This is a declared transfer choice and not a literal reproduction of the paper's view-generation protocol.

## Prototype initialization choice

The paper describes a learnable prototype bank and gives its EMA update, but the text does **not specify prototype initialization**. This implementation therefore uses deterministic all-zero prototype buffers.

Consequences:

- before a node is observed, normalized prototype similarity is 0 and contributes `exp(0)=1` to the balanced denominator;
- all hierarchy categories are still represented from the first mini-batch, including categories absent from that batch;
- after observation, Eq. (10) is applied literally.

For `L=2`, `epsilon=0.1` gives:

- coarse level `alpha = epsilon^(2-1) = 0.1`;
- leaf level `alpha = epsilon^(2-2) = 1.0`.

Thus a seen leaf prototype is replaced by that batch's leaf mean, whereas a coarse prototype moves by 0.1 toward the mean of all positive descendants in that family. We do not alter this seemingly aggressive leaf update because it follows Eq. (10) exactly.

Prototype state is stored as model buffers so resume/checkpoint behavior is reproducible.

## Equation-sign note

The PDF text renders Eq. (6)/(8) as a **negative log** pair loss, but Eq. (7)/(9) also prints an additional leading minus around the aggregation. Reading both signs literally would turn the contrastive objective negative. The implementation uses the conventional positive aggregation of the already-defined `-log(probability)` pair losses. This is recorded as a notation inconsistency in the source paper rather than silently applying a double negative.

## Efficient exact Eq. (8) computation

A full `[N,N]` exponential-similarity matrix can become large with TAL positives. The implementation chunks anchors while preserving the exact Eq. (8) denominator:

1. compute anchor-to-all-positive similarities for a chunk;
2. aggregate exponentials by hierarchy category;
3. subtract each anchor from its own category inner sum;
4. add every class prototype;
5. divide each category contribution by `|I_c|+1`;
6. sum categories;
7. compute the mean positive logit using category embedding sums plus the own prototype.

This avoids changing the loss while bounding peak memory.

## Loss bookkeeping

Ultralytics exposes a fixed three-component detection-loss vector. The paper treats BHCL as a separate classification-representation term. We add `0.6 * L_BHCL` to the classification component for logging/aggregation only; native box/DFL components are unchanged.

## Frozen breadth-search setup

- seed: 42
- candidate: `BH1`
- embedding dimension: 128
- `tau`: 0.1
- `lambda_BHCL`: 0.6
- `epsilon`: 0.1
- anchor chunk: 256
- epochs: 50 (project discovery schedule)
- image size: 640
- batch: 16 (project-controlled comparison budget, not paper batch 8)
- validation only
- test split unavailable/forbidden

## Discovery gate

Compared to D0FT:

- macro mAP50-95 no worse by more than 0.2 pp;
- bottom-3 no worse by more than 2 pp;
- worst-class no worse by more than 3 pp;
- at least one discovery signal: macro +0.2 pp, bottom-3 +0.5 pp, or worst +0.5 pp.

`RETAIN` only places BH1 into the later multi-seed candidate pool. It is not a final result.
