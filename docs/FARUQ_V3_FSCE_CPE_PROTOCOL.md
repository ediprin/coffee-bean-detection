# Faruq-v3 FSCE-CPE Breadth Screening Protocol

## Purpose

Screen instance-to-instance contrastive proposal encoding from FSCE on the controlled 21-class coffee benchmark without conflating it with prototype losses such as PCL/APCL or hierarchical contrastive losses such as BHCL.

This is a **YOLO26 transfer experiment**, not a literal Faster R-CNN reproduction.

## Paper-derived mechanism retained

FSCE adds a contrastive proposal encoding branch after proposal feature extraction. The proposal feature is projected to a contrastive embedding, and proposals of the same class are pulled together while proposals of different classes are separated by a supervised contrastive objective.

Frozen paper-level settings used here:

- contrastive embedding dimension: **128**;
- temperature: **0.2**;
- contrastive loss coefficient: **0.5**;
- proposal-consistency IoU cutoff for the clipped arm: **0.7**.

The IoU filtering is used to avoid forcing poorly localized proposals to obey a strong category-level representation constraint.

## YOLO26 transfer boundary

YOLO26 does not use Faster R-CNN RoIs. Therefore:

1. Native YOLO26 P3/P4/P5 one-to-many dense locations replace RoI proposal features.
2. Each P3/P4/P5 spatial token is linearly projected to a 128-D embedding by a 1x1 convolution, the dense linear analogue of FSCE's one-layer proposal projection.
3. Native Task-Aligned Assigner remains unchanged.
4. Native box and leaf-classification outputs are unchanged before the auxiliary loss.
5. CPE is attached only to the one-to-many training branch.
6. The projection branch is skipped entirely at inference, so inference remains native YOLO26.

This transfer tests the **instance-level embedding-geometry hypothesis**, not the exact two-stage detector pipeline.

## Predeclared arms

### CPE0 — all TAL positives

All native one-to-many TAL positive locations participate in the supervised contrastive objective.

Purpose: isolate whether instance-to-instance contrastive geometry itself is useful without localization filtering.

### CPE7 — IoU-consistent positives

Only native TAL positives whose predicted box has aligned IoU > 0.7 with its assigned target box participate in the CPE objective.

Purpose: test the proposal-consistency principle used by FSCE while retaining YOLO26's native assignment and localization loss.

## Contrastive objective

For normalized embedding `z_i`, the similarity logit is

`sim(i,j) = z_i^T z_j / tau`.

For each anchor `i`, positives are all other selected instances with the same leaf class. The loss averages the negative log probability of those positives against all non-self selected instances.

Anchors without another same-class selected instance contribute zero rather than an undefined term.

No prototype is maintained. This is the key mechanistic distinction from PCLDet/APCL.

## Scientific safeguards

- no validation confusion pair is used in sampling or loss construction;
- no test image/label is restored or opened;
- native TAL assignment is not modified;
- native box loss remains unchanged;
- native class logits remain unchanged before the auxiliary CPE loss;
- CPE has zero inference overhead by construction;
- seed-42 output is discovery evidence only, not confirmation.

## Frozen breadth settings

- seed: 42
- epochs: 50
- image size: 640
- batch: 16
- embedding dimension: 128
- temperature: 0.2
- loss weight: 0.5
- CPE0 IoU threshold: 0.0
- CPE7 IoU threshold: 0.7
- evaluation: validation only
- D0FT: primary optimization-matched control
- ACMC1: current selected-model reference

## Discovery decision rule

An arm is retained only if it shows at least one useful discovery signal relative to D0FT while satisfying broad-search regression safeguards:

- macro drop no worse than 1.0 pp;
- bottom-3 drop no worse than 2.0 pp;
- worst-class drop no worse than 2.0 pp;
- and at least one of:
  - macro +0.2 pp;
  - bottom-3 +0.5 pp;
  - worst +0.5 pp.

`RETAIN` means the mechanism enters the candidate pool. It does not authorize a test-set evaluation and does not establish a final thesis result.
