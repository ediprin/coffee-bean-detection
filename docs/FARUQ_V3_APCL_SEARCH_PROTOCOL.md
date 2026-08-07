# Faruq-v3 APCL Prototype Search Protocol

Date frozen: 2026-08-07
Stage: broad candidate search
Candidate: `APCL1`
Branch: `agent/apcl-prototype-screening`

## Research question

Does explicit prototype-space regularization improve the fine-grained class
representation learned by YOLO26 when residual errors are predominantly
classification/ranking limited?

## Paper-derived operator

The implementation follows Li, Chen, and Li, *IEEE TGRS*, 2025, APCL equations
(9)-(13):

- each class has a prototype;
- the current-batch prototype is the mean of embeddings belonging to that class;
- the persistent prototype is updated using EMA;
- the paper fixes the EMA coefficient to 0.4;
- APCL penalizes positive cosine similarity between an instance and prototypes
  of *different* classes;
- APCL does not explicitly attract an instance toward its own prototype.

## YOLO26 adaptation boundary

This is not a literal reproduction of the ORCNN/RoI contrast head. For a clean
one-stage screen:

- P3/P4/P5 dense features are projected by 1x1 Conv-BN-SiLU into a shared
  128-dimensional embedding space;
- APCL is computed only on positive one-to-many assignments from the native
  YOLO26 training loss;
- one-to-one training remains native;
- the dense detector's box and class logits are unchanged by the APCL head;
- prototype vectors are EMA buffers, not trainable classifier weights;
- APCL projection is not executed during evaluation/inference, so there is no
  APCL inference-time parameter/FLOP path;
- the native D0 Detect state is copied strictly into the wrapped head.

The 128-dimensional projection is an efficiency adaptation; the paper reports a
1024-dimensional contrast embedding by default. Therefore results from APCL1
must be described as an APCL-inspired one-stage adaptation, not as an exact
reproduction of the paper configuration.

## Fixed training setup

- leakage-safe Faruq-v3 grouped train/validation only;
- locked test unavailable and never restored;
- seed 42;
- D0 seed-42 checkpoint initialization;
- embedding dimension: 128;
- EMA eta: 0.4;
- APCL loss weight: 1.0;
- epochs: 50;
- imgsz: 640;
- batch: 16;
- patience: 15;
- optimizer: Ultralytics `auto`;
- close mosaic: 10;
- comparators: existing D0FT seed42 and ACMC1 seed42.

## Frozen broad-search retention gate

`RETAIN` requires at least one signal versus D0FT:

- Macro mAP50-95 gain >= +0.20 percentage points, or
- Bottom-3 mAP50-95 gain >= +0.50 points, or
- Worst-class mAP50-95 gain >= +0.50 points,

plus all safeguards:

- Macro drop no worse than -1.00 point;
- Bottom-3 drop no worse than -2.00 points;
- Worst-class drop no worse than -2.00 points.

This broad-search gate does not declare a winner and does not authorize locked
test evaluation.

## Required pre-training contract

`tests/test_apcl.py` must demonstrate:

- inference is exactly native D0 and does not execute the projection head;
- dense APCL embeddings exist only on the one-to-many training branch;
- embeddings align one-for-one with dense predictions;
- the implemented APCL loss matches the wrong-class positive-cosine formula;
- prototype state follows EMA with eta=0.4;
- own-class similarity is not an explicit attraction term;
- native training box/class predictions are unchanged before the auxiliary loss
  is applied.

## Decision boundary

- `RETAIN`: keep APCL1 in the candidate pool for later confirmation or
  composition (e.g. SAFPN+APCL or DC2+APCL).
- `REJECT`: archive this APCL adaptation and continue the broad search.
- No extra seed or locked test is implied by a single screening result.
