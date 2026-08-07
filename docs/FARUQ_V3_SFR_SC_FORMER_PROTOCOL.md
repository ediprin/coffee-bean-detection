# Faruq-v3 SFRNet SC-Former Transfer Protocol

## Purpose

`SC1` is a seed-42 **breadth-discovery** arm for the classification bottleneck. It tests whether the combination of spatial long-range interaction and channel-correlation refinement described by SFRNet is useful after transfer to dense YOLO26 classification fields.

This arm is **not** a literal reproduction of SFRNet.

## Paper-derived mechanism

Reference: Cheng et al. (2023), *SFRNet: Fine-Grained Oriented Object Recognition via Separate Feature Refinement*, IEEE TGRS.

The following parts are taken from the paper's SC-Former description:

- the classification refinement contains a spatial transformer (S-Former) and a channel transformer (C-Former) in parallel;
- S-Former applies multi-head self-attention to spatial tokens;
- C-Former transposes the representation so channels become the token sequence;
- C-Former uses one shared linear mapping for Q and K to keep the LSH assignments consistent;
- locality-sensitive hashing follows `h(x) = argmax([xR; -xR])`;
- the paper's default number of hash buckets is `B=4`;
- the original SC-Former operates on RoI features with a 7x7 spatial extent.

The public SFRNet repository currently exposes the spatial classification branch and OR-Former implementation, but the released code inspected for this study does not expose a C-Former/LSH module. Therefore the C-Former implementation here follows the paper equations rather than claiming code-level reproduction.

## Frozen YOLO26 transfer

- Detector family: YOLO26n.
- Starting checkpoint: the exact D0 seed-42 checkpoint bound to the existing D0FT/ACMC1 control summary.
- P3/P4/P5 are passed to the native YOLO26 box and class branches unchanged.
- A parallel SC correction is added only to the class logits.
- Box branch is never replaced or refined.
- Hidden channel dimension: 64.
- S-Former: non-overlapping 7x7 windows, sinusoidal position encoding, 4-head MSA, MLP ratio 2.
- C-Former: the same 7x7 windows; after 1x1 channel projection, each projected channel is a token with 49 components.
- Q/K: one shared `Linear(49,49)` module.
- V: separate `Linear(49,49)` module.
- LSH projection: fixed non-trainable random matrix with seed 2023.
- Hash buckets: 4.
- Attention is masked so a query attends only to channel tokens in the same hash bucket.
- Spatial and channel refined fields are combined with the **fixed arithmetic mean** `(S+C)/2`.
- A zero-initialized 1x1 classifier maps the combined field to class-logit corrections.
- Correction scale: 1.0.

### Explicit adaptation boundary

The paper motivates parallel S/C refinement, shared Q/K LSH, and four buckets. The following choices are specific to this one-stage YOLO26 transfer and must not be described as literal SFRNet equations:

1. applying the RoI operators independently inside non-overlapping dense 7x7 P3/P4/P5 windows;
2. projecting each pyramid level to 64 channels;
3. combining spatial and channel outputs by a fixed arithmetic mean;
4. adding the result as a zero-initialized residual correction to native YOLO26 class logits.

MRL and OR-Former are excluded from `SC1` so the experiment isolates SC-Former-style representation refinement. MRL already has a separate branch in the repository; OR-Former targets localization, which is not the dominant Faruq-v3 residual bottleneck.

## Data and holdout policy

- seed: 42 only;
- evaluation: validation only;
- the development root must not expose a `test` directory;
- no validation confusion pair is encoded in the module or loss;
- all classes are treated uniformly.

## Training schedule

The schedule mirrors the existing SF1 S-Former breadth arm:

- 50 epochs;
- image size 640;
- batch 16;
- workers 2;
- patience 15;
- optimizer `auto`;
- close mosaic at 10 epochs;
- max detections 500.

## Frozen breadth gate

The gate deliberately matches the SF1 breadth criteria so family arms remain comparable.

`SC1` is retained only if all conditions hold relative to D0FT:

1. macro mAP50-95 delta >= -0.20 pp;
2. bottom-3 mean delta >= -2.00 pp;
3. worst-class delta >= -3.00 pp;
4. at least one discovery signal exists:
   - macro >= +0.20 pp, or
   - bottom-3 >= +0.50 pp, or
   - worst-class >= +0.50 pp.

A retained arm is **not** automatically authorized for three-seed confirmation. It only enters the later master breadth comparison.

## Non-claims

`SC1` does not claim:

- literal reproduction of the SFRNet RoI head;
- reproduction of unpublished/unavailable C-Former source code;
- implementation of MRL;
- implementation of OR-Former;
- improvement on the locked test set;
- that dense-window arithmetic-mean fusion is the fusion used in the original paper.
