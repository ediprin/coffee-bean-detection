# Faruq-v3 DC2 MSFA Mechanism Screening Protocol

Date frozen: 2026-08-07
Protocol: `faruq-v3-dc2-msfa-screening-v1`
Branch: `agent/dc2-msfa-screening`
Stage: broad candidate search / global-local feature aggregation

## Paper basis

Zheng et al. (IEEE Transactions on Image Processing, 2025), *Detector With Classifier²: An End-to-End Multi-Stream Feature Aggregation Network for Fine-Grained Object Detection in Remote Sensing Images*.

The paper defines a local stream from cropped original-image instances, a global stream from the detection backbone, crops global features using the instance boxes, applies global average pooling to align spatial resolution, and fuses the streams as Eq. (8):

`F_l = F_l + tau(F_g)`

where `tau(.)` is global average pooling. The paper later jointly optimizes detection and fine-grained classification with Eq. (9).

## Scope of this arm

This experiment tests **only the MSFA principle** after DC2b proves that a predicted raw-RGB local stream is useful.

It is deliberately not claimed as full DC2 because:

- the selected ACMC1 detector remains frozen;
- the DC2b MobileNetV3-Small local stream remains frozen;
- IFE is not introduced;
- the detector is not collapsed to a coarse parent class;
- the paper's multiple paired Block1-8 stage interactions are not reproduced literally;
- only the deepest detector pyramid input P5 is used as the global semantic stream;
- only a trainable projection from P5 GAP descriptor to the DC2b local pooled dimension is learned.

Thus this is a **final-stage Eq. (8) transfer hypothesis** for YOLO26/coffee, not a literal reproduction.

## Why P5 is frozen

P5 is predeclared as the single global source before results because this arm is testing whether a high-semantic global detector descriptor adds information to the high-resolution local crop. No P3/P4/P5 search is allowed here. A multilevel search would confound the MSFA mechanism with level selection and would reuse validation repeatedly.

## Frozen prerequisites

1. Faruq-v3 grouped development dataset only; `test/` must be absent.
2. Seed fixed to 42.
3. DC2b must use protocol `faruq-v3-dc2-predicted-raw-crop-screening-v2`.
4. DC2b decision must be `PASS` with next action `AUTHORIZE_DC2_GLOBAL_LOCAL_MSFA_SCREENING`.
5. DC2b must use predicted raw RGB crops at the independently frozen 128 x 128 resolution.
6. Detector checkpoint hash must equal the DC2b detector hash.
7. Exactly the DC2b matched predicted-box train/validation records are reused through the checkpoint-keyed DC2b caches.
8. Train and validation matched coverage must each be at least 90%.

## Frozen representation

For every DC2b matched predicted box:

1. Local stream: original RGB predicted crop, 128 x 128, passed through the frozen DC2b MobileNetV3-Small.
2. Global stream: the same predicted box is mapped into the 640 x 640 letterboxed detector coordinate system.
3. P5 RoI feature is sampled with RoIAlign and spatially averaged (GAP), producing `g`.
4. A trainable linear projection maps `g` to the pooled local feature dimension.
5. Fusion is additive before the frozen local classifier:

`f_fused = f_local + W_g g + b_g`

6. `W_g` and `b_g` are initialized to zero. Therefore before training:

`f_fused = f_local`

and DC2c must reproduce DC2b local-only predictions exactly.

Only the global projection is trainable. The detector and complete local classifier are frozen, including local BN statistics.

## Fixed optimization

- seed: 42;
- epochs: 20;
- batch size: 64;
- optimizer: AdamW;
- learning rate: 3e-4;
- weight decay: 1e-4;
- cosine learning-rate schedule;
- local crop augmentation: horizontal + vertical flip only, inherited from DC2b;
- checkpoint selection: best development-validation Macro-F1;
- metrics: Accuracy, Macro-F1, Bottom-3 F1, Worst-F1.

This is matched-object classification evaluation, not detector mAP.

## Mandatory replay control

Before any projection update, the transferred DC2b local checkpoint is replayed on the same validation records. Zero-initialized MSFA must match that replay exactly. The replay Macro-F1, Bottom-3 F1 and Worst-F1 must also match the frozen DC2b report within `1e-5`.

If this fails, the experiment is invalid rather than a model failure.

## Frozen seed-42 gate

Let `L` be the DC2b local-only replay and `M` the best MSFA arm.

All must pass:

1. train matched coverage >= 90%;
2. validation matched coverage >= 90%;
3. DC2b replay agrees with the DC2b report within 1e-5 for Macro/Bottom3/Worst;
4. global P5 descriptor standard deviation > 1e-6;
5. Macro-F1(M) - Macro-F1(L) >= +0.50 percentage point;
6. Bottom-3-F1(M) - Bottom-3-F1(L) >= -0.25 point;
7. Worst-F1(M) - Worst-F1(L) >= -1.00 point.

PASS authorizes a separate end-to-end DC2 integration screen. FAIL only rejects this frozen final-stage P5-GAP additive arm; it does not prove all global-local fusion impossible.

## Leakage and thesis safeguards

- locked test remains absent and unopened;
- no validation confusion pair is used in training;
- no class-specific global feature rule;
- no feature-level search;
- no crop-resolution search;
- no detector/box-branch training;
- no claim of full DC2 reproduction;
- no three-seed confirmation until a later predeclared gate explicitly authorizes it.
