# Faruq-v3 DC2 Global–Local MSFA Screening Protocol

Date frozen: 2026-08-07
Stage: broad candidate search / frozen-detector feature aggregation
Candidate family: `DC2 multi-stream feature aggregation`
Branch: `agent/dc2-msfa-global-local-screening`

## Paper basis

Zheng et al. (IEEE Transactions on Image Processing, 2025) define the detector
backbone as the global stream and the raw detected-object classifier as the
local stream. For an object/prediction, the paper crops the detector feature
map to obtain the global-stream descriptor and separately encodes the raw RGB
object crop to obtain the local-stream descriptor. Their MSFA operation is
reported as Eq. (8):

`F_l = F_l + tau(F_g)`

where `tau` is global average pooling used to align the spatial resolution of
the global feature before residual addition. The original DC2 architecture can
perform stage-paired addition because its corresponding global/local blocks
have equal channel dimensions.

## YOLO26 adaptation boundary

YOLO26 and MobileNetV3 do not have the stage/channel correspondence assumed by
DC2. Therefore this screen does **not** claim literal reproduction of DC2 MSFA.
The frozen adaptation is:

1. use the selected YOLO26 detector as the global-stream network;
2. hook the three detection-pyramid inputs P3, P4, and P5;
3. for every detector-matched predicted box, crop each feature map with ROIAlign;
4. apply spatial global average pooling to each cropped feature patch;
5. concatenate the P3/P4/P5 global descriptors;
6. linearly project this descriptor to the terminal MobileNetV3 local-feature
   dimension;
7. add the projected global descriptor residually to the local descriptor before
   the original MobileNetV3 classifier.

The projection is zero-initialized. Before training, enabling MSFA must therefore
produce bitwise-identical logits to the DC2b local classifier.

This arm intentionally keeps the detector frozen. Joint detector/classifier
optimization belongs to a later end-to-end DC2 experiment.

## Frozen prerequisites

- Faruq-v3 grouped train/validation only; locked holdout must remain unavailable.
- seed 42 only.
- DC2b predicted raw-crop screen must be `PASS`.
- detector checkpoint is the same checkpoint used by DC2b and is hash-tracked.
- raw-crop resolution is inherited from DC2b; no resolution search occurs here.
- train and validation predicted-box coverage must each remain >= 90%.
- the DC2b predicted-crop best checkpoint initializes both arms.

## Global feature extraction

- detector input size: 640;
- global levels: P3, P4, P5;
- predicted boxes are transformed from original-image coordinates to the exact
  letterboxed detector coordinates;
- each feature patch is ROIAligned to 3x3 and then globally averaged;
- P3/P4/P5 vectors are concatenated;
- global caches are keyed by detector checkpoint SHA-256 and a signature of the
  exact ordered predicted-box records;
- detector parameters are never optimized during this screen.

## Optimization-matched comparison

Two arms start from the same DC2b local checkpoint and receive the same additional
optimization schedule:

- `LOCAL_FT`: predicted raw-crop local classifier only; global descriptor ignored.
- `MSFA`: identical local model plus zero-initialized global projection and residual
  addition.

This control prevents extra epochs from being misattributed to MSFA.

Fixed schedule for both arms:

- 10 additional epochs;
- AdamW learning rate 1e-4;
- weight decay 1e-4;
- cosine schedule;
- batch size 64;
- same predicted-crop augmentation inherited from DC2b;
- checkpoint selection by validation Macro-F1.

## Frozen seed-42 gate

Primary comparison is `MSFA - LOCAL_FT`. All must pass:

1. Macro-F1 gain >= +0.50 percentage point;
2. Bottom-3 F1 is not lower;
3. Worst-F1 drop is no more than 1.00 percentage point.

If all pass: `PASS` and authorize a separately designed end-to-end DC2 integration.
If any fail: `FAIL`; do not claim the paper's complete DC2 mechanism has failed,
because this arm uses a frozen detector and a YOLO26-specific terminal fusion
adaptation.

## Boundaries

- no locked holdout access;
- no validation confusion pairs as training knowledge;
- no class-specific fusion rules;
- no adaptive choice of pyramid levels after seeing results;
- no box-branch change;
- no claim of literal stage-paired MSFA reproduction;
- no claim of end-to-end DC2 until the later joint-training arm is implemented.
