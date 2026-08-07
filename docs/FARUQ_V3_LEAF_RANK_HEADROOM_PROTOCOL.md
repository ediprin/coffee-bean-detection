# Faruq-v3 Leaf-Rank Headroom Protocol

Version: `v1.0.0`

Frozen: 2026-08-02, before inference

## Question

When D0 localizes a validation object but predicts the wrong class, does its
unchanged one-to-one score vector still rank the correct SNI-21 leaf near the
top?

This distinguishes two mechanisms:

- high top-3 recovery: the representation contains a competitive true-class
  signal and a leaf-boundary/reranking design may be rational;
- low top-3 recovery: the correct leaf is absent from the competitive logits,
  so another classifier loss or reranker is not justified.

## Fixed setup

- Checkpoint: completed Faruq-v3 D0 seed 42.
- Split: validation only; test remains unavailable and locked.
- Image size: 640.
- Raw branch: YOLO26 one-to-one.
- Candidate count: 500 per image.
- Matching: class-agnostic greedy IoU matching at IoU >= 0.50.
- Ranking: all 21 sigmoid class scores of each matched raw candidate.
- No training, calibration, threshold search, or checkpoint mutation.

## Required metrics

- localized/matched target count;
- conditional top-1, top-2, top-3, and top-5 accuracy;
- mean reciprocal rank and median true-class rank;
- true-class probability margin against the strongest other class;
- rank distribution and per-class top-1/top-3 recovery;
- largest expected-versus-top-1 confusion pairs with true-class rank.

## Frozen decision gate

`AUTHORIZE_LEAF_RERANKING_PROTOCOL` only when all are true:

1. conditional top-1 is below 80%;
2. conditional top-3 is at least 80%; and
3. top-3 minus top-1 is at least 15 percentage points.

Otherwise:

- top-1 >= 80%: `STOP_HEAD_REFINEMENT_NEAR_SATURATION`;
- top-3 < 80%: `STOP_RERANKING_REPRESENTATION_LIMITED`;
- recovery < 15 points: `STOP_RERANKING_HEADROOM_TOO_SMALL`.

Passing this audit authorizes protocol design and a static control only. It
does not authorize training, extra seeds, or test access.

