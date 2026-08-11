# SNI-21 Gradient-Conflict Audit Protocol

Version: `v1.0.0`

Frozen: 2026-08-02, before running the audit

## Research question

Does the ontology-marginal objective produce gradients that conflict with the
flat 21-class classification objective at the same D0 detector state?

This audit tests the mechanism suggested by the failed S0 screen. It does not
train a new model and does not establish that gradient projection will improve
validation performance.

## Fixed evidence source

- Dataset: `faruq-development-v3-grouped`.
- Split: train only; validation and test are not read.
- Checkpoint: completed `D0_seed42/weights/best.pt`.
- Images: 192 deterministic samples, implemented as 24 batches of 8.
- Image size: 640.
- Runtime augmentation: disabled.
- Sampling seed: 42.
- Ontology tasks: the five tasks frozen in S0 v1.

The D0 checkpoint is used instead of the trained S0 checkpoint so the audit
measures objective compatibility at a common baseline state rather than only
describing a state already changed by semantic training.

## Compared gradients

For each batch, the detector produces the normal YOLO26 end-to-end assignments.
On the same foreground assignments and logits, compute:

1. `g_leaf`: gradient of the ordinary 21-class classification loss; and
2. `g_ontology`: gradient of the unscaled ontology-marginal auxiliary loss.

The one-to-many and one-to-one branches are combined using schedule-mean
weights `0.45` and `0.55`. These are the arithmetic means of the frozen linear
YOLO26 schedule from `0.80/0.20` to `0.10/0.90`; branch-level components remain
available in the raw report.

Cosine similarity is reported for:

- the feature extractor before the Detect module;
- the Detect classification towers (`cv3` and `one2one_cv3`); and
- every parameter receiving both gradients.

A negative dot product means that, at that model state and batch, a sufficiently
large ontology update can increase the leaf objective to first order.

## Frozen gate

A parameter group has material conflict only when both are true:

- at least 50% of sampled batches have negative gradient cosine; and
- the median gradient cosine is below zero.

Decision routing:

- feature extractor and classification towers conflict:
  `AUTHORIZE_DUAL_HEAD_WITH_SHARED_GRADIENT_PROJECTION`;
- classification towers only:
  `AUTHORIZE_DUAL_HEAD_ISOLATION_ONLY`;
- feature extractor only:
  `AUTHORIZE_SHARED_GRADIENT_PROJECTION_ONLY`;
- neither:
  `STOP_CONFLICT_AWARE_DIRECTION`.

This gate authorizes implementation and a static audit only. It does not
authorize model training, extra seeds, or test access.

## Required output

- cosine median, mean, quartiles, and negative fraction per parameter group;
- gradient norms and shared-gradient parameter count;
- per-batch raw values;
- exact checkpoint, dataset, seed, sampling, and branch-weight provenance;
- explicit confirmation that training, validation, and test were not accessed.

