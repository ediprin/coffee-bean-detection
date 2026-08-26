# Faruq-v3 fresh DLRBC protocol — 2026-08-26

## Status

**FROZEN BEFORE TRAINING.** This is a validation-only seed-42 screening study.
The test split remains closed. Every arm starts a new optimizer from the same
SHA-locked official YOLO26n checkpoint; no D0, D0FT, AF2, or other
coffee-trained checkpoint is accepted.

## Research question

Can a detector-native low-rank bilinear classification residual improve
fine-grained coffee-defect discrimination when trained end to end from the
official pretrained YOLO26n initialization, and is any gain caused by the
quadratic interaction rather than merely by an added low-rank linear path?

The design adapts the low-rank bilinear principle to the native YOLO26n
classification towers. It is not a literal reproduction of the VGG-based
image-classification architecture from the original paper.

## Frozen arms

| Arm | Classification path | Purpose |
|---|---|---|
| `B0_FRESH` | native YOLO26n | fresh end-to-end baseline |
| `LRLIN_FRESH` | native logits + matched low-rank linear residual | parameter- and optimization-matched control |
| `DLRBC_FRESH` | native logits + low-rank quadratic residual | proposed fine-grained mechanism |

Both candidate arms retain the native classification tower. Their residual is
added only to class logits; the box path is unchanged. One-to-many and
one-to-one heads have separate parameters, matching the native dual-head
structure.

## Frozen residual

For each pyramid level with native class-tower output `z`:

1. project `z` from 64 to 32 channels with a learned `1x1` projection;
2. expand to `nc × r`, with rank `r=8`;
3. reshape the rank dimension into four pairs;
4. multiply each pair elementwise and sum the four products;
5. apply signed square-root normalization;
6. add the result to the native class logits with fixed scale `0.1`.

`LRLIN_FRESH` has exactly the same projection, expansion, parameter count, and
initial parameter tensors, but sums the rank components without pairwise
multiplication. This makes it the primary causal comparator.

The values `64 -> 32`, rank 8, and scale 0.1 are frozen from architecture
constraints and the source mechanism before validation. They are not selected
through a validation sweep.

## Initialization lock

Official source:

```text
artifact = yolo26n.pt
SHA-256 = 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
Ultralytics = 8.4.96
```

All arms use the same construction seed, shape-compatible source transfer,
training schedule, augmentations, data, and optimizer selection. Added
projection weights are orthogonal and expansion weights are Xavier-initialized
under the same RNG fork. Resume is permitted only from `last.pt` belonging to
the same arm, seed, config, and output directory.

## Data and training schedule

- Faruq-v3 grouped development set only;
- train: 1,665 images;
- validation: 294 images;
- all 21 classes required in both splits;
- seed 42;
- 50 maximum epochs, patience 15;
- image size 640, batch 16, workers 2;
- deterministic mode, `max_det=500`;
- no test directory may exist.

## Static gates

Training is blocked unless the audit establishes:

1. exact official checkpoint SHA;
2. identical model YAML and training schedule;
3. no coffee-domain parent checkpoint;
4. identical native detector initialization across arms;
5. identical initial adapter tensors and parameter counts for linear and
   quadratic candidates;
6. native class width `(64,64,64)`, projected width `(32,32,32)`, rank 8;
7. separate one-to-many and one-to-one adapters;
8. both residuals are active and functionally different;
9. both preserve native box outputs at the static probe;
10. finite nonzero input and adapter gradients;
11. test remains inaccessible.

## Seed-42 decision

Headline metrics are Macro, Bottom-3, and Worst-class mAP50-95. The primary
deltas are `DLRBC_FRESH - LRLIN_FRESH`; deltas versus `B0_FRESH` are reported
as secondary context.

Promote only when:

- DLRBC improves at least two of the three headline metrics against the
  matched linear control; and
- no headline metric drops by more than 0.5 percentage point.

Only a promotion may authorize fresh seeds 123 and 2026. A seed-42 result is a
screen, not a final superiority claim. The locked test is not opened by this
protocol.

## Parallel Colab execution

Each arm has its own notebook and run directory. The notebooks may run in
parallel on separate Colab accounts, write checkpoints directly to the shared
Drive project, suppress large progress output, and resume only their own
`last.pt`. A separate decision notebook performs no training.
