# Faruq-v3 FC-STB Frequency-Consistent Distillation Protocol

Status: frozen before training (2026-08-13).

## Question

Can a single STB detector absorb complementary AF2 frequency decisions through
ground-truth-bounded classification distillation, without changing the native
YOLO26 localization path or adding inference-time modules?

## Evidence and boundary

The seed-42 breadth screen retained STB1 (Macro 88.67%, Bottom-3 83.64%, Worst
80.81%) and AF2 (88.20%, 80.04%, 79.35%). SGFR later showed that a frozen IGEM
residual did not beat an optimization-matched STB control. FC-STB therefore does
not stack IGEM or add another inference head. AF2 is used only as a frozen
training teacher.

The design is motivated by detector feature distillation and fine-grained
expert diversity, but its coffee transfer and GT-bounded loss are research
choices. No superiority claim exists before the frozen gates pass.

## Models

- `STB1`: frozen reported reference and initialization checkpoint.
- `FCT0`: optimization control. STB blocks and native classification heads are
  trained; backbone, neck, and box heads are frozen.
- `FCD1`: identical initialization, trainable set, optimizer, schedule, and
  parameter count as FCT0. The only difference is AF2 teacher distillation.

Both candidates contain exactly one STB detector. The AF2 teacher is owned by
the training criterion, is frozen, and is not registered or serialized in the
student. Inference is a single forward and has no AF2 preprocessing.

## Loss

For positive anchors assigned by the unchanged YOLO26 assigner:

```text
L = L_YOLO + lambda * L_KD
```

`L_KD` is temperature-scaled class-logit KL. A teacher anchor contributes only
when AF2's top-1 class equals its assigned ground-truth class and its
ground-truth probability is at least 0.10. This prevents known AF2 errors from
being copied. Box, DFL, assignment, and localization losses remain native.

Frozen settings: temperature 2.0, lambda 0.50, 20 epochs, batch 16, image 640,
AdamW, learning rate 0.001, cosine schedule, seed 42.

## Fail-fast stages

1. Static audit: strict STB transfer, exact raw boxes/scores, identical student
   parameter schema, exact freeze policy, no registered teacher.
2. Validation-only headroom diagnostic: AF2 must exclusively rescue at least
   1% of validation targets across at least three classes. It performs no
   training and does not select class-specific weights.
3. Train FCT0 and FCD1 from the same STB1 checkpoint.

FCD1 must pass against both STB1 and FCT0:

- Macro mAP50-95 gain at least +0.5 point;
- Bottom-3 mAP50-95 not lower;
- Worst-class mAP50-95 drop no more than 1 point.

Failure stops FC-STB. No extra seed and no test access. Passing only authorizes
paired multi-seed validation confirmation; it is not a final thesis result.

## Data and test lock

Only grouped Faruq-v3 train/validation development data are available. Test
must not be extracted, mounted, evaluated, or used for decisions in this
screening. All checkpoints and reports are written to the one shared
`Coffee_Bean_Detection` Drive project so account changes can resume safely.
