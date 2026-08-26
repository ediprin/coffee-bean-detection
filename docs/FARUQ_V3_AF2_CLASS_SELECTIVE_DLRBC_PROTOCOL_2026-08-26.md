# Faruq-v3 AF2 Class-Selective DLRBC Protocol — 2026-08-26

## Research question

Can DLRBC complement AF2 only on classes where an independently trained DLRBC model has train-only evidence of complementarity, without changing AF2 scores for other classes or its box regression?

This is a new exploratory screen. It does not reopen or revise the failed global DLRBC fresh result.

## Frozen mechanism

- Parent: completed `AF2DIRECT_seed42` checkpoint from the direct-from-pretrained protocol.
- The entire AF2 detector is frozen, including BatchNorm statistics.
- A rank-8 quadratic DLRBC residual is attached to each native classification tower.
- A per-class bounded `tanh` gate is initialized at zero.
- A persistent class mask makes the raw residual exactly zero for every non-selected class.
- The box branch is never modified.
- Initial candidate output must be exactly equal to AF2.

## Class selection

Class selection uses only train-split classwise AP from completed AF2DIRECT and DLRBC_FRESH checkpoints. A class is eligible when DLRBC train AP exceeds AF2 train AP by at least 2 points. The selected set must contain 2–10 classes. Validation is not read during selection and no hyperparameter is chosen from validation.

If this train-only complementarity gate fails, training is forbidden.

## Seed-42 screen

- Dataset: leakage-safe Faruq-v3 grouped development set.
- Epochs: 20; early stopping patience: 8.
- Optimizer: AdamW, learning rate 0.001.
- Trainable parameters: only selective residual and gate parameters.
- Validation is evaluated once after completion.
- Test remains unavailable.

Relative to AF2DIRECT seed 42, `RETAIN_SEED42_PARETO` requires all of:

1. Macro mAP50–95 drop no more than 0.1 point.
2. Bottom-3 mAP50–95 not lower.
3. Worst-class mAP50–95 drop no more than 0.5 point.
4. Mean AP of the train-selected classes improves by at least 0.5 point.
5. All 21 validation classes are present.

Failure stops the direction without test or extra seeds. Passing only authorizes a future paired multi-seed protocol; it is not a superiority claim.

## Static gates

- zero gate is exactly AF2 for raw boxes and raw scores;
- an active gate changes selected scores;
- non-selected scores remain bitwise equal;
- raw boxes remain bitwise equal;
- gradients are finite and the gate receives a nonzero gradient;
- no test access.
