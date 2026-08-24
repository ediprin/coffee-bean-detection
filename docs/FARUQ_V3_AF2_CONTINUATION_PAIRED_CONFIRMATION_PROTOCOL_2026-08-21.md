# Faruq-v3 AF2 Continuation Paired Confirmation

Status: **frozen before seed 123/2026 training**
Date: 2026-08-21

## Question

Does the validation improvement observed after continuing AF2 seed 42 for 30
epochs represent a seed-stable optimization effect, rather than a new module
effect or a single-seed fluctuation?

## Frozen design

- Dataset: Faruq-v3 grouped development train/validation only.
- Test: locked and unavailable to every runner in this protocol.
- Baseline: completed AF2 checkpoints for seeds 42, 123, and 2026.
- Candidate: the same AF2 architecture continued for 30 epochs (`AF2CT30`).
- Seed 42: reuse frozen `AF2FT30` evidence; do not retrain.
- Seeds 123 and 2026: start from the corresponding completed AF2 checkpoint.
- Model, AF2 operator, input size, optimizer schedule, and validation evaluator
  are identical to the frozen seed-42 continuation control.
- No hyperparameter selection and no external/test evaluation occur here.

## Primary metrics and gate

Paired deltas are `AF2CT30 - AF2` for each seed. PASS requires all of:

1. mean Macro mAP50-95 gain at least +0.5 percentage point;
2. Macro improves in at least two of three seeds;
3. mean Bottom-3 mAP50-95 is not lower and improves in at least two seeds;
4. mean Worst-class mAP50-95 is not lower and improves in at least two seeds.

PASS retains a two-stage AF2 optimization protocol. FAIL retains original AF2
and closes continuation as a single-seed effect. The locked test is not opened
under either outcome.

PASS establishes seed stability, not the causal superiority of restart over an
uninterrupted 80-epoch budget. That attribution requires a separate frozen,
budget-matched control; it is not inferred from this confirmation.
