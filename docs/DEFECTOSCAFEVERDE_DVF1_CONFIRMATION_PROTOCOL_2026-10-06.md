# DefectosCafeVerde DVF1 Paired Fuser Confirmation Protocol

Frozen: **2026-10-06**, after DVF2 failed its single exploratory screen and
before training confirmation seeds 123 or 2026.

## Evidence and question

Seed-42 DVF1 improved the source paper maximum-confidence rule by 1.06 points
physical-pair accuracy, 1.00 point Macro accuracy, and 2.90 points Bottom-3,
with unchanged worst-class accuracy. Its frozen transition review found five
rescues and one regression and authorized two fuser-only confirmation seeds.

DVF2 subsequently produced one rescue and one regression relative to DVF1,
leaving every headline metric effectively unchanged. DVF2 was rejected and no
further architecture variant is authorized on this validation split.

Question: does the fixed DVF1 gain persist across fuser initialization and
train-order seeds when the detector evidence and all other choices remain
identical?

## Frozen paired design

- Candidate: the exact `DVF1` architecture and `DVF1.yaml` configuration.
- Evidence: immutable seed-42 D0 train and validation pair caches.
- Detector: not loaded, trained, or evaluated again.
- Seeds: 42 (completed), 123, and 2026.
- New runs: seed 123 and seed 2026 only.
- Schedule per new seed: 100 epochs, batch 64, AdamW, learning rate 0.001,
  weight decay 0.001, label smoothing 0.05, preservation weight 0.05.
- No validation early stopping, checkpoint selection, threshold tuning, or
  hyperparameter change. Epoch 100 `last.pt` is the only endpoint.
- Test is not extracted or accessed.

Each new seed runs independently and writes a resumable checkpoint every
epoch. The completed DVF1 seed-42 result, passing transition review, and failed
DVF2 result are mandatory immutable inputs.

## Frozen metrics and decision

For each seed, report physical-pair, Macro, Bottom-3, worst-class, and all
class accuracies for the paper rule, DVF1, and pair oracle. The paper and oracle
endpoints must reproduce seed 42 exactly.

Across three seeds, DVF1 passes only when:

- mean physical-pair gain over the paper rule is at least 0.5 point;
- physical-pair accuracy improves in at least two of three seeds;
- mean Macro accuracy is not lower and improves in at least two seeds;
- mean Bottom-3 accuracy is not lower and improves in at least two seeds; and
- mean worst-class accuracy falls no more than 2 points.

Pass: `PROMOTE_DVF1_DUAL_VIEW_FUSION`.

Fail: `REPORT_DVF1_AS_EXPLORATORY_SEED42_ONLY`.

## Claim boundary

This confirms fuser-seed stability on reused validation evidence. It is not an
independent locked-test result, does not establish single-view improvement,
and applies only when both physical sides of a bean are available at inference.
