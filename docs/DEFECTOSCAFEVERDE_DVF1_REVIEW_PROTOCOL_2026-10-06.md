# DefectosCafeVerde DVF1 Pair-Transition Review Protocol

Frozen: **2026-10-06**, after the aggregate seed-42 DVF1 result and before
reading per-pair transitions or running the bootstrap.

## Known result and unresolved question

DVF1 passed its frozen screen against the paper maximum-confidence rule:
physical-pair accuracy improved by 1.06 points, Macro class accuracy by 1.00
point, Bottom-3 by 2.90 points, and worst-class accuracy was unchanged. The
class table showed a 15.15-point `agrio` improvement and a 3.12-point `concha`
regression, with all other class accuracies unchanged.

This review does not retune DVF1. It asks whether the aggregate gain represents
a coherent paired correction pattern sufficient to justify two inexpensive
fuser-only confirmation seeds.

## Frozen analysis

- Endpoint: completed `DVF1_last.pt`, epoch 100, seed 42.
- Population: the same 376 eligible validation physical pairs in the frozen
  `val_pair_cache.pt`.
- Controls: the paper maximum-confidence rule and the saved DVF1 result must
  reproduce exactly.
- Transition categories: both correct, paper-wrong/DVF1-correct rescue,
  paper-correct/DVF1-wrong regression, and both wrong.
- Report transitions globally and per class, predicted-class changes, fused
  margins, and maximum residual magnitude.
- Run an exact two-sided McNemar/binomial test on rescue versus regression.
- Run 10,000 class-stratified physical-pair bootstrap iterations with seed
  20261006. Report point delta, 95% percentile interval, probability positive,
  and probability nonnegative for physical-pair, Macro, Bottom-3, and
  worst-class accuracy.

The bootstrap is a validation diagnostic, not locked-test inference. Classes
are resampled independently with their observed validation support so every
replicate retains all 12 classes.

## Frozen multiseed authorization

Two additional fuser-only seeds are authorized only when:

- the reproduced paper and DVF1 endpoints exactly match the saved result;
- net rescues are at least three physical pairs;
- no more than two paper-correct pairs regress;
- no class loses more than one net-correct pair;
- bootstrap probability of positive physical-pair gain is at least 0.80;
- detector training remains false and test remains unopened.

Pass: `AUTHORIZE_DVF1_FUSER_MULTISEED`.

Fail: `STOP_DVF1_AT_SEED42`; retain the result as exploratory only.

## Locks

- No model, detector, fuser, threshold, or cache is trained or changed.
- No validation selection or hyperparameter adjustment is performed.
- Test is not extracted or accessed.
- A passing review authorizes only fuser seeds 123 and 2026 on the same frozen
  train/validation caches and fixed configuration.
