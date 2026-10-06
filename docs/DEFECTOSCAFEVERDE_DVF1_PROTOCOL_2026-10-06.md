# DefectosCafeVerde DVF1 Dual-View Fusion Protocol

Frozen: **2026-10-06**, after the paired-side audit passed and before DVF1
training or validation evaluation.

## Evidence and question

The validation-only audit found 376 eligible two-side physical beans. The
better fixed side reached 92.55% physical-pair accuracy, the source paper's
maximum-confidence rule reached 94.41%, and the pair oracle reached 97.34%.
Exactly one side was correct for 44 beans across 10 of 12 classes. The 2.93
point oracle headroom over the published rule authorizes one learned fusion
screen.

Question: can a small order-invariant learner recover part of this paired-side
headroom without changing D0 localization or fitting validation?

## Frozen architecture

`DVF1` uses the completed seed-42 `D0DIRECT` detector as a frozen shared
evidence extractor for both physical sides. For each side, it takes the full
12-class score vector of the highest-confidence native one-to-one candidate.
The candidate must calibrate exactly to the detector's final top detection.

The fuser contains:

1. one shared 12-to-24 view encoder;
2. symmetric mean and absolute-difference encoded features;
3. symmetric maximum, mean, and absolute-difference class logits;
4. a bounded residual classifier over the paper's maximum-confidence view.

The final residual layer starts at exactly zero. Therefore the untrained DVF1
endpoint is exactly the paper rule. Swapping Side A and Side B must leave the
output unchanged. At inference, the predicted fused class selects the side
whose native candidate gives that class greater support; the selected native
box is retained. No box regressor, detector parameter, FFT, wavelet transform,
ROI crop, or second detector is introduced.

## Training

- Dataset: grouped DefectosCafeVerde development set.
- Unit: train physical groups with exactly two views, one annotated object per
  view, and matching labels.
- Detector: frozen; it has no optimizer.
- Trainable component: DVF1 fuser only, fewer than 5,000 parameters.
- Seed: 42.
- Fixed schedule: 100 epochs, batch 64, AdamW, learning rate 0.001, weight
  decay 0.001.
- Objective: inverse-square-root class-balanced cross entropy with 0.05 label
  smoothing plus 0.05 KL preservation toward the paper rule.
- There is no validation early stopping, checkpoint selection, or
  hyperparameter search. `last.pt` is saved every epoch and is the only
  evaluated fuser checkpoint.

## Evaluation and controls

Validation is evaluated once after all 100 train-only epochs. The report must
include physical-pair accuracy, macro class accuracy, Bottom-3 class accuracy,
worst-class accuracy, and every class accuracy for:

- Side A;
- Side B;
- the paper maximum-confidence rule;
- DVF1; and
- the non-deployable pair oracle.

Runtime gates require train/validation physical groups to be disjoint, the D0
state fingerprint to remain unchanged, raw candidates to reproduce final D0
candidates, the zero-residual model to reproduce the paper rule, all 12
validation classes, and exact calibration to the prior audit.

## Frozen decision

DVF1 passes only if, relative to the paper maximum-confidence rule:

- physical-pair accuracy improves by at least 0.5 percentage point;
- macro class accuracy does not decrease;
- Bottom-3 class accuracy does not decrease; and
- worst-class accuracy decreases by no more than 2 points.

Pass: `REVIEW_BEFORE_MULTISEED`. Fail: `STOP_DVF1`.

## Locks and claim boundary

- This is one seed and one predeclared fuser, not a search.
- D0 detector training is forbidden.
- Validation fitting and threshold tuning are forbidden.
- Test is not extracted or accessed.
- Pair-oracle performance is an upper bound, never model performance.
- The screen applies only when both physical sides are available at inference.
