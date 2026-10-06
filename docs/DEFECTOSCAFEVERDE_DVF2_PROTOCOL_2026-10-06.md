# DefectosCafeVerde DVF2 Selective Dual-View Fusion Protocol

Frozen: **2026-10-06**, after the completed DVF1 seed-42 transition review and
before DVF2 training or evaluation.

## Scientific status and question

DVF1 improved the paper maximum-confidence rule by four net-correct physical
pairs: five rescues and one regression. It captured 1.06 of the 2.93 points of
pair-oracle headroom, while 16 eligible validation pairs remained wrong. The
review authorized confirmation of DVF1, but the user explicitly chose one
final architecture-maximization screen before confirmation.

Because the DVF2 design follows inspection of reused validation aggregates,
this is an **exploratory architecture screen**, not independent confirmation.
No DVF2 choice may be tuned after its validation result. Test remains locked.

Question: can an explicit train-supervised fallback gate retain the published
paper rule on pairs that do not need correction while a richer symmetric
expert recovers more dual-view complementarity than DVF1?

## Frozen architecture

`DVF2` reuses the immutable DVF1 train and validation pair caches produced by
the frozen seed-42 D0 detector. The detector is not loaded or trained.

The fuser is order invariant and contains:

1. a shared 12-to-32 encoder for each view;
2. symmetric encoded mean and absolute difference;
3. per-class maximum, mean, absolute difference, and cross-view probability
   product;
4. symmetric entropy, margin, and cross-view cosine agreement statistics;
5. a bounded 48-hidden-unit correction expert; and
6. a separate 24-hidden-unit fallback gate.

At initialization the correction expert is exactly zero, so DVF2 equals the
paper maximum-confidence rule. During training the gate is differentiable. At
inference a fixed probability threshold of 0.5 is used: gate-negative pairs
return the paper endpoint exactly; gate-positive pairs receive the bounded
expert correction. Swapping the two views must not change the result.

## Frozen training

- Unit: the same 1,266 eligible train physical pairs cached for DVF1.
- Seed: 42.
- Schedule: 100 epochs, batch 64, AdamW, learning rate 0.001, weight decay
  0.001; no early stopping or validation checkpoint selection.
- Classification: inverse-square-root class-balanced cross entropy with 0.05
  label smoothing.
- Gate target: whether the paper maximum-confidence class is wrong on the
  train pair; balanced binary cross entropy, weight 0.25.
- Rescue emphasis: additional cross entropy on paper-wrong train pairs,
  weight 0.5.
- Preservation: KL toward the paper endpoint on paper-correct train pairs,
  weight 0.10.
- `DVF2_last.pt` is saved each epoch and epoch 100 is the only evaluated
  checkpoint.

## Frozen evaluation and decision

Validation is evaluated once on the unchanged 376-pair cache. Report paper,
DVF1, DVF2, and pair-oracle physical-pair, Macro, Bottom-3, worst-class, and
per-class accuracy, plus gate activation overall and by paper-correct status.

DVF2 is retained when either:

- physical-pair accuracy improves at least 0.5 point over DVF1, Macro and
  Bottom-3 do not fall, and worst-class falls no more than 2 points; or
- physical accuracy and Macro do not fall, Bottom-3 improves at least 2
  points, and worst-class falls no more than 2 points.

Pass: `REVIEW_DVF2_BEFORE_CONFIRMATION`. Fail: `RETAIN_DVF1`.

## Locks

- No validation tuning, threshold sweep, early stopping, detector training,
  ROI crop, or second detector.
- DVF1 cache hashes and endpoints must reproduce exactly.
- Test is not extracted or accessed.
- If DVF2 fails, no DVF3 is authorized from this validation split.
