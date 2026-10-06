# DefectosCafeVerde Dual-View Side-Reliability Routability Protocol

Frozen: **2026-10-06**, after DVF1 three-seed confirmation and before this
train-only audit is executed.

## Motivation and question

The source paper selects one of two physical bean views by maximum detector
confidence. On 376 eligible validation pairs, that rule is correct for 355
pairs, DVF1 for 359, and the non-deployable pair oracle for 366. DVF1 therefore
captures four of eleven oracle-recoverable errors. Seven DVF1 errors still have
one native side that is already correct, while ten pairs have neither side
correct.

DVF2 showed that a more expressive post-logit correction expert did not improve
the aggregate endpoint. The next question is narrower: **can inference-visible
score-distribution evidence identify which native side is reliable, without
synthesizing a new class or modifying boxes?**

## Frozen train-only audit

- Input: the immutable DVF1 `train_pair_cache.pt` produced by the frozen
  seed-42 D0 detector.
- Scope: grouped train pairs only. Validation and test caches are forbidden.
- Each view is represented by its 12 native class scores, predicted-class
  one-hot vector, maximum and second score, margin, normalized class entropy,
  and total score mass.
- The selector is a shared linear reliability scorer. For a pair, it scores
  the feature difference between Side A and Side B. Swapping sides negates the
  score and therefore swaps the decision exactly.
- Supervision uses only train pairs where exactly one native side is correct.
  The scorer learns which side is correct through fixed ridge regression
  (`alpha=10`) with sign-augmented samples and no intercept.
- Five folds are assigned deterministically from physical group ID. Every OOF
  decision is made by weights fitted without that physical pair.
- Controls: paper maximum confidence, maximum top-1 margin, minimum entropy,
  learned OOF selector, and the pair oracle.
- A final full-train scorer is serialized only as a prospective artifact; it
  is not evaluated on validation by this audit.

No detector, DVF1 fuser, or validation endpoint is trained or evaluated.

## Frozen authorization gate

One later validation screen (`DVSR1`) is authorized only when all conditions
hold on train-only OOF predictions:

1. pair-oracle gain over paper maximum confidence is at least 1.0 point;
2. at least 50 train pairs contain exactly one correct native side;
3. learned selector gain over paper is at least 0.5 point;
4. learned selector captures at least 25% of oracle headroom;
5. learned selector produces at least three net rescues; and
6. at least three of five folds have positive rescue-minus-regression count.

Pass: `AUTHORIZE_DVSR1_VALIDATION_SCREEN`.

Fail: `STOP_SIDE_RELIABILITY_ROUTING` and retain confirmed DVF1.

## Locks and claim boundary

- validation access: forbidden;
- test extraction or access: forbidden;
- threshold, feature, fold, alpha, or gate changes after seeing the audit:
  forbidden;
- this audit measures train-only OOF routability, not validation superiority;
- the prospective selector may only choose an existing native side prediction.
