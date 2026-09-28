# DefectosCafeVerde Train-Only Expert Routability Protocol

Frozen: **2026-09-29**

## Research question

The completed validation complementarity audit found a 5.71-point object-level
oracle gain over `D0DIRECT`.  That oracle used the ground-truth class to choose
an expert and is not deployable.  This protocol asks the necessary next
question: can an expert be selected from signals available at inference time?

No detector is trained.  No validation result is used to fit or select the
router, and test remains absent.  All fitting and out-of-fold evaluation use
the grouped physical **train split only**.

## Frozen endpoints

The endpoints and exact seed-42, 50-epoch checkpoints remain:

`D0DIRECT`, `AF2DIRECT`, `DCWCF1`, `LIFRPF1`, and `RAFC1`.

Their result contracts, dataset-audit SHA, checkpoint SHA, ontology, seed, and
training schedule must pass the same checks as the validation complementarity
audit.

## Deployable association and features

`D0DIRECT` final detections at confidence 0.001 are the object anchors.  Ground
truth is matched to anchors only to score the audit.  Candidate detections are
associated to D0 anchors class-agnostically by one-to-one IoU matching at 0.30.
An expert is counted correct only when its associated box has IoU at least 0.50
to the target and predicts the target class.

Router inputs never include target class, target box, target IoU, correctness,
or validation statistics.  They contain only:

- per-expert association presence, confidence, anchor IoU, and predicted-class
  one-hot encoding;
- inter-expert class-vote counts;
- RGB/luminance mean and standard deviation plus horizontal/vertical gradient
  energy for the complete input and D0-anchor crop.

Images use the native Ultralytics inference order: OpenCV BGR input is
converted to RGB before the detector forward pass.  The original v1 audit used
the diagnostic helper's unconverted BGR tensor and is retained only as a
non-native exploratory run.  The frozen decision is based exclusively on the
`v2-rgb-native` output.

Targets without a D0 anchor are retained in the denominator and cannot be
rescued by this routing formulation.

## Train-only evaluation

Images, not objects, are assigned deterministically to five folds.  All objects
from one image remain in one fold.  The primary router is a fixed unweighted
multiclass ridge classifier (`alpha=10`) fitted on four folds and evaluated on
the held-out fold.  The target prefers D0 whenever D0 is correct; otherwise it
selects the highest-confidence correct expert.  This target is used only in
the training folds.

Controls are:

- D0 anchor decisions;
- every expert alone under the same association;
- highest-confidence expert without fitting;
- full and subset oracle ceilings.

All subsets containing D0 are enumerated.  The minimal subset is the smallest
one retaining at least 90% of the full oracle gain; ties use higher oracle
correctness and then endpoint order.  Subset selection is repeated inside each
training fold so held-out outcomes cannot choose their experts.  The full-train
subset is reported only as the prospective architecture choice after the OOF
pipeline has been scored.

## Frozen gate

Routing is justified for one later architecture screen only if all conditions
hold:

1. associated full oracle gain over D0 is at least 1.0 point;
2. the selected subset retains at least 90% of that gain;
3. out-of-fold router gain over D0 is at least 0.5 point;
4. the router captures at least 25% of the selected-subset oracle gain;
5. the router is not below the best standalone associated expert.

Failure stops routed fusion.  Passing does not establish validation or test
superiority; it only authorizes a separately frozen, single-seed architecture
screen.

## Locks

- detector training: forbidden;
- validation inference or fitting: forbidden;
- test extraction or access: forbidden;
- manual threshold, fold, feature, subset, or regularization changes after
  observing the output: forbidden.

Per-image target records and an exact dataset/checkpoint/settings contract are
persisted so an interrupted audit can resume without mixing runs.

