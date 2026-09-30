# DefectosCafeVerde Dual-View Complementarity Audit

Frozen: **2026-09-30**, before running the audit and before implementing a
learned dual-view model.

## Research question

Does the opposite face of the same physical bean contain complementary
information that the completed native detector can already recognize, and is
there enough unresolved headroom beyond the source paper's max-confidence
fusion rule to justify one learned dual-view architecture?

This question follows from the negative single-image experiments. AF2DIRECT,
DCWCF1, LIFRPF1, RAFC1, RATF1, and FSRC1 did not beat the native D0DIRECT
endpoint. The source paper's actual acquisition contribution is different: it
captures Side A and Side B of the same bean. The grouped rebuild preserved
1,969 two-view physical groups and kept each group inside one split, but every
model screen so far has processed each image independently.

The paper combines the two independent YOLO decisions using maximum
confidence. It does not learn a cross-view feature representation. A learned
fusion model is therefore considered only if the leak-safe grouped validation
split demonstrates both cross-side complementarity and headroom beyond that
published rule.

## Frozen endpoint and eligible population

- Dataset: `defectoscafeverde-grouped-physical-v1` development root.
- Split: validation only; the test directory must not exist.
- Endpoint: completed 50-epoch `D0DIRECT` seed 42 with exact result/checkpoint
  SHA verification.
- Eligible unit: one inferred physical group with exactly two source views,
  exactly one annotated object in each view, and the same expert class label on
  both views.
- View order: numeric source filename order inside the inferred physical group.

Groups with one view, more than one object per view, or disagreeing paired
labels are reported and excluded rather than forced into a correspondence.

## Decisions measured without training

For each eligible physical bean, the frozen D0 detector runs independently on
both views. The deployable candidate for each view is its highest-confidence
final detection at confidence 0.001. It is correct only if its class is correct
and its IoU to the single annotated bean is at least 0.50.

The audit reports:

1. Side-A and Side-B physical-bean accuracy;
2. the better single-side accuracy;
3. the source paper's deployable max-confidence rule;
4. a pair oracle that is correct when either side is correct;
5. both/exactly-one/neither-side-correct counts;
6. the same quantities per class; and
7. cross-side complementarity in D0DIRECT's worst AP class.

The pair oracle uses the ground truth only to measure an upper bound. It is not
a deployable method and is never reported as model performance.

## Frozen authorization gate

One fresh learned dual-view architecture screen is authorized only when all of
the following hold:

- eligible pairs cover at least 75% of validation physical groups;
- the pair oracle exceeds the better single side by at least 1.0 percentage
  point;
- the pair oracle exceeds the paper max-confidence rule by at least 0.5 point;
- exactly-one-side-correct complementarity occurs in at least three classes.

Worst-class complementarity is reported as a stronger diagnostic, not made a
mandatory gate because a dual-view architecture may address localized defects
without repairing a globally chromatic worst class.

Pass: `AUTHORIZE_ONE_LEARNED_DUAL_VIEW_ARCHITECTURE_SCREEN`.

Fail: `STOP_LEARNED_DUAL_VIEW_FUSION`; no dual-view model is trained.

## Locks

- No detector, classifier, router, or fusion model is trained.
- No threshold, pairing rule, eligibility rule, or gate changes after output.
- No validation fitting is performed.
- Test is not extracted or accessed.
- The inferred physical identity limitation remains explicit; author-provided
  pair identifiers are unavailable.
