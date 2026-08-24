# Faruq-v3 AF2 + FFAB2 Parent-Preserving Residual Protocol

Date: 2026-08-24
Status: **frozen before parent-preserving training; validation only; test remains locked.**

## Motivation from completed evidence

The direct matched from-start study rejected `AF2FFAB2FS` as a global AF2
upgrade: mean Macro changed by about -0.07 point, while Bottom-3 improved by
about +1.42 points and Worst-class by about +2.11 points. The subsequent
selectivity analysis found no inference-only strength, P3/P4/P5, parent-mix,
or ambiguity-gating condition that passed its frozen diagnostic gate.

A key diagnostic observation was that turning the trained FFAB residual down to
beta=0 did not restore the AF2FS metrics. This means the completed FFAB2-trained
network cannot be interpreted as "AF2 plus a removable inference multiplier";
the joint training trajectory had already changed the non-adapter network.

This follow-up therefore tests one narrower hypothesis: **can the known FFAB2
spectral residual be learned on top of a completed AF2 parent without allowing
that AF2 parent to move?**

The rejected from-start decision remains rejected. No earlier threshold is
changed.

## Parent and candidate

For each seed 42, 123, and 2026:

- parent: the completed matched `AF2FS` best checkpoint for that same seed;
- candidate: `AF2FFAPR1`;
- source is never D0;
- the AF2 input enhancer, backbone, neck, native box heads, and native
  classification heads are frozen;
- every parent BatchNorm running statistic is frozen in eval mode;
- only the three FFAB adapters (`scale`, `bias`, `alpha`) are trainable;
- regression always consumes the original P3/P4/P5 features;
- all P3/P4/P5 FFAB adapters are active;
- no ambiguity routing is used.

The candidate classification branch is explicitly parent-preserving. For a
feature `F`, native parent logits are `z0 = C(F)`. The spectral adapter produces
`F'`, and the same frozen classifier produces `z1 = C(F')`. The final score is

`z = z0 + (z1 - z0)`.

Because every adapter residual amplitude starts at zero, `F' = F` at
initialization and therefore `z = z0` exactly. The first trainable deviation
comes only through the FFAB adapter.

## Frequency descriptor retained from FFAB2

For every feature channel, FFAB2 keeps the previously tested rFFT high-frequency
energy ratio with radial cutoff 0.35. The residual amplitude remains bounded to
approximately +/-10% with the gradient-matched parameterization. This study
does not introduce DCT, a new attention block, or a new descriptor.

## Negative control

`AF2FFAPR0` is an exact zero-information frozen-parent control with the same
parameter schema and schedule but `conditioning: zero`. Under the frozen parent
and zero-output initialization its adapter has no live information gradient and
is expected to remain the AF2 parent. It is used primarily by the static audit
as a fail-closed negative control; the main three-seed experiment does not need
to spend 90 epochs retraining an intentionally inert arm.

Accordingly, a positive `AF2FFAPR1` result establishes a parent-preserving
spectral-residual architecture result versus its exact AF2FS parent. It does not
by itself separate frequency evidence from every possible active generic
residual parameterization. If the candidate succeeds, an active constant/random
descriptor mechanism control may be justified afterward.

## Frozen schedule

- Development data: leakage-safe Faruq-v3 grouped train/validation only.
- Seeds: 42, 123, 2026.
- Parent: seed-matched completed `AF2FS` best checkpoint.
- Epochs: 30 continuation epochs.
- Image size: 640.
- Batch: 16.
- Workers: 2.
- Patience: 10.
- Optimizer: Ultralytics `auto`, identical across seeds.
- Close mosaic: 10.
- Resume from canonical `last.pt` is allowed.
- Test: locked and must not exist in the development root.

The 30-epoch duration matches the earlier FFAB continuation family while the
parent is now frozen. No parent parameter or running statistic may change.

## Static authorization gates

Before a seed may train, its parent-specific audit must prove:

1. AF2 parent checkpoint exists and is transferred into both residual configs;
2. both residual models are numerically the AF2 parent at initialization;
3. only FFAB adapter parameters are trainable and trainable fraction is <1%;
4. all parent BatchNorm modules remain in eval mode during training mode;
5. parent parameters receive no gradient;
6. a candidate adapter optimizer step leaves all non-adapter state unchanged;
7. the spectral candidate has a finite non-zero adapter gradient;
8. active FFAB changes classification scores while boxes stay bitwise
   unchanged;
9. zero vs spectral descriptor semantics are correct;
10. candidate/control configs differ only in conditioning;
11. test access is false.

Any failed gate stops that seed before training.

## Frozen three-seed decision

`AF2FFAPR1` is compared directly with its three matched `AF2FS` parents. PASS
requires **all** of the original strict upgrade conditions:

- Macro mean gain >= +0.50 point;
- Macro improves in at least 2/3 seeds;
- Bottom-3 mean gain >= +0.50 point;
- Bottom-3 improves in at least 2/3 seeds;
- Worst-class mean delta >= 0;
- Worst-class improves in at least 2/3 seeds.

PASS -> `RETAIN_PARENT_PRESERVING_FFAB2_ON_DEVELOPMENT_VALIDATION`.

FAIL -> `STOP_PARENT_PRESERVING_FFAB2_ROUTE`.

There is no FFAB3/FFAB4 fallback and no post-hoc threshold change.

## Interpretation boundary

This architecture was motivated after examining the same development
validation set. Therefore even a PASS is **development-validation evidence**,
not an independent generalization confirmation. It would justify retaining the
architecture for thesis ablation and, if a genuinely independent set becomes
available, one final confirmation. Test remains unopened in this protocol.

## Implementation

- parent-frozen model/trainer: `src/coffee_detector/af2_ffa/parent_preserving.py`
- parent audit: `src/coffee_detector/af2_ffa/parent_preserving_audit.py`
- arm runner: `run_faruq_v3_af2_ffab2_parent_arm.py`
- decision: `run_faruq_v3_af2_ffab2_parent_decision.py`
- configs: `configs/af2_ffa_parent_preserving/`
- tests: `tests/test_af2_ffab2_parent_preserving.py`
