# Faruq-v3 AF2 + FFAB2 selective-refinement follow-up

Date: 2026-08-24
Status: **frozen before running this follow-up; validation only until a diagnostic candidate passes the predeclared gate. Test remains locked.**

## Why this follow-up exists

The matched from-start three-seed experiment rejected FFAB2 as a global AF2 upgrade because Macro mAP50-95 changed by about -0.07 point, even though Bottom-3 improved by about +1.42 points and Worst-class improved by about +2.11 points. The older continuation experiment showed a similar lower-tail gain but a positive Macro gain. The follow-up therefore tests one specific hypothesis: **the FFAB2 frequency signal may be useful, but its classification recalibration may be insufficiently selective.**

This protocol does not revise the rejected Stage-1 decision and does not change its thresholds post hoc.

## Stage A — no-training diagnosis

Inputs are the completed `AF2FS` and `AF2FFAB2FS` validation results/checkpoints for seeds 42, 123, and 2026. Test data must not be present in the development root.

### A1. Per-class delta matrix

For every class `c` and seed `s`:

`delta_AP[c,s] = AP_FFAB2[c,s] - AP_AF2[c,s]`.

Report mean delta, minimum/maximum delta, and number of improved seeds. This identifies which classes pay for the lower-tail gain.

### A2. FFAB2 residual-strength sweep

Reuse the trained FFAB2 checkpoint without training and evaluate:

`beta in {0, 0.25, 0.50, 0.75, 1.0}`

with all P3/P4/P5 adapters active. Runtime beta multiplies only the learned FFAB2 residual amplitude; it does not alter checkpoint parameters.

### A3. P3/P4/P5 bypass ablation

At beta=1, evaluate all non-empty level subsets:

- P3
- P4
- P5
- P3+P4
- P3+P5
- P4+P5
- P3+P4+P5

Regression remains untouched in every condition.

### A4. Parent-preserving logit interpolation

For the same completed checkpoint, compute native classification scores `z0` from the unadapted feature and refined scores `z1` from the adapted feature, then evaluate:

`z = z0 + lambda * (z1 - z0)`

for `lambda in {0.25, 0.50, 0.75, 1.0}`.

This is an inference-only diagnostic of whether preserving the parent classification path recovers Macro. It is not evidence that a newly trained parent-residual architecture will behave identically.

### A5. Ambiguity-gated parent residual

Using native sigmoid class scores, define top-1/top-2 margin `m`. The detached gate is:

`g = sigmoid((tau - m) / T)`

with `T=0.05` and `tau in {0.05, 0.10, 0.15, 0.20}`. Then:

`z = z0 + g * (z1 - z0)`.

The gate is large only where the native classifier is ambiguous. It is detached so the classifier cannot improve the objective merely by manipulating the routing variable.

## Frozen exploratory authorization gate

A runtime condition is eligible for **fresh retraining only** when all conditions hold versus the matched `AF2FS` controls:

- Macro mean gain >= +0.25 point;
- Macro improves in at least 2/3 seeds;
- Bottom-3 mean gain >= +0.50 point;
- Bottom-3 improves in at least 2/3 seeds;
- Worst mean delta >= 0;
- Worst improves in at least 2/3 seeds.

These thresholds authorize a retraining screen; they are not a confirmation claim. Because the runtime search uses validation, the selected setting is validation-tuned.

If several conditions pass, select the one with the largest Macro delta, then Bottom-3 delta, then Worst delta.

## Stage B — conditional fresh retraining

Only if Stage A finds an eligible setting, train one candidate `AF2FFASR1` for seeds 42, 123, and 2026. Each seed:

- starts from the same seed-matched D0 used by its AF2FS control;
- uses AF2 from the first epoch;
- uses the selected FFAB2 strength/levels/fusion/gate from the first epoch;
- uses the same 50-epoch training schedule as AF2FS/AF2FFAB2FS;
- keeps regression on the original P3/P4/P5 features;
- evaluates validation only;
- keeps test locked.

For parent-residual modes, the FFAB adapter is zero-initialized, so the initial refined feature equals the native feature. The explicit native classification path is preserved and FFAB contributes through the residual formulation.

## Stage B frozen confirmation gate

The confirmation gate is deliberately restored to the original stricter Stage-1 thresholds:

- Macro mean gain >= +0.50 point;
- Macro improves in at least 2/3 seeds;
- Bottom-3 mean gain >= +0.50 point;
- Bottom-3 improves in at least 2/3 seeds;
- Worst mean delta >= 0;
- Worst improves in at least 2/3 seeds.

PASS means only that the selective candidate improves AF2 under matched from-start validation retraining. Because Stage A selected the candidate on validation, an independent confirmation is still required before treating it as a final generalization claim.

FAIL means stop the selective FFAB2 route. Do not create FFAB3/FFAB4 by changing thresholds after seeing the result.

## Implementation

- Runtime/selective controls: `src/coffee_detector/af2_ffa/model.py`
- Diagnosis: `run_faruq_v3_af2_ffab2_selectivity_analysis.py`
- Conditional retraining: `run_faruq_v3_af2_ffab2_selective_arm.py`
- Confirmation: `run_faruq_v3_af2_ffab2_selective_decision.py`
- Tests: `tests/test_af2_ffab2_selective_refinement.py`

No DCT stage is authorized by this protocol. DCT remains stopped because the original from-start FFAB2 Stage-1 was rejected.
