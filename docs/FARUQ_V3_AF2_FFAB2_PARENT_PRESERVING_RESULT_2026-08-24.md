# Faruq-v3 AF2 + FFAB2 parent-preserving result

Date: 2026-08-24

Status: **FINAL — REJECT**

Decision: `REJECT`

Next: `STOP_PARENT_PRESERVING_FFAB2_ROUTE`

Test opened: **false**

## Research question

This follow-up tested whether a classification-only FFAB2 residual can add useful frequency-conditioned refinement on top of a completed AF2FS detector while preserving the trained AF2 parent. The seed-matched AF2FS checkpoint was frozen, including BatchNorm state, and only the FFAB2 adapter parameters were trainable.

Comparison: `AF2FFAPR1_vs_frozen_AF2FS_parent`

Seeds: 42, 123, 2026.

## Frozen decision gate

- Macro mean gain >= +0.50 pp and Macro improves in at least 2/3 seeds.
- Bottom-3 mean gain >= +0.50 pp and Bottom-3 improves in at least 2/3 seeds.
- Worst-class mean delta >= 0 and Worst improves in at least 2/3 seeds.

The numerical gate representation in the decision artifact used fractions: Macro/Bottom-3 mean gain minimum `0.005`, Worst mean delta minimum `0.0`.

## Three-seed result

| Metric | Frozen AF2FS parent mean | AF2FFAPR1 mean | Mean delta | Improved seeds | Gate |
|---|---:|---:|---:|---:|---|
| Macro mAP50-95 | 87.6195% | 87.6252% | **+0.0057 pp** | 2/3 | FAIL mean-gain gate |
| Bottom-3 class mAP50-95 | 78.2931% | 78.2708% | **-0.0223 pp** | 0/3 | FAIL |
| Worst-class mAP50-95 | 75.1328% | 75.1112% | **-0.0216 pp** | 0/3 | FAIL |

### Per-seed values

| Seed | Parent Macro | Candidate Macro | Delta Macro | Parent B3 | Candidate B3 | Delta B3 | Parent Worst | Candidate Worst | Delta Worst |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 88.1973% | 88.2017% | +0.0044 pp | 80.0428% | 80.0428% | 0.0000 pp | 79.3470% | 79.3470% | 0.0000 pp |
| 123 | 87.4176% | 87.4173% | -0.0003 pp | 75.9345% | 75.9345% | 0.0000 pp | 72.8440% | 72.8440% | 0.0000 pp |
| 2026 | 87.2436% | 87.2566% | +0.0130 pp | 78.9019% | 78.8351% | -0.0668 pp | 73.2075% | 73.1428% | -0.0648 pp |

Standard deviations recorded by the decision artifact:

- Macro: parent `0.0050789992`, candidate `0.0050569281`.
- Bottom-3: parent `0.0212070591`, candidate `0.0211144967`.
- Worst: parent `0.0365409401`, candidate `0.0367130892`.

## Frozen criteria

- `macro_mean_gain_at_least_0_5pp`: **false**
- `macro_improves_at_least_2_of_3`: **true**
- `bottom3_mean_gain_at_least_0_5pp`: **false**
- `bottom3_improves_at_least_2_of_3`: **false**
- `worst_mean_not_lower`: **false**
- `worst_improves_at_least_2_of_3`: **false**

Final decision: **REJECT**.

## Interpretation

The parent-preserving FFAB2 residual did not provide a meaningful incremental improvement over the frozen AF2FS parent. Macro was effectively unchanged (+0.0057 pp), while Bottom-3 and Worst were slightly lower and did not improve in any seed.

This result closes the parent-preserving FFAB2 route under the frozen three-seed development-validation protocol. It does **not** invalidate the earlier staged continuation result, whose causal comparator was the equally continued zero-information AF2FFA0 control, and it does **not** revise the matched from-start result. The three regimes answer different questions.

Across the FFAB2 experiment chain:

1. staged continuation vs AF2FFA0: +1.44 Macro / +2.83 Bottom-3 / +2.27 Worst pp — PASS against the continuation control;
2. matched from-start vs AF2FS: -0.0687 Macro / +1.4151 Bottom-3 / +2.1082 Worst pp — REJECT because the frozen Macro gate failed;
3. selective runtime diagnosis: no runtime candidate passed the diagnostic gate;
4. frozen-parent AF2FFAPR1 vs AF2FS: +0.0057 Macro / -0.0223 Bottom-3 / -0.0216 Worst pp — REJECT.

A mechanistic interpretation consistent with this chain is that the FFAB2 tail benefit observed when the network is allowed to adapt jointly is not reproduced by training FFAB2 alone on a fixed mature AF2 representation. This is an evidence-consistent interpretation, not proof of a unique causal mechanism.

## Important boundary

The decision artifact recorded `zero_control: null`; therefore this completed run is a direct spectral-candidate-versus-frozen-parent test, not a completed spectral-versus-zero-information residual comparison. This does not alter the REJECT decision because AF2FFAPR1 itself failed to improve the frozen parent under the preregistered gate.

The follow-up was motivated after development-validation inspection. Even a hypothetical PASS would therefore have remained development-validation evidence rather than independent test/generalization confirmation.

Test remained locked throughout this follow-up.
