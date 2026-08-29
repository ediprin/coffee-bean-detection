# Faruq-v3 AF2-SPDS Refinement Result — 2026-08-29

## Decision

**Formal frozen decision: FAIL_KILL_GATE; retain original AF2.** Neither
refinement is authorized for confirmatory extra seeds or test evaluation under
the frozen protocol. Separately, AF2CUE1 is retained as a
**Pareto-exploratory candidate** for validation-only analysis because the
failure margin was only 0.21 point on one gate and it improved all three
headline metrics over AF2BASE.

| Arm | Macro mAP50–95 | Bottom-3 | Worst class | Decision |
|---|---:|---:|---:|---|
| AF2BASE | 88.89% | 80.84% | 77.53% | matched base |
| AF2SPDS | 88.65% | **84.67%** | 83.44% | prior treatment |
| **AF2CUE1** | **89.31%** | 83.96% | **83.58%** | REJECT near-miss |
| AF2DECAY1 | 88.49% | 82.81% | 80.39% | REJECT |

## Causal comparisons

`AF2CUE1 - AF2BASE` was **+0.42/+3.12/+6.05 points** for
Macro/Bottom-3/Worst. Relative to original AF2SPDS it was
**+0.66/-0.71/+0.14 points**.

AF2CUE1 therefore recovered Macro and preserved the large Worst-class gain,
but missed the frozen Bottom-3-retention requirement. The protocol allowed at
most a 0.50-point Bottom-3 loss from AF2SPDS; the observed loss was 0.71 point,
missing the boundary by 0.21 point. This is scientifically relevant near-miss
evidence, but the threshold cannot be revised after observing validation.

`AF2DECAY1 - AF2BASE` was **-0.40/+1.96/+2.86 points**, and relative to
AF2SPDS it was **-0.16/-1.86/-3.05 points**. Late loss decay did not preserve
the SPDS advantage and is rejected without ambiguity.

## Interpretation

The pure normalized AF2-gate target is substantially better aligned with the
detector than reconstructing the RGB-coupled residual `AF2(x)-x`. That supports
the diagnosis that the original SPDS target mixed frequency structure with
raw appearance. However, AF2CUE1 still trades some Bottom-3 performance for
Macro, so it does not dominate AF2SPDS under the pre-registered objective.

This result closes AF2CUE1/AF2DECAY1 as *confirmed thesis-model replacements*
under this protocol. It does not justify erasing the numerical evidence:
AF2CUE1 remains `RETAIN_PARETO_EXPLORATORY`, while AF2DECAY1 is rejected. The
post-hoc per-class and paired-class-composition bootstrap is implemented in
`coffee_detector.analysis.af2_spds_refinement_posthoc`; it cannot override the
frozen decision and is not an image- or parent-level inferential test.

AF2CUE1 also does not collide with AF2RN: AF2CUE1 changes a removable
training-only target, whereas AF2RN changes the parameter-free input angular
statistic.

Test was not opened. Raw evidence is stored in
`docs/evidence/FARUQ_V3_AF2_SPDS_REFINEMENT_SEED42_2026-08-29.json`.
