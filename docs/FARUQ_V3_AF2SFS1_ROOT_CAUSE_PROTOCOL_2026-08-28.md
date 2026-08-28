# Faruq-v3 AF2SFS1 Root-Cause Diagnostic Protocol

Date frozen: 2026-08-28

Status: **AUTHORIZED — VALIDATION-ONLY DIAGNOSTIC**

Training: **forbidden**

Test: **closed**

## Question

Why did `AF2SFS1` improve seed-42 Macro mAP50–95 by 0.95 point and
Bottom-3 by 0.31 point over the matched `AF2CTRL`, while Worst-class AP fell
0.29 point?

The completed aggregate result is insufficient to distinguish among:

1. improved raw localization accessibility;
2. improved final proposal selection/suppression;
3. improved class discrimination after localization;
4. improvements concentrated in specific classes or object sizes;
5. an active inference contribution from the spatial/frequency selector;
6. an optimization-only effect that remains when the selector is bypassed.

## Frozen inputs

- grouped Faruq-v3 development data;
- completed `AF2CTRL_seed42` and `AF2SFS1_seed42` checkpoints;
- their completed arm-result JSON files;
- image size 640;
- IoU threshold 0.50;
- final confidence threshold 0.25;
- raw one-to-one top-500 proposals;
- validation only.

Object-size thresholds are the 33rd and 67th percentiles of normalized box
area computed from **train labels only**. Validation labels do not select the
thresholds.

## Analyses

### Completed AP attribution

Report all 21 per-class AP deltas and identify the five largest improvements
and regressions. These values are read from the completed arm reports; AP is
not approximated by the diagnostic matcher.

### Paired target decomposition

For every validation target and both checkpoints, record:

- raw top-500 proposal accessibility;
- maximum raw proposal IoU;
- final matched status and IoU;
- correct versus wrong class after localization;
- correct-decision recall.

Aggregate globally, per class, and by train-defined object-size bin. Report
paired outcome transitions such as `wrong_to_correct`, `miss_to_correct`, and
`correct_to_wrong`.

### Selector observability

At each target region on P3, report:

- spatial and frequency selector weights;
- selector entropy;
- spatial-path and high-frequency energy;
- frequency contribution fraction;
- residual-to-input magnitude ratio.

Aggregate these measurements globally, per class, per size bin, and by paired
outcome transition. Correlations with per-class AP delta are descriptive and
must not be presented as causal evidence.

### Frozen inference interventions

Without changing checkpoint weights, evaluate the AF2SFS1 detector under:

- `normal`: learned selector and residual;
- `bypass`: adapter output replaced by its input;
- `spatial_only`: selector logits forced to the spatial path;
- `frequency_only`: selector logits forced to the frequency-detail path.

These interventions use the same images, targets, model state, thresholds, and
post-processing. They diagnose active inference dependence; they do not
produce replacement AP claims and cannot authorize model selection.

## Validity gates

The report is interpretable only if:

1. arm codes and seed equal `AF2CTRL`, `AF2SFS1`, and 42;
2. both reports prove test was not accessed;
3. checkpoint paths and arm reports identify the exact seed-42 run, while the
   diagnostic records fresh SHA256 values for both checkpoint inputs;
4. dataset and checkpoint ontologies match exactly and contain 21 classes;
5. no test directory is exposed;
6. selector weights are finite and sum to one;
7. all four inference interventions complete on the same 294 validation
   images;
8. training remains false.

## Interpretation boundary

The diagnostic may attribute the observed seed-42 gain and identify a
mechanism for paired confirmation. It must not tune the selector, alter the
frozen seed-123/2026 confirmation gate, open test, or claim general causal
validity beyond this checkpoint and validation set.
