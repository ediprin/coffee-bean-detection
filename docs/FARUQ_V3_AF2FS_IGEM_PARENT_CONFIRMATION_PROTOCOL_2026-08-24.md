# Faruq-v3 AF2FS + IGEM Frozen-Parent Paired Confirmation

Date frozen: 2026-08-24  
Branch: `codex/af2-igem-parent-confirmation`  
Status: **FROZEN BEFORE TRAINING**  
Evaluation: Faruq-v3 grouped development validation only  
Locked test: **closed**

## Research question

The earlier `AF2IGEM1` direct combination was trained jointly from D0, so it did not answer whether IGEM can add useful classification evidence to an already trained AF2 without changing the AF2 representation. The older parent-residual protocol answered this only as a seed-42 screen using a historical AF2 parent.

This confirmation asks a narrower and stronger question:

> With each completed AF2FS parent frozen, does a classification-only IGEM residual receiving the real frozen AF2 P3/P4/P5 features outperform an architecture-, capacity-, schedule-, and optimization-matched zero-information residual across seeds 42/123/2026?

This is a new preregistered study. It does not revise the 2026-08-23 seed-42 protocol and does not retroactively reinterpret any previous result.

## Frozen parents

The parent for each seed is the completed `AF2FS` best checkpoint from the already completed AF2FFAB2 from-start state:

- seed 42 candidate/control -> AF2FS seed 42 `best.pt`;
- seed 123 candidate/control -> AF2FS seed 123 `best.pt`;
- seed 2026 candidate/control -> AF2FS seed 2026 `best.pt`.

The notebook must resolve each checkpoint by the SHA-256 recorded in the corresponding `AF2FS_seed{seed}_result.json`; stale absolute paths are not accepted.

For every run, the AF2 frontend, YOLO backbone, neck, native localization path, and native classification path are frozen. Frozen parent modules stay in evaluation mode during residual training. Only IGEM residual parameters are optimized.

## Matched arms

| Arm | Conditioning entering IGEM residual | Purpose |
|---|---|---|
| `AF2IGEM0` | zeros shaped like P3/P4/P5 | matched capacity/optimization control |
| `AF2IGEM1` | real frozen AF2 P3/P4/P5 | spectral-parent IGEM candidate |

The two arms use the existing `configs/af2_parent_residual/AF2IGEM0.yaml` and `AF2IGEM1.yaml`. Their only intended residual difference is `conditioning: zero` versus `conditioning: feature`.

The existing IGEM settings remain frozen: reference depth 3, mask-loss weight 0.05, kernel size 3, four attention heads, channel reduction 4, correction scale 1.0. Training remains 20 epochs, image size 640, batch 16, workers 2, patience 7, optimizer `auto`, no cache, close-mosaic 10, and max-det 500.

The IGEM coarse bbox-derived mask objective remains present in **both** arms. Therefore the comparison isolates information entering the residual rather than auxiliary-loss presence.

## Forward contract

For level `l`, native AF2 classification logits are preserved and the residual contributes only an additive correction:

\[
z_l = z_l^{AF2} + \Delta z_l^{IGEM}.
\]

For the matched control, the residual receives `0` instead of `F_l`:

\[
\Delta z_l^{IGEM0}=R_l(0), \qquad
\Delta z_l^{IGEM1}=R_l(F_l).
\]

The box branch always consumes the native frozen AF2 features and cannot be modified by the residual.

At zero-output initialization both arms must reproduce the frozen AF2 parent numerically before training.

## Static authorization gate

Training for a seed is forbidden unless the IGEM-only static audit records `PASS` and binds itself to that seed's exact AF2FS checkpoint SHA. The audit must establish:

1. `AF2IGEM0` and `AF2IGEM1` have identical model YAML, AF2 config, training schedule, parameter count, trainable count, and state schema;
2. the only intended residual-config difference is zero versus feature conditioning;
3. both arms numerically reproduce the AF2FS parent at zero initialization;
4. activating the feature-conditioned residual changes class scores while preserving boxes bitwise;
5. activating the zero-conditioned residual cannot extract P3/P4/P5 information;
6. gradients are finite/nonzero in the residual and absent from every parent parameter;
7. the optimizer contains exactly the residual trainable parameters;
8. no test data are exposed.

## Execution

All six validation-development runs are authorized by this frozen protocol after their per-seed static audits pass:

`AF2IGEM0/1 x seeds 42,123,2026`.

Runs may resume from their own `last.pt`; resume does not change the requested 20-epoch budget. A completion marker must bind arm, seed, requested epochs, and parent checkpoint SHA.

The notebook follows the current Kaggle pattern: index `/kaggle/input` once, restore only required AF2FS result/checkpoint members from prior state, prepare the leakage-safe development core, run audits, run/resume paired arms, snapshot state, aggregate validation results, and keep test locked.

## Frozen three-seed decision

Let `d_m(s)` be candidate minus matched zero-control for metric `m` at seed `s`, and `p_m(s)` be candidate minus its frozen AF2FS parent.

### Parent safety gate

Using the three-seed mean candidate-minus-parent deltas:

- Macro >= -0.20 pp;
- Bottom-3 >= -1.00 pp;
- Worst >= -1.00 pp.

All 21 validation classes must be present for every parent/control/candidate report and test must remain unopened.

### Route A — aggregate superiority over matched control

All must hold:

- mean Macro `AF2IGEM1 - AF2IGEM0` >= +0.20 pp;
- Macro improves in at least 2/3 seeds;
- mean Bottom-3 delta >= -0.50 pp;
- mean Worst delta >= -1.00 pp.

### Route B — lower-tail Pareto improvement over matched control

All must hold:

- mean Macro delta >= -0.10 pp;
- mean Bottom-3 delta >= +0.50 pp and improves in at least 2/3 seeds;
- mean Worst delta >= +1.00 pp and improves in at least 2/3 seeds.

`RETAIN` requires the parent safety gate plus Route A or Route B. Otherwise the decision is `REJECT`.

These thresholds preserve the numerical tolerances of the older parent-residual screen while adding three-seed directional consistency. They are frozen here before the new AF2FS paired runs.

## Claim boundary

A `RETAIN` supports only:

> Under matched three-seed development-validation training with completed AF2FS parents frozen, real P3/P4/P5 conditioning gives the IGEM residual useful classification information beyond its zero-information capacity/optimization control.

It does **not** establish independent test generalization, does not prove that arbitrary module stacking is beneficial, and does not authorize adding SAF/STB in the same training run. SAF and STB-vs-CMC remain separate later hypotheses.

A `REJECT` closes this AF2FS+IGEM frozen-parent route under the frozen formulation; it does not invalidate standalone AF2 or standalone IGEM evidence.
