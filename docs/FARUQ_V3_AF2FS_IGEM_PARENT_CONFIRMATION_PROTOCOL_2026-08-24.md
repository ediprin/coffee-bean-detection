# Faruq-v3 AF2FS + IGEM Frozen-Parent Paired Confirmation

Date frozen: 2026-08-24  
Branch: `codex/af2-igem-parent-confirmation`  
Status: **FROZEN BEFORE TRAINING**  
Evaluation: Faruq-v3 grouped development validation only  
Locked test: **closed**  
Current pre-training audit revision: `2026-08-24d`

## Research question

The prior `AF2+IGEM1` direct combination was trained jointly from D0 and therefore did not answer whether IGEM provides information complementary to an already trained AF2 while the AF2 representation is preserved.

This confirmation asks:

> With each completed AF2FS parent frozen, does a classification-only IGEM residual receiving real frozen AF2 P3/P4/P5 features outperform an architecture-, capacity-, schedule-, initialization-, and optimization-matched zero-information residual across seeds 42/123/2026?

No result from this new confirmation has been observed yet. The implementation corrections documented below happened before the first IGEM training epoch.

## Canonical frozen parents

Each arm is bound to the completed `AF2FS` parent of the same seed from the completed AF2FFAB2 from-start state:

- seed 42 -> AF2FS seed 42 `best.pt`;
- seed 123 -> AF2FS seed 123 `best.pt`;
- seed 2026 -> AF2FS seed 2026 `best.pt`.

The checkpoint must match the SHA-256 in `AF2FS_seed{seed}_result.json`. The runner also requires that canonical result file and copies its already-recorded validation metrics as the parent baseline. The parent is therefore **not re-evaluated once per arm**, avoiding redundant GPU validation and avoiding a second numerical realization of the same baseline.

For every arm the AF2 input enhancer, YOLO backbone, neck, native classification heads, native localization heads, and parent BatchNorm statistics are frozen. Only `model.23.residual.*` is trainable.

## Matched arms

| Arm | Residual input | Purpose |
|---|---|---|
| `AF2IGEM0` | zeros shaped like frozen P3/P4/P5 | zero-information capacity/optimization control |
| `AF2IGEM1` | real frozen P3/P4/P5 | feature-conditioned candidate |

Both use the same architecture and settings. The only intended config difference is `conditioning: zero` versus `conditioning: feature`.

Frozen IGEM settings: reference depth 3, mask-loss weight 0.05, kernel size 3, attention heads 4, channel reduction 4, correction scale 1.0. Training: maximum 20 epochs, imgsz 640, batch 16, workers 2, patience 7, optimizer `auto`, cache false, close-mosaic 10, max-det 500. Early stopping is allowed; the requested maximum remains 20 epochs.

The coarse bbox-derived IGEM mask objective is present in both arms. Thus the candidate-control comparison isolates information entering the residual rather than merely the presence of the auxiliary objective.

## Forward contract

For each level `l`, native AF2 classification logits remain intact and the residual adds only a classification correction:

\[
z_l = z_l^{AF2} + \Delta z_l^{IGEM}.
\]

For the control and candidate:

\[
\Delta z_l^{IGEM0}=R_l(0), \qquad
\Delta z_l^{IGEM1}=R_l(F_l).
\]

The native box branch consumes the original frozen AF2 features and receives no IGEM correction.

## Dedicated IGEM static authorization — revision 2026-08-24d

Training is forbidden unless a dedicated IGEM-only static audit passes for the exact seed-matched parent SHA. The legacy shared SAF/IGEM audit is deliberately left unchanged and is no longer used to authorize this experiment.

The dedicated audit must establish all of the following before training:

1. control and candidate have the same model YAML, AF2 config, training schedule, parameter count, trainable count, and state schema;
2. their residual modules have **identical initialized state** when constructed from the same fixed audit seed;
3. only conditioning differs in the residual config;
4. transfer of the frozen parent backbone/neck and native Detect head is **bitwise exact in state**;
5. both arms reproduce the parent output numerically at zero-output initialization;
6. repeated-forward box outputs remain within the pre-existing `ATOL=5e-5`, `RTOL=1e-5` numerical envelope;
7. after activating the final IGEM correction projection, the zero-information control must produce an **exactly zero residual correction tensor before native-logit addition**;
8. under the same artificial activation, the feature-conditioned candidate must produce a **finite, nonzero residual correction tensor before native-logit addition**;
9. only residual parameters are trainable;
10. parent BatchNorm modules remain in `eval()` while residual BatchNorm modules are trainable;
11. residual gradients are finite/live, parent parameters receive no gradients, and a residual optimizer step leaves all parent parameters/buffers unchanged;
12. test access remains false.

Whole-model score displacement after the artificial projection activation is retained as a diagnostic only. It is not an authorization threshold because its observed magnitude depends on native-logit scale, floating-point ULP size, and repeat-forward numerical realization. Parent and box preservation continue to use the predeclared `ATOL/RTOL` tolerance; residual-path activity is instead tested directly at the residual correction tensor where zero-information versus feature-conditioned routing is the actual invariant of interest.

### Why revisions a, b, c, and d exist

The first seed-42 CUDA audit stopped before training because the earlier audit required bitwise equality between separate CUDA forwards. The observed control score jitter was `1.52587890625e-05`, while the feature-conditioned score change was `2.23699951171875`. The audit already used `ATOL=5e-5`, `RTOL=1e-5` for parent identity before these values were observed.

Revision `2026-08-24a` first applied that existing numerical tolerance consistently. A later repo audit found that changing the **shared** SAF/IGEM legacy audit could disturb unrelated regression contracts, so revision `2026-08-24b` isolated this study in a dedicated IGEM audit and restored the shared audit.

CPU CI on revision b then exposed a different issue: `torch.allclose` is appropriate for identity/preservation, but its relative term scales with large-magnitude logits and can classify a deterministic residual correction as numerically close. Revision `2026-08-24c` therefore kept `allclose(ATOL=5e-5, RTOL=1e-5)` for parent/box preservation and temporarily defined residual activity using absolute whole-score displacement.

The revision-c CI fixture then exposed that whole-score displacement is still not an invariant activity test: on a serialized synthetic AF2 parent, the feature-conditioned route produced a nonzero displacement of `2.956390380859375e-05`, below `ATOL`, while zero-control displacement remained exactly zero. That fixture result is not model-performance evidence; it only demonstrates scale dependence of the audit criterion. Revision `2026-08-24d` therefore moves the activity test to the residual correction tensor itself, before addition to native AF2 logits. Zero conditioning must give exactly zero correction; feature conditioning must give a finite nonzero correction. This removes the arbitrary coupling between pathway liveness and the preservation tolerance without relaxing any performance, parent-safety, training, dataset, or test-access gate.

No IGEM training epoch had run before revisions a, b, c, or d. Model architecture, dataset, optimizer, arm definitions, parent checkpoints, training schedule, decision thresholds, and test lock are unchanged.

## Training and resume contract

All six development-validation runs are authorized only after all three seed-specific static audits pass:

`AF2IGEM0/1 x seeds 42,123,2026`.

Every run writes `run_contract.json` binding:

- arm and conditioning;
- seed;
- canonical AF2FS checkpoint SHA;
- canonical AF2FS result-file SHA;
- config SHA;
- static-audit revision and parent SHA;
- requested epochs;
- trainable scope;
- validation-only/test-locked status.

A `last.pt` is resumable only inside a directory with the matching contract. Stale weights without a contract are rejected. Ultralytics 8.4.96 is pinned. Its trainer may temporarily re-enable frozen floating-point parameters during generic setup, but the custom trainer re-applies the residual-only freeze **before optimizer construction**, filters optimizer groups to the residual set exactly, and the custom model keeps parent BatchNorm modules in evaluation mode during training.

### Exact EMA parent preservation

Ultralytics `ModelEMA` normally averages every floating tensor in the model state, including tensors whose optimizer gradients are frozen. That behavior is acceptable for ordinary training but is too weak for a parent-preserving experiment because the serialized `best.pt` is derived from EMA rather than directly from the live model.

Therefore the IGEM confirmation trainer performs ordinary EMA updates for `model.23.residual.*`, then immediately copies every **non-residual** parameter and buffer from the live frozen model into the EMA model exactly. This includes backbone, neck, native Detect head, and frozen BatchNorm state. The runner then reloads `best.pt` and requires the serialized parent portion to be bitwise identical to the canonical AF2FS parent before any validation result is accepted. If this exact check fails, the arm fails closed and cannot enter the three-seed decision.

This EMA hardening was added before the first IGEM training epoch and does not change the residual optimizer, loss, dataset, schedule, decision thresholds, or test lock.

## Frozen three-seed decision

Let `d_m(s)` be candidate minus zero-control and `p_m(s)` candidate minus canonical AF2FS parent.

Parent safety, all required:

- mean Macro `p >= -0.20 pp`;
- mean Bottom-3 `p >= -1.00 pp`;
- mean Worst `p >= -1.00 pp`.

Route A — aggregate superiority over matched control, all required:

- mean Macro `d >= +0.20 pp`;
- Macro improves in at least 2/3 seeds;
- mean Bottom-3 `d >= -0.50 pp`;
- mean Worst `d >= -1.00 pp`.

Route B — lower-tail Pareto improvement over matched control, all required:

- mean Macro `d >= -0.10 pp`;
- mean Bottom-3 `d >= +0.50 pp` and improves in at least 2/3 seeds;
- mean Worst `d >= +1.00 pp` and improves in at least 2/3 seeds.

`RETAIN` requires parent safety plus Route A or Route B. Otherwise `REJECT`.

All 21 validation classes must be present. The locked test remains unopened regardless of decision.

## Claim boundary

A `RETAIN` supports only that, under matched three-seed development-validation training with canonical AF2FS parents frozen, real P3/P4/P5 conditioning gives the IGEM residual useful classification information beyond its zero-information matched control. It does not establish independent test generalization, does not prove generic module stacking, and does not authorize SAF or STB in the same run.

A `REJECT` closes only this AF2FS+IGEM frozen-parent formulation. It does not invalidate standalone AF2 or standalone IGEM evidence.
