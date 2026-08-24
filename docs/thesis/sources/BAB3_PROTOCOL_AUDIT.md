# Bab III Protocol Audit

Purpose: prevent methodology drift between old experiment configs, the frozen direct-AF2 protocol, implementation code, pilot result records, and final proposal prose.

Audit target: `docs/thesis/proposal/05_METHODOLOGY.md`.

Status: **FIRST SOURCE-GROUNDED DRAFT**.

## Authority hierarchy

For the direct-AF2 thesis design, use this order:

1. `docs/FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_PROTOCOL_2026-08-24.md` — frozen experimental contract;
2. `src/coffee_detector/afab/operator.py` — actual AF2 mathematical/implementation behavior;
3. `configs/afab/AF2_yolo26n_chaotic_amplitude.yaml` — frozen AF2 parameter mapping and training config;
4. immutable Faruq-v3 grouped dataset contract recorded by the protocol;
5. `docs/FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_SEED42_SCREEN_RESULT_2026-08-24.md` — pilot promotion result;
6. `docs/thesis/foundation/04_PILOT_EVIDENCE.md` — proposal-oriented user-verified numeric capture, not the primary machine artifact.

Older experimental configs/protocols may be used as genealogy only and must not silently override the direct protocol.

## Known config conflict

`configs/D0_yolo26n.yaml` currently contains an older baseline schedule:

```text
epochs   = 100
workers  = 4
patience = 20
```

The frozen direct-from-pretrained protocol instead defines:

```text
epochs   = 50
workers  = 2
patience = 15
```

For `D0DIRECT` vs `AF2DIRECT`, the **50/2/15 direct protocol is authoritative**. Bab III must never copy the old D0 config values merely because its filename contains `D0`.

## AF2 implementation audit

From `operator.py` + frozen config:

| Item | Status | Proposal treatment |
|---|---|---|
| mode | `af2` | active |
| patch_size | 32 | active |
| overlap | 0.50 | active; stride 16 |
| gamma | 0.10 | active |
| angular_bins | 360 | active |
| chunk_size | 128 | implementation/memory parameter |
| eps | 1e-8 | numerical stability |
| RGB channels | independent | explicit transfer choice |
| angular discretization | floor-to-bin | explicit transfer choice |
| overlap reconstruction | fold + averaging | explicit transfer choice |
| output | `raw + raw * minmax(recovered)` | active residual gate |
| learned parameters | none | parameter-free |
| radius_ratio | 0.05 in shared config | **inactive in `mode=af2`**; used only by AF1/AF12 radial mask |

This last distinction is mandatory. The thesis title says frequency-angular, but the active AF2 operator does **not** use the AF1 radial high-pass mask.

## Dataset audit facts

Frozen direct protocol records:

```text
train      = 1,665 images / 2,986 annotations
validation =   294 images /   526 annotations
classes    = 21 present in both splits
```

The grouped contract has already closed parent/exact-hash cross-split leakage gates. Direct screening requires the development root to contain no `test` directory.

Proposal wording should say the test remains locked/unused during screening. Do not invent a final test result or imply the held-out split has already been evaluated.

## Matched initialization audit

Direct protocol requires:

- same exact official `yolo26n.pt` source;
- pretrained SHA-256:
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`;
- same isolated RNG construction for the 21-class target detector;
- complete persistent detector state equality before training;
- same parameter count;
- AF2 zero learned parameters.

This is stronger than merely saying “both use pretrained weights.” Bab III should retain this fairness logic.

## Pilot evidence provenance audit

There are currently two repository evidence layers:

### Machine-capture result record

`docs/FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_SEED42_SCREEN_RESULT_2026-08-24.md`

Records exactly:

- D0DIRECT numeric metrics;
- 50/50 epochs for both arms;
- test access `False`;
- PASS for localization safety;
- PASS Route A;
- PASS Route B;
- `PROMOTE_TO_3_SEED`.

It explicitly states that the supplied machine excerpt did **not** preserve exact AF2DIRECT metric values.

### Proposal foundation numeric capture

`docs/thesis/foundation/04_PILOT_EVIDENCE.md`

Contains user-verified AF2DIRECT numbers and paired deltas from the completed run.

Until the saved Kaggle state/output is imported into the machine evidence JSON/result record, final thesis numerical tables should distinguish these provenances. The proposal may use the user-verified pilot numbers with a note that they are preliminary, but the repository should not pretend they were already artifact-normalized.

## Evaluation audit

Study-defined metrics:

```text
Macro    = mean per-class AP50-95
Bottom-3 = mean of three lowest per-class AP50-95 values
Worst    = minimum per-class AP50-95
```

These are internal summary metrics. They must not be described as standard COCO metrics.

Mechanism diagnostics are validation-only attribution aids:

- raw top-500 proposal accessibility;
- localization-conditioned Top-1;
- correct-decision recall.

Mechanism language must be correlational/diagnostic (`consistent with`) rather than causal.

## Temporal audit

At proposal time:

| Item | Status |
|---|---|
| Direct seed-42 screen | completed |
| D0DIRECT 50 epochs | completed |
| AF2DIRECT 50 epochs | completed |
| Promotion decision | passed |
| Seed 123 direct confirmation | planned / not completed in proposal evidence |
| Seed 2026 direct confirmation | planned / not completed in proposal evidence |
| Final direct-AF2 superiority claim | not established |
| Locked-test direct result | not established |
| Final direct efficiency confirmation | not established |

## Promotion gate for Bab III prose

Before marking methodology `PROPOSAL-READY`, verify:

- [x] direct protocol is the training authority;
- [x] AF2 equations match current `operator.py`;
- [x] `radius_ratio` is not presented as active AF2 behavior;
- [x] dataset split counts match the frozen protocol;
- [x] AF2 is outside YOLO26 architecture;
- [x] seed 42 is labelled preliminary;
- [x] seeds 123/2026 are labelled planned;
- [x] Bottom-3/Worst are identified as study-defined;
- [x] parameter-free is not equated with compute-free;
- [ ] final hardware table populated from actual confirmatory environment;
- [ ] final metric equations paired with definitive evaluation specification/source where needed;
- [ ] AF2DIRECT exact numeric artifact imported from saved state for final thesis result provenance;
- [ ] locked-test use, if any, frozen prospectively before opening.
