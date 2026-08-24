# Faruq-v3 AF2 Direct-from-Pretrained Seed-42 Screen Result

Date: 2026-08-24  
Branch: `codex/af2-direct-from-pretrained`  
Status: **SEED-42 SCREEN PASSED**  
Decision: **PROMOTE_TO_3_SEED**  
Evaluation: Faruq-v3 grouped development validation only  
Locked test: **not opened**

## Protocol

This result records the completed seed-42 screen defined prospectively in `docs/FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_PROTOCOL_2026-08-24.md`.

Both `D0DIRECT` and `AF2DIRECT` started from the same exact official `yolo26n.pt` source and the same matched 21-class target-head initialization. The only intended treatment difference was the retained AF2 input frontend being active from the first optimization step in `AF2DIRECT`.

Frozen official pretrained artifact SHA-256:

`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

Common initialized detector-state SHA recorded by the run:

`f21cfa9fd1e23624494ad57a48bb3fdd878f46a742f57f0d490bee3ac0d08e1a`

## Training completion

The final run reported:

- `D0DIRECT`: 50 / 50 epochs recorded;
- `AF2DIRECT`: 50 / 50 epochs recorded;
- test access: `False`.

A final state snapshot was written as:

`/kaggle/working/af2-direct-from-pretrained-seed42-state.zip`

with reported final size `19,489,440` bytes. An earlier intermediate snapshot after the control arm was `47,178,545` bytes.

## D0DIRECT control result

Primary validation metrics:

| Metric | D0DIRECT |
|---|---:|
| Macro mAP50-95 | 79.90236487299918% |
| Bottom-3 mAP50-95 | 64.56923842347432% |
| Worst-class mAP50-95 | 61.76885884935709% |

Mechanism diagnostic:

| Diagnostic | D0DIRECT |
|---|---:|
| Raw top-500 proposal accessibility | 99.80988593155894% |
| Localization-conditioned Top-1 | 57.02917771883289% |
| Correct-decision recall | 40.87452471482890% |

## Frozen seed-42 screen

The notebook reported the following final screen object:

```json
{
  "localization_safe": true,
  "route_a_direct_overall_gain": true,
  "route_b_lower_tail_pareto": true,
  "decision": "PROMOTE_TO_3_SEED"
}
```

Therefore, under the thresholds frozen before training:

- localization safety: **PASS**;
- Route A — direct overall gain: **PASS**;
- Route B — lower-tail Pareto signal: **PASS**;
- final seed-42 promotion decision: **PROMOTE_TO_3_SEED**.

The protocol defines `PROMOTE_TO_3_SEED = Route A OR Route B`; this run passed both routes.

## Numeric-capture boundary

The pasted completion excerpt supplied for this repository record contains the exact `D0DIRECT` metrics and the final Boolean `SCREEN`, but it does **not** contain the exact `AF2DIRECT` primary metrics, AF2DIRECT mechanism-diagnostic values, or paired numerical deltas.

Those missing values are therefore **not reconstructed or inferred** in this document. The Boolean screen result is recorded exactly as emitted by the completed notebook. Exact AF2DIRECT numbers should be imported from the saved Kaggle state/output artifact before this record is used as a full numerical result table.

Because Route A and Route B were both reported true, the completed notebook asserts that the AF2DIRECT result satisfied all prospectively frozen conditions for both routes. This is sufficient for the protocol-level promotion decision, but not a substitute for preserving the exact candidate metrics.

## Interpretation and claim boundary

This seed-42 result supports only the next prospective step: run the exact direct-from-pretrained protocol on seeds 123 and 2026 for three-seed confirmation.

It does **not** establish final AF2DIRECT superiority, untouched-test generalization, universal localization preservation, external-domain robustness, or deployment readiness. The locked test remains closed.

No threshold is revised after observing this result.

Repository evidence snapshot:
`docs/evidence/FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_SEED42_SCREEN_2026-08-24.json`.
