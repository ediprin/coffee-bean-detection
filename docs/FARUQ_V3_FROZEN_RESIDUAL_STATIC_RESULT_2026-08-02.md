# Faruq-v3 Frozen-D0 Residual Static Audit

Date: 2026-08-02

Protocol: `faruq-v3-frozen-residual-static-v1`

Decision: **PASS — authorize FRM1 seed-42 validation screening only**

No dataset was accessed, no training was run, and test remained unavailable.

## Bound D0 artifact

- checkpoint: `experiments/faruq-v3-yolo26n-baseline-v1/D0_seed42/weights/best.pt`;
- SHA-256: `0c458841b84bedce4e0ddada6a5773f6a5ac8a91dad084a4a5f24e89f04e6367`;
- native Detect state items explicitly remapped: **240**.

Ultralytics' generic loader transferred only 468 of 722 compatible items after
the Detect head was wrapped. FRM1 therefore performs an additional strict
native-head remap. The audit compares the resulting candidate directly with
the bound D0 checkpoint.

## Static evidence

| Check | Result |
|---|---:|
| Total parameters | 3,180,943 |
| Trainable refiner/gate parameters | 668,953 (21.03%) |
| Initial gate mean | 0.010000 |
| Zero-initialized output maximum absolute difference vs D0 | **0.0** |
| Native D0 head state bitwise preserved | PASS |
| Raw native predictions bitwise equal | PASS |
| Only refiner and gate trainable | PASS |
| Native BatchNorm remains in eval mode | PASS |
| Residual classifier exactly zero initialized | PASS |
| Refiner/gate gradients finite | PASS |
| Active residual changes class output | PASS |
| Active residual preserves native raw boxes | PASS |
| State-dict round trip | PASS |

All ten mandatory gates passed. This establishes only implementation safety
and exact D0 initialization. It does not predict whether FRM1 will improve
validation performance.

Raw artifact:
`experiments/faruq-v3-frozen-residual-v1/static_audit.json`
