# Faruq-v3 Capacity-Matched Multilevel Head Static Result

Date: 2026-08-02

Protocol: `faruq-v3-multilevel-head-static-v1`, protocol document `v1.0.1`

## Decision

`PASS -> AUTHORIZE_MULTILEVEL_HEAD_TRAINING_PROTOCOL`

This result authorizes only the freezing and implementation of a one-seed
validation training protocol. It does not authorize test access or additional
seeds. The static audit accessed no dataset and performed no optimizer step.

## Audited variants

- `MHC0`: capacity-matched P5 control;
- `MHF1`: parameter-free P3+P4+P5 fusion using the same modules and tensor
  dimensions as MHC0.

Both wrap the native YOLO26 end-to-end Detect head. Native box and class
branches remain present and unchanged. The added refiner is candidate
conditioned and remains inside one serialized detector graph.

## Static evidence

| Item | Result |
|---|---:|
| D0 parameters | 2,511,990 |
| MHC0 parameters | 3,180,939 |
| MHF1 parameters | 3,180,939 |
| Added parameters | 26.63% |
| MHC0/MHF1 state schema | identical |
| Native head state after injection | unchanged |
| Raw zero-weight tensors vs D0 | bitwise equal |
| Final zero-weight max absolute difference vs D0 | 5.82e-11 |
| Final identity tolerance | `rtol=0, atol=1e-7` |
| Active MHC0/MHF1 max absolute output difference | 142.52 |
| Direct refiner loss | 2.8693 |
| Refiner gradients | present and finite |
| State-dict round trip | exact |

The CPU latency smoke check measured 44.55 ms for MHC0 and 31.47 ms for MHF1
on the small synthetic audit input. Its ratio passed the wiring gate, but the
ordering is not an efficiency claim; final latency must be measured repeatedly
on the same T4 after a trained candidate exists.

## Initial fail and correction

The first audit under protocol `v1.0.0` returned FAIL because it required
bitwise equality after decoding. Diagnosis showed that native raw box, class,
and feature tensors were bitwise identical and the only discrepancy was CPU
floating-point round-off of `5.82e-11`. Before dataset access or training, the
protocol was amended to `v1.0.1`: final decoded output uses the documented
absolute tolerance `1e-7`, while raw native tensors retain the bitwise gate.
The rerun passed every gate.

## Boundaries

This is architecture and serialization evidence, not evidence of improved
detection. The next experiment must compare MHC0 and MHF1 under one frozen seed,
schedule, candidate assignment rule, auxiliary loss, and validation gate. Faruq
test remains locked.
