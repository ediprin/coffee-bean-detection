# Faruq-v3 AF2 Isolated Radial/Orientation Result

Date: 2026-08-21  
Protocol: `docs/AF2_ISOLATED_RADIAL_ORIENTATION_PROTOCOL_2026-08-21.md`  
Evaluation: grouped Faruq-v3 validation, seed 42 only  
Test accessed: **no**

## Result

Both isolated arms used the same D0 seed-42 checkpoint, YOLO26n-P3 detector,
50-epoch schedule, and original AF2 settings except for the single factor under
test.

| Model | Macro mAP50-95 | Bottom-3 | Worst class | Delta Macro | Delta Bottom-3 | Delta Worst |
|---|---:|---:|---:|---:|---:|---:|
| AF2 reference | 88.20% | 80.04% | 79.35% | - | - | - |
| `AF2_RADIAL` | 86.57% | 78.21% | 75.41% | -1.62 | -1.83 | -3.94 |
| `AF2_ORIENT` | **88.33%** | **81.38%** | **80.14%** | **+0.13** | **+1.34** | **+0.79** |

The deltas are percentage points relative to the frozen AF2 seed-42 result.
`AF2_RADIAL` median batch-1 640 latency was 23.03 ms; `AF2_ORIENT` was
23.88 ms on the recorded GPU runtime.

## Decision

- `AF2_RADIAL`: **REJECT**. Fixed low/mid/high radial partitioning reduced all
  three primary metrics.
- `AF2_ORIENT`: **RETAIN for paired seed confirmation**. Folding signed
  360-degree directions into unsigned 180-degree orientations improved Macro,
  Bottom-3, and Worst-class AP simultaneously.
- Radial-plus-orientation combination: **not authorized**. The protocol says
  to promote only the single useful isolated arm when the other arm fails.

The frozen isolation protocol did not define a +0.5-point Macro threshold. Its
decision rule required a defensible trade-off with non-degraded Macro and a
meaningful lower-tail improvement. Therefore no post-result +0.5 threshold is
applied here. The result is still one-seed screening evidence, not a
seed-stability claim.

## Authorized next step

Run `AF2_ORIENT` for seeds 123 and 2026 from the corresponding seed-matched D0
checkpoints and compare it pairwise with the already completed AF2 results.
Test remains locked. No combined arm or additional tuning is authorized before
that confirmation.

Machine-readable evidence:
`docs/evidence/FARUQ_V3_AF2_ISOLATED_RADIAL_ORIENTATION_2026-08-21.json`.
