# Faruq-v3 AF2SFS1 Root-Cause Diagnostic Result — 2026-08-28

## Status

**INTERPRETABLE, but the active inference selector is not supported at the
frozen confidence-0.25/IoU-0.50 operating point.** No training or test access
occurred.

The completed AP endpoints remained:

| Model | Macro | Bottom-3 | Worst |
|---|---:|---:|---:|
| AF2CTRL | 89.00% | 83.88% | **83.55%** |
| AF2SFS1 | **89.96%** | **84.19%** | 83.25% |

## Paired target decomposition

The audit matched the same 526 validation targets:

| Metric | AF2CTRL | AF2SFS1 | Delta |
|---|---:|---:|---:|
| Raw top-500 accessibility | 99.81% | 99.81% | +0.00 point |
| Mean raw maximum IoU | 0.950 | 0.948 | -0.002 |
| Final matched recall | 91.44% | 92.78% | +1.33 points |
| Mean final matched IoU | 0.943 | 0.939 | -0.004 |
| Conditional top-1 accuracy | 72.77% | 71.52% | -1.25 points |
| Correct-decision recall | 66.54% | 66.35% | -0.19 point |

Raw localization accessibility did not improve. More targets reached a final
match, but lower conditional classification accuracy cancelled that gain at
this single operating point.

## Inference intervention

AF2SFS1 normal and adapter-bypass both produced 66.35% correct-decision
recall. Normal versus bypass changed final matched recall by -0.19 point,
conditional accuracy by +0.15 point, and correct-decision recall by exactly
0.00 point. Spatial-only matched normal on these headline diagnostics, while
frequency-only matched bypass closely. Therefore the audit's explicit
`active_selector_inference_effect_supported` flag was `false`.

This does not negate the completed +0.95-point Macro AP result. AP integrates
confidence ranking and IoU thresholds, whereas this diagnostic used one fixed
confidence and matching threshold. Full native-mAP interventions are required
before deciding whether the gain is an active selector effect or an
optimization-mediated effect retained in the detector weights.

## Class redistribution

Largest completed per-class AP gains:

| Class | Delta |
|---|---:|
| biji_berkulit_tanduk | +5.10 points |
| biji_berlubang_satu | +4.54 points |
| biji_berlubang_lebih_satu | +4.24 points |
| kulit_kopi_ukuran_sedang | +2.96 points |
| biji_bertutul_tutul | +2.31 points |

Largest regressions:

| Class | Delta |
|---|---:|
| kulit_tanduk_ukuran_besar | -1.59 points |
| biji_hitam | -1.58 points |
| kulit_kopi_ukuran_besar | -1.56 points |
| biji_coklat | -1.14 points |
| kulit_tanduk_ukuran_sedang | -0.37 point |

The -0.29-point headline Worst change does not mean every class stayed within
0.29 point; the identity of the minimum changed and masks larger individual
regressions.

Protocol:
`docs/FARUQ_V3_AF2SFS1_ROOT_CAUSE_PROTOCOL_2026-08-28.md`.

Drive summary:
`experiments/faruq-v3-af2-complement-v1/root_cause/af2sfs1_root_cause.json`.
