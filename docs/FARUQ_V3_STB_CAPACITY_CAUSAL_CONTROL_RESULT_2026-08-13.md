# Faruq-v3 STB Capacity-Causal Control Result

Date: 2026-08-13  
Decision: **PASS — authorize paired seed 123/2026 confirmation**  
Evaluation: Faruq-v3 grouped validation, seed 42  
Test accessed: **no**

## Purpose

This experiment tests whether the seed-42 advantage of `STB1` comes from its
shifted-window spatial token interaction or merely from extra head capacity.
`CMC0` is a non-spatial channel-mixing control with the same initialization,
P3/P4/P5 placement, two-block depth, identity gate, and 50-epoch schedule.

## Static control

| Model | Parameters |
|---|---:|
| STB1 | 4,589,201 |
| CMC0 | 4,588,025 |
| Difference | 1,176 (0.0256%) |

Both candidates preserve D0 boxes and scores bitwise at a zero gate, change
classification scores without changing boxes when active, and use the same
three pyramid levels. CMC0 contains no spatial attention or spatial
convolution. The static gate passed before training.

## Validation result

| Model | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---|---:|---:|---:|
| D0FT | 86.69% | 74.98% | 72.02% |
| CMC0 | 87.10% | 81.87% | **81.31%** |
| STB1 | **88.67%** | **83.64%** | 80.81% |

### CMC0 minus D0FT

- Macro: +0.41 point
- Bottom-3: +6.89 points
- Worst-class: +9.29 points
- Control-validity decision: **PASS**

### STB1 minus CMC0

- Macro: **+1.57 points**
- Bottom-3: **+1.77 points**
- Worst-class: **-0.49 point**
- Causal-screening decision: **PASS**

The frozen gate required at least +0.5 Macro point, no Bottom-3 loss, and no
more than one Worst-class point loss. STB1 passed all three criteria.

## Interpretation and boundary

At seed 42, STB1's advantage cannot be explained only by parameter count,
head depth, pyramid placement, initialization, or training schedule. The
remaining controlled distinction is spatial token interaction. The result is
therefore evidence for an STB spatial-mixing contribution on Faruq-v3.

This seed-42 screening result subsequently failed multi-seed confirmation.
Across seeds 42/123/2026, STB1 exceeded CMC0 by only +0.07 Macro point on
average, below the frozen +0.50-point threshold. Therefore this section must
not be cited without the completed follow-up:
`FARUQ_V3_STB_CAPACITY_PAIRED_CONFIRMATION_RESULT_2026-08-14.md`.
Test access was not authorized.

## Resume provenance

The original interrupted run produced overlapping resume rows. The runner
quarantined the corrupt resume artifacts, retained a checkpoint-consistent
monotonic prefix, and completed the unchanged 50-epoch protocol under a
Drive-visible exclusive lock. No architecture, seed, schedule, metric, or
acceptance threshold changed after observing validation.

## Authoritative artifacts

- `experiments/faruq-v3-stb-capacity-control-v1/static_audit.json`
- `experiments/faruq-v3-stb-capacity-control-v1/val_reports/CMC0_seed42_val.json`
- `experiments/faruq-v3-stb-capacity-control-v1/val_reports/stb_capacity_control_seed42_decision.json`
- `experiments/faruq-v3-stb-capacity-control-v1/CMC0_seed42/weights/best.pt`
