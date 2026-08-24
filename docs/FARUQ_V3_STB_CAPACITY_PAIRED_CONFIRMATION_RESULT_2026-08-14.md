# Faruq-v3 STB versus CMC0 Paired Confirmation Result

Date: 2026-08-14  
Protocol: `faruq-v3-stb-capacity-paired-confirmation-v1`  
Decision: **FAIL — stop the STB spatial-causal claim without test access**  
Evaluation: grouped Faruq-v3 validation, seeds 42/123/2026  
Test accessed: **no**

## Question

Does STB1's shifted-window spatial interaction improve the fine-grained
classification path beyond a parameter-, depth-, initialization-, schedule-,
and placement-near-matched non-spatial channel mixer (`CMC0`)?

The parameter difference remains 1,176 parameters, or 0.0256% of the roughly
4.59M-parameter candidates.

## Per-seed validation results

| Seed | Model | Macro mAP50-95 | Bottom-3 | Worst class |
|---:|---|---:|---:|---:|
| 42 | CMC0 | 87.10% | 81.87% | **81.31%** |
| 42 | STB1 | **88.67%** | **83.64%** | 80.81% |
| 123 | CMC0 | **89.19%** | **78.28%** | **74.21%** |
| 123 | STB1 | 87.81% | 76.59% | 73.90% |
| 2026 | CMC0 | 86.96% | 76.83% | 70.85% |
| 2026 | STB1 | **86.98%** | **81.26%** | **80.37%** |

### Paired STB1 minus CMC0 deltas

| Seed | Macro | Bottom-3 | Worst class |
|---:|---:|---:|---:|
| 42 | +1.57 | +1.77 | -0.49 |
| 123 | -1.38 | -1.70 | -0.31 |
| 2026 | +0.02 | +4.43 | +9.52 |

The seed-42 spatial advantage did not repeat at seed 123. Seed 2026 was
effectively tied on Macro but produced a large lower-tail increase.

## Three-seed aggregate

| Model | Macro mean ± std | Bottom-3 mean ± std | Worst mean ± std |
|---|---:|---:|---:|
| CMC0 | 87.75 ± 1.25% | 78.99 ± 2.59% | 75.45 ± 5.34% |
| STB1 | **87.82 ± 0.84%** | **80.50 ± 3.59%** | **78.36 ± 3.87%** |
| Paired delta | **+0.07 ± 1.47** | **+1.50 ± 3.07** | **+2.90 ± 5.73** points |

STB1 improved Macro in 2/3 seeds, Bottom-3 in 2/3 seeds, and Worst class in
1/3 seeds. The positive mean Worst delta is driven by seed 2026 rather than a
consistent three-seed direction.

## Frozen gate

| Criterion | Result |
|---|---|
| Mean Macro gain at least +0.5 point | **FAIL** (+0.07) |
| Macro improves in at least 2/3 seeds | PASS |
| Mean Bottom-3 not lower | PASS |
| Bottom-3 improves in at least 2/3 seeds | PASS |
| Mean Worst decline no greater than 1 point | PASS |

Final decision: **FAIL**. The protocol therefore does not validate shifted-
window spatial interaction as a robust causal improvement over non-spatial
classification-head capacity.

## Descriptive comparison with existing three-seed evidence

This table is contextual, not a new pre-frozen head-to-head gate.

| Model | Macro mean | Bottom-3 mean | Worst mean | Evidence status |
|---|---:|---:|---:|---|
| CMC0 | 87.75% | 78.99% | 75.45% | paired validation control |
| STB1 | **87.82%** | **80.50%** | **78.36%** | paired validation; causal gate FAIL |
| ACMC1 | 87.62% | 79.13% | 76.30% | paired validation PASS; locked test NOT_CONFIRMED |

STB1 has the highest descriptive validation means in this table, but this does
not override the failed causal gate and does not authorize test evaluation.
ACMC1 remains the only one of these architectural candidates already evaluated
under the locked-test protocol; its locked-test direction was positive but its
paired-parent bootstrap probability (0.928) missed the frozen 0.950 threshold.

## Scientific interpretation

1. Increasing classification-head representation/capacity is useful: CMC0 is
   itself strong and explains most of the seed-42 STB improvement.
2. STB's spatial interaction is not a stable Macro improvement beyond CMC0.
3. STB may help lower-tail classes, but the effect is seed-sensitive and cannot
   be promoted as a robust spatial-mechanism contribution.
4. STB remains a high-performing validation reference, not the confirmed core
   of a new fusion architecture.
5. STB-centered CPE/AF2 fusion is paused. The next already frozen action is the
   standalone AF2 and IGEM1 paired validation confirmation.

## Authoritative artifact

Drive report:

`experiments/faruq-v3-stb-paired-confirmation-v1/val_reports/stb_capacity_paired_confirmation.json`

Repository machine-readable snapshot:

`docs/evidence/FARUQ_V3_STB_CAPACITY_PAIRED_CONFIRMATION_2026-08-14.json`
