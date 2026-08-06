# Faruq-v3 ACMC2 Paired Three-Seed Confirmation Result

Date: 2026-08-07

Protocol: `faruq-v3-acmc2-paired-optimization-confirmation-v1`

Architecture under test: ACMC2 (`ambiguity_mode=entropy_margin`), i.e. ACMC1 entropy-conditioned multilevel classification refinement augmented with a learned top1-top2 margin-conditioned gate delta. Boxes remain native YOLO26n outputs; the correction path changes classification scores only.

Decision: **FAIL — keep ACMC1 as the selected model**

The paired confirmation used seeds 42, 123, and 2026. Seed 42 reused the locked ACMC2 screening result. Seeds 123 and 2026 trained ACMC2 from the exact D0 checkpoints already used by the prior paired D0FT/ACMC1 confirmation. Evaluation was validation-only. The locked test split was not opened or accessed.

## Aggregate result

| Metric | D0 mean | D0FT mean | ACMC1 mean | ACMC2 mean | ACMC2 - D0FT mean | ACMC2 - ACMC1 mean |
|---|---:|---:|---:|---:|---:|---:|
| Macro mAP50-95 | 80.12% | 86.62% | **87.62%** | 87.56% | **+0.94 pp** | **-0.06 pp** |
| Bottom-3 class mAP50-95 | 66.58% | 76.58% | 79.13% | **79.37%** | **+2.80 pp** | **+0.24 pp** |
| Worst-class mAP50-95 | 60.18% | 73.05% | 76.30% | **77.23%** | **+4.17 pp** | **+0.93 pp** |

Against D0FT, ACMC2 improved Macro on all three seeds, Bottom-3 on two of three seeds, and Worst-class on two of three seeds. Against ACMC1, ACMC2 improved Macro on two of three seeds, Bottom-3 on two of three seeds, and Worst-class on two of three seeds.

## Per-seed result

| Seed | Model | Macro mAP50-95 | Bottom-3 mAP50-95 | Worst-class mAP50-95 |
|---:|---|---:|---:|---:|
| 42 | D0 | 79.97% | 68.72% | 65.09% |
| 42 | D0FT | 86.69% | 74.98% | 72.02% |
| 42 | ACMC1 | 87.62% | 80.40% | **79.49%** |
| 42 | ACMC2 | **87.81%** | **81.94%** | 79.33% |
| 123 | D0 | 79.50% | 61.18% | 47.79% |
| 123 | D0FT | 86.35% | 74.75% | 67.83% |
| 123 | ACMC1 | **87.88%** | 77.30% | 73.55% |
| 123 | ACMC2 | 87.32% | **77.37%** | **76.02%** |
| 2026 | D0 | 80.89% | 69.85% | 67.65% |
| 2026 | D0FT | 86.81% | **79.99%** | **79.31%** |
| 2026 | ACMC1 | 87.34% | 79.68% | 75.86% |
| 2026 | ACMC2 | **87.55%** | 78.81% | 76.33% |

### ACMC2 minus ACMC1

| Seed | Macro | Bottom-3 | Worst-class |
|---:|---:|---:|---:|
| 42 | +0.19 pp | +1.54 pp | -0.16 pp |
| 123 | -0.57 pp | +0.06 pp | +2.47 pp |
| 2026 | +0.21 pp | -0.87 pp | +0.47 pp |
| Mean | **-0.06 pp** | **+0.24 pp** | **+0.93 pp** |

## Frozen criteria

The confirmation criteria were fixed before seeds 123 and 2026 were trained.

| Criterion | Result |
|---|---|
| Macro gain over D0FT mean at least 0.5 pp | PASS |
| Macro improves over D0FT on at least 2/3 seeds | PASS |
| Bottom-3 mean not lower than D0FT | PASS |
| Bottom-3 improves over D0FT on at least 2/3 seeds | PASS |
| Worst-class mean drop versus D0FT no more than 1 pp | PASS |
| **Macro mean not lower than ACMC1** | **FAIL** |
| Macro improves over ACMC1 on at least 2/3 seeds | PASS |
| At least one tail mean improves over ACMC1 | PASS |
| At least one tail metric improves over ACMC1 on at least 2/3 seeds | PASS |
| Neither tail mean drops more than 1 pp versus ACMC1 | PASS |

All criteria were conjunctive. Because `macro_mean_not_lower_than_acmc1` failed, the predeclared decision is **FAIL** and the selected model remains ACMC1.

## Scientific interpretation

ACMC2 is not a failed refinement in the sense of being ineffective. Relative to the optimization-matched native continuation D0FT, it still produced clear three-seed gains: +0.94 pp Macro, +2.80 pp Bottom-3, and +4.17 pp Worst-class mAP50-95. Therefore ambiguity-conditioned multilevel classification refinement remains supported.

The narrower question for ACMC2 was whether adding top1-top2 margin uncertainty improves the entropy-only ACMC1 gate. The answer is **not consistently on the primary aggregate objective**. Margin conditioning shifted the trade-off toward the tail: Bottom-3 improved by +0.24 pp mean and Worst-class by +0.93 pp mean, while Macro decreased by 0.06 pp mean versus ACMC1.

This supports the following bounded interpretation:

- ACMC1 entropy-only gating provides the best overall validation trade-off and remains the selected architecture.
- ACMC2 entropy+margin gating appears more aggressive toward hard/tail classes.
- The margin signal is useful but not sufficiently stable to replace ACMC1 under the predeclared selection rule.
- The ACMC2 result should be reported as a valid ablation/negative selection result, not hidden or redefined as a win after observing the data.

## Selection boundary

Do **not** open the locked test split for ACMC2. Do not retune ACMC2 on these three validation seeds solely to recover the 0.06 pp Macro deficit; that would convert the paired confirmation into validation-driven model selection after the fact.

Selected model after this experiment: **ACMC1**.

Next action: preserve ACMC2 as an ablation result and continue thesis/report consolidation around ACMC1. The test split remains locked until a separately authorized final evaluation protocol is used.

## Raw artifacts

Google Drive experiment root:

`experiments/faruq-v3-acmc2-paired-confirmation-v1/`

Final summary:

`experiments/faruq-v3-acmc2-paired-confirmation-v1/val_reports/acmc2_paired_optimization_confirmation.json`

Per-seed validation reports include:

- `experiments/faruq-v3-acmc2-entropy-margin-v1/val_reports/acmc2_seed42_screening.json`
- `experiments/faruq-v3-acmc2-paired-confirmation-v1/val_reports/ACMC2_seed123_val.json`
- `experiments/faruq-v3-acmc2-paired-confirmation-v1/val_reports/ACMC2_seed2026_val.json`

Reference ACMC1 paired confirmation:

`experiments/faruq-v3-acmc-paired-confirmation-v1/val_reports/acmc1_paired_optimization_confirmation.json`

A machine-readable snapshot of the final ACMC2 confirmation is committed at:

`docs/evidence/FARUQ_V3_ACMC2_PAIRED_RESULT_2026-08-07.json`
