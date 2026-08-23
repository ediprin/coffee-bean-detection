# Faruq-v3 AF2 Family Status Snapshot

Date: 2026-08-23

Purpose: freeze the latest AF2-family results in one place before any further analysis or design changes. This is a result ledger, not a new mechanistic interpretation.

Test status: do not open test for any arm unless an existing frozen protocol explicitly authorizes it.

## 1. Original AF2 three-seed reference

| Model | Macro mAP50-95 | Bottom-3 | Worst class | Status |
|---|---:|---:|---:|---|
| AF2 | 87.94% | 79.37% | 78.15% | RETAINED original AF2 reference |

Original AF2 remains the conservative retained spectral frontend reference.

## 2. AF2_ORIENT paired three-seed confirmation

AF2_ORIENT folds antipodal directions into a 180-degree orientation representation.

| Model | Macro mean | Bottom-3 mean | Worst mean |
|---|---:|---:|---:|
| AF2 | 87.94% | 79.37% | 78.15% |
| AF2_ORIENT | 88.26% | 79.25% | 76.66% |
| Delta, AF2_ORIENT - AF2 | +0.32 pp | -0.12 pp | -1.50 pp |

Decision: **FAIL -- retain original AF2.**

Per-seed AF2_ORIENT minus AF2 deltas:

| Seed | Macro | Bottom-3 | Worst |
|---:|---:|---:|---:|
| 42 | +0.13 pp | +1.34 pp | +0.79 pp |
| 123 | +1.00 pp | +0.84 pp | +1.48 pp |
| 2026 | -0.17 pp | -2.53 pp | -6.75 pp |

Seed-2026 class attribution recorded in the paired evidence includes:

- `biji_muda`: 84.72% -> 71.90%, delta -12.82 pp;
- `biji_berlubang_lebih_satu`: delta -7.28 pp;
- `tanah_batu_ranting_besar`: delta -5.11 pp;
- `biji_berkulit_tanduk`: delta -4.60 pp;
- `kopi_gelondong`: delta -3.06 pp;
- `biji_normal`: +6.86 pp;
- `biji_hitam_sebagian`: +6.06 pp;
- `biji_hitam_pecah`: +5.94 pp;
- `biji_bertutul_tutul`: +3.86 pp.

The class-specific cause of the seed-2026 `biji_muda` drop is **not yet diagnosed**. Do not record an unverified explanation such as a specific confusion target, localization failure, or gradient conflict.

Authoritative result already in repo:
`docs/FARUQ_V3_AF2_ORIENT_PAIRED_CONFIRMATION_RESULT_2026-08-21.md`.

## 3. AF2FFAB2 paired three-seed confirmation

Matched comparison: AF2FFAB2 versus AF2FFA0 continuation control.

| Metric | AF2FFA0 mean | AF2FFAB2 mean | Mean delta | Improved seeds |
|---|---:|---:|---:|---:|
| Macro | 87.10% | **88.54%** | **+1.44 pp** | 3/3 |
| Bottom-3 | 77.68% | **80.52%** | **+2.83 pp** | 3/3 |
| Worst | 74.69% | **76.96%** | **+2.27 pp** | 3/3 |

Decision: **PASS -- retain AF2FFAB2 as validated Pareto refinement against its matched control.**

Descriptive comparison to the pre-continuation original AF2 three-seed mean:

- Macro: +0.60 pp;
- Bottom-3: +1.15 pp;
- Worst: -1.19 pp.

Therefore AF2FFAB2 is not recorded as universally superior to original AF2. Original AF2 remains higher on the single Worst-class mean.

Authoritative result already in repo:
`docs/FARUQ_V3_AF2_FFA_B2_PAIRED_CONFIRMATION_RESULT_2026-08-22.md`.

## 4. Direct AF2 plus strong retained-model pairs

These were joint-from-D0 combinations and are negative composition results.

| Pair | Candidate Macro / B3 / Worst | Comparator Macro / B3 / Worst | Delta | Decision |
|---|---|---|---|---|
| AF2+IGEM1 vs IGEM1 | 87.4575 / 80.4218 / 80.1446 | 88.0080 / 82.1830 / 82.0776 | -0.5506 / -1.7612 / -1.9330 pp | REJECT |
| AF2+STB1 vs STB1 | 87.2204 / 78.8575 / 75.8284 | 88.6694 / 83.6394 / 80.8137 | -1.4491 / -4.7819 / -4.9853 pp | REJECT |
| AF2+SAF1 vs SAF1 | 88.0888 / 80.8476 / 79.6697 | 87.3373 / 81.3302 / 80.3359 | +0.7515 / -0.4826 / -0.6661 pp | REJECT |

Do not convert these negative pair results into an unsupported universal statement about incompatibility. They establish only the tested direct compositions under their actual training protocol.

## 5. AF2_ORIENT + CMC0 seed-42 screen

Canonical postprocessed validation result:

| Model | Macro | Bottom-3 | Worst |
|---|---:|---:|---:|
| AF2_ORIENT parent | 88.326% | 81.381% | 80.136% |
| AF2_ORIENT + CMC0 | 87.229% | 79.785% | 76.462% |
| Delta | -1.097 pp | -1.595 pp | -3.674 pp |

Raw screen decision: **FAIL**.

Frozen superiority route: FAIL.
Frozen tail-Pareto route: FAIL.

Execution note: training reached epoch 50 and the completed checkpoint was postprocessed, but `results.csv` contains duplicate epoch 35 at the resume boundary. The run is therefore not decision-clean under the strict execution-history audit. This anomaly does not turn the raw screen into a PASS. No multi-seed continuation is authorized from the observed screen result.

Status already recorded on branch `agent/af2-orient-cmc0-screening` in:
`docs/FARUQ_V3_AF2_ORIENT_CMC0_RESULT_2026-08-23.md`.

## 6. AF2 + CPE seed-42 matched continuation experiment

Observed final validation output supplied from the completed Colab decision cell on 2026-08-23:

| Arm | Macro | Bottom-3 | Worst |
|---|---:|---:|---:|
| AF2CPE0 | 84.97% | 76.12% | 75.75% |
| AF2CPE5 | 84.57% | 77.43% | 76.86% |
| Delta, CPE5 - CPE0 | -0.40 pp | +1.31 pp | +1.11 pp |

Exact deltas from the decision output:

- Macro: `-0.003967905868800248`;
- Bottom-3: `+0.013086345461147841`;
- Worst: `+0.011131678609525686`.

Frozen decision output:

- safety Macro not below -0.20 pp: **false**;
- safety Bottom-3 drop not over 1 pp: true;
- safety Worst drop not over 1 pp: true;
- superiority route: **false**;
- tail-Pareto route: **false**;
- final decision: **REJECT**;
- test: **do not open**.

Important evidence boundary: the result establishes only that AF2CPE5 failed the preregistered matched-control gate. Do not record an unverified mechanistic explanation for the failure. AF2CPE0 itself is substantially below the original AF2 seed-42 reference (88.20 / 80.04 / 79.35), so comparisons to original AF2 must be reported separately from the frozen CPE5-vs-CPE0 causal comparison.

## 7. Current AF2-family headline

| Candidate | Macro | Bottom-3 | Worst | Evidence level / decision |
|---|---:|---:|---:|---|
| Original AF2 | 87.94% | 79.37% | **78.15%** | three-seed retained reference |
| AF2_ORIENT | 88.26% | 79.25% | 76.66% | three-seed FAIL |
| AF2FFAB2 | **88.54%** | **80.52%** | 76.96% | three-seed PASS vs matched AF2FFA0 control; Pareto refinement |
| AF2_ORIENT+CMC0 | 87.229% | 79.785% | 76.462% | seed-42 raw FAIL; resume-history anomaly |
| AF2CPE5 | 84.57% | 77.43% | 76.86% | seed-42 REJECT vs AF2CPE0 |

The rows are not all interchangeable head-to-head experiments: AF2/AF2_ORIENT are paired from seed-matched D0; AF2FFAB2 is causally confirmed against AF2FFA0 continuation control; AF2_ORIENT+CMC0 is a seed-42 screen against AF2_ORIENT; AF2CPE5 is a seed-42 matched continuation comparison against AF2CPE0. Preserve these comparator boundaries in future reporting.

## 8. No unsupported explanation rule

At this snapshot, record measured outcomes separately from hypotheses. In particular:

- the seed-2026 `biji_muda` AF2_ORIENT drop has not yet been causally diagnosed;
- the AF2+CPE rejection has not yet established a specific optimization or representation-drift mechanism;
- the repeated failures of direct AF2 compositions do not by themselves prove a universal incompatibility law;
- AF2FFAB2 is the only currently confirmed positive AF2 composition in this ledger, but it is a Pareto refinement rather than a universal winner over original AF2.
