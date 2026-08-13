# Faruq-v3 Model Experiment Master Log

Snapshot date: 2026-08-13

This document consolidates the completed Faruq-v3 detector experiments that
led from the YOLO26n baseline through ACMC, breadth screening, and the latest
synthesis attempts. Numbers are transcribed from raw JSON reports in the shared
Drive project. Validation-only seed-42 discovery results are not mixed with
three-seed validation or locked-test evidence.

## Shared controls

| Control | Scope | Macro mAP50-95 | Bottom-3 | Worst class |
|---|---|---:|---:|---:|
| D0 | original seed-42 baseline | 79.97% | 68.72% | 65.09% |
| D0FT | optimization-matched seed-42 control | 86.69% | 74.98% | 72.02% |
| ACMC1 | seed-42 classification correction | 87.62% | 80.40% | 79.49% |

The large D0-to-D0FT change proves that every architectural candidate must be
compared with an optimization-matched control, not only the original D0.

## Earlier focused model results

| Experiment | Candidate result | Decision |
|---|---|---|
| Ontology marginal S0 | 84.06 / 74.46 / 70.96 | FAIL: conditional top-1 collapsed |
| Multilevel head MHC0 | 73.01 / 59.55 / 58.10 | FAIL |
| Multilevel head MHF1 | 78.31 / 64.02 / 61.81 | FAIL versus D0 despite beating MHC0 |
| Frozen residual FRM1 | 79.89 / 68.65 / 65.13 | FAIL: effectively tied with D0 |
| Hong YOLO26 transfer | Macro -4.92, conditional top-1 -16.89, Bottom-3 -23.58, Worst -33.01 points | FAIL |

Values in the middle column are Macro / Bottom-3 / Worst mAP50-95 unless the
row explicitly states deltas. The CM512 representation probe is separate from
detector AP: fused P3+P4+P5 reached 80.14% validation Macro-F1 versus 73.69% for
P5 alone, but that diagnostic did not itself authorize a superior detector.

## ACMC evidence chain

### Three-seed validation

| Model | Macro mean | Bottom-3 mean | Worst mean |
|---|---:|---:|---:|
| D0 | 80.12% | 66.58% | 60.18% |
| D0FT | 86.62% | 76.58% | 73.05% |
| ACMC1 | **87.62%** | **79.13%** | **76.30%** |

ACMC1 minus D0FT averaged +1.00 Macro, +2.56 Bottom-3, and +3.24 Worst-class
points. Macro improved in 3/3 seeds; both tail metrics improved in 2/3.

### Locked test

| Model | Macro mean | Bottom-3 mean | Worst mean |
|---|---:|---:|---:|
| D0FT | 86.31% | 72.85% | 69.07% |
| ACMC1 | 87.55% | 76.13% | 73.08% |
| Paired delta | +1.24 points | +3.29 points | +4.01 points |

All three seed pairs had positive point estimates, but paired-parent bootstrap
probability was 0.928 against the frozen 0.950 requirement and its 95% interval
included zero (-0.41 to +3.09 points). Final conclusion: **NOT_CONFIRMED**.

### Post-hoc stress tests

- Synthetic density B0--B3: ACMC1 Macro was directionally higher in all 4/4
  conditions, mean +0.49 point, but absolute AP collapsed; development-only.
- Adrian external source: ACMC1 was lower than D0FT in 3/3 seeds, mean Macro
  delta -0.30 point, while both models collapsed to about 3--4% Macro AP.
  Status: `DOES_NOT_SUPPORT_EXTERNAL_DIRECTION`.
- ACMC2 passed seed 42 (87.81 / 81.94 / 79.33) but failed the paired three-seed
  gate because mean Macro 87.56% was below ACMC1's 87.62%; keep ACMC1.
- ACMC1H reached 87.01 / 79.69 / 76.23 and failed against ACMC1 on every
  headline metric; stop that optimization.

## Canonical breadth screen

The following 23 rows are the current contents of
`faruq-v3-breadth-screening-batch-v1/master_results.json`. They use the master
controller's common retention/discovery gate.

| Family/arm | Macro | Bottom-3 | Worst | Master decision |
|---|---:|---:|---:|---|
| STB1 | **88.67%** | **83.64%** | 80.81% | RETAIN |
| AFAB AF2 | 88.20% | 80.04% | 79.35% | RETAIN |
| IGEM1 | 88.01% | 82.18% | **82.08%** | RETAIN |
| AFAB AF1 | 87.94% | 80.07% | 77.05% | RETAIN |
| SAF1 | 87.34% | 81.33% | 80.34% | RETAIN |
| CPE0 | 86.91% | 77.36% | 74.50% | RETAIN |
| HVIP1 | 86.90% | 81.21% | 78.36% | RETAIN |
| CPE7 | 86.56% | 76.25% | 72.70% | RETAIN |
| PW1 | 86.36% | 78.84% | 76.62% | RETAIN |
| SG1/LPS1 | 86.12% | 79.14% | 76.35% | RETAIN |
| SEMAUX/LPS1 | 86.12% | 79.14% | 76.35% | RETAIN* |
| PCL1 | 83.81% | 76.76% | 74.24% | REJECT |
| MRL1 | 83.81% | 71.42% | 70.19% | REJECT |
| SSCB S0 | 81.96% | 73.74% | 71.42% | REJECT |
| SSCB M0 | 80.84% | 72.20% | 71.77% | REJECT |
| PWCA SA0 | 79.57% | 67.66% | 66.84% | REJECT |
| CG1 | 79.34% | 64.52% | 56.84% | REJECT |
| DRNET | 77.74% | 63.53% | 57.40% | REJECT |
| APCL1 | 76.95% | 58.08% | 49.87% | REJECT |
| AFAB AF12 | 76.08% | 60.08% | 57.74% | REJECT |
| FBNR | 73.31% | 58.44% | 55.41% | REJECT |
| BH1 | 70.23% | 46.27% | 35.28% | REJECT |
| SSCB S1 | 67.65% | 47.87% | 35.09% | REJECT |

`*` Source conflict preserved: the candidate-local SEMAUX report says REJECT
under its stricter local Macro-retention threshold, while the later breadth
master recomputation says RETAIN under the common canonical gate. Do not cite
SEMAUX status without naming which gate is used.

## FTIF completion not yet merged into breadth master

The completed FTIF report exists in Drive but is absent from the 23-row
`master_results.json` snapshot. Its local decisions are therefore recorded
separately rather than silently inserted into the canonical master table.

| FTIF arm | Macro | Bottom-3 | Worst | Local decision |
|---|---:|---:|---:|---|
| FT1 | **87.72%** | **80.66%** | **80.24%** | RETAIN |
| FT2 | 87.39% | 79.19% | 75.22% | RETAIN |
| FT3 | 87.50% | 78.41% | 69.37% | RETAIN |

FT1 is the only FTIF arm competitive with ACMC1 across all three headline
metrics. These remain seed-42 discovery results, not confirmed superiority.

## Synthesis attempts

| Synthesis | Models | Result | Decision |
|---|---|---|---|
| AGSF | STB1 88.67/83.64/80.81; SYN0 86.98/79.69/78.76 | SYN0 lower on all metrics | FAIL; SYN1/SYN2 not run |
| SGFR | SGC0 89.07/84.52/82.37; SGI1 88.71/84.08/81.67 | IGEM residual lower than matched control | FAIL; SGF2 not run |
| FC-STB | FCT0 89.40/84.83/84.15; FCD1 89.19/83.24/79.16 | AF2 distillation lower than matched control | FAIL; no extra seeds/test |

The synthesis results show one repeated pattern: continued or controlled STB
fine-tuning can improve the frozen STB1 reference, but the proposed added
mechanism has not beaten its matched control. FCT0 is the highest seed-42 row
observed here, but it is an optimization control and has no multi-seed or test
confirmation.

## Current defensible status

1. ACMC1 remains the only architectural candidate with paired three-seed
   validation and locked-test evaluation. Its locked-test trend is positive but
   formally `NOT_CONFIRMED`.
2. STB1 is the strongest canonical seed-42 breadth candidate by Macro and
   IGEM1 by Worst-class AP.
3. AFAB, FTIF, and the other retained breadth candidates remain discovery
   evidence only.
4. AGSF, SGFR, and FC-STB are completed negative synthesis results. None
   authorizes additional seed or test access under its frozen protocol.
5. Cross-source generalization is unsupported; Adrian evaluation collapsed for
   both D0FT and ACMC1.

## FMH1 focal-modulation completion

FMH1 replaced STB1's shifted-window blocks with two official-style FocalNet
modulation blocks on each P3/P4/P5 classification input while preserving the
native YOLO26 box path and an exact D0 identity start.

| Model | Macro | Bottom-3 | Worst | Decision |
|---|---:|---:|---:|---|
| FMH1 | 87.60% | 79.40% | 78.99% | FAIL |

FMH1 was lower than STB1 by 1.07 Macro, 4.24 Bottom-3, and 1.82 Worst-class
points. It was lower than FCT0 by 1.80, 5.43, and 5.16 points respectively.
Every frozen criterion failed, so no capacity control, extra seed, or test
evaluation is authorized.

Protocol: `docs/FARUQ_V3_FOCAL_MODULATION_PROTOCOL.md`. Result:
`docs/FARUQ_V3_FOCAL_MODULATION_RESULT_2026-08-13.md`.

## Authoritative raw sources

- `experiments/faruq-v3-breadth-screening-batch-v1/master_results.json`
- `experiments/faruq-v3-breadth-screening-batch-v1/candidates/FTIF/val_reports/lfdet_ftif_seed42_screening.json`
- `experiments/faruq-v3-acmc-paired-confirmation-v1/val_reports/acmc1_paired_optimization_confirmation.json`
- `experiments/faruq-v3-acmc-locked-test-v2/faruq_v3_acmc_locked_test_summary.json`
- `experiments/faruq-v3-acmc2-paired-confirmation-v1/val_reports/acmc2_paired_optimization_confirmation.json`
- `experiments/faruq-v3-agsf-synthesis-v1/val_reports/agsf_core_seed42_decision.json`
- `experiments/faruq-v3-sgfr-frozen-synthesis-v1/val_reports/sgfr_geometry_seed42_decision.json`
- `experiments/faruq-v3-fcstb-distillation-v1/val_reports/fcstb_seed42_decision.json`
- `experiments/faruq-v3-focal-modulation-v1/val_reports/fmh1_seed42_decision.json`
