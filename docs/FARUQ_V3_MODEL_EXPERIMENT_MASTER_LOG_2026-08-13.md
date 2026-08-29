# Faruq-v3 Model Experiment Master Log

Snapshot date: 2026-08-16

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
   IGEM1 by Worst-class AP. STB1 passed its seed-42 capacity-near-matched
   screen, but **failed** the paired three-seed causal gate against CMC0:
   Macro gain was only +0.07 point against the frozen +0.50 requirement.
   STB remains a high-performing validation reference, not a confirmed spatial
   mechanism or a test-authorized candidate.
3. AFAB, FTIF, and the other retained breadth candidates remain discovery
   evidence only.
4. AGSF, SGFR, and FC-STB are completed negative synthesis results. None
   authorizes additional seed or test access under its frozen protocol.
5. Cross-source generalization is unsupported; Adrian evaluation collapsed for
   both D0FT and ACMC1.
6. FCT0 is the highest seed-42 optimization-control observation (89.40 / 84.83
   / 84.15), not an architectural winner. No candidate may claim superiority
   from that row without a matched mechanism comparison.

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

## STB capacity-causal control completion

`CMC0` was frozen before training to distinguish STB1's shifted-window spatial
interaction from added head capacity. CMC0 uses the same D0 initialization,
P3/P4/P5 placement, two-block depth, identity gate, and 50-epoch schedule as
STB1, but every operation is pointwise in H/W. CMC0 has 4,588,025 parameters
versus STB1's 4,589,201 (0.0256% difference).

| Model | Macro | Bottom-3 | Worst | Role |
|---|---:|---:|---:|---|
| D0FT | 86.69% | 74.98% | 72.02% | native optimization control |
| CMC0 | 87.10% | 81.87% | **81.31%** | capacity-near-matched non-spatial control |
| STB1 | **88.67%** | **83.64%** | 80.81% | shifted-window spatial candidate |

CMC0 first passed its viability gate against D0FT. Relative to CMC0, STB1
gained +1.57 Macro and +1.77 Bottom-3 points while losing only 0.49
Worst-class point, within the frozen one-point tolerance. Every seed-42 causal
criterion passed. This supports a spatial-token-mixing effect beyond parameter
count at seed 42; it does not yet establish a seed-robust or test-confirmed
causal effect. Paired CMC0/STB1 confirmation on seeds 123 and 2026 is now
authorized. Test remains locked.

Frozen protocol: `docs/FARUQ_V3_STB_CAPACITY_CAUSAL_CONTROL_PROTOCOL.md`.
Result: `docs/FARUQ_V3_STB_CAPACITY_CAUSAL_CONTROL_RESULT_2026-08-13.md`.

### Paired seed 42/123/2026 result

| Model | Macro mean | Bottom-3 mean | Worst mean |
|---|---:|---:|---:|
| CMC0 | 87.75% | 78.99% | 75.45% |
| STB1 | **87.82%** | **80.50%** | **78.36%** |
| STB1 minus CMC0 | +0.07 | +1.50 | +2.90 points |

Per-seed STB1-minus-CMC0 Macro deltas were `+1.57`, `-1.38`, and `+0.02`
points for seeds 42, 123, and 2026. Bottom-3 deltas were `+1.77`, `-1.70`,
and `+4.43`; Worst-class deltas were `-0.49`, `-0.31`, and `+9.52` points.

The paired confirmation **FAILS** because mean Macro gain (+0.07 point) is
below the frozen +0.50-point threshold. Although mean lower-tail metrics are
higher, Worst improves in only 1/3 seeds and its positive mean is driven by
seed 2026. The STB spatial-causal claim is stopped and test remains closed.

Full result:
`docs/FARUQ_V3_STB_CAPACITY_PAIRED_CONFIRMATION_RESULT_2026-08-14.md`.

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
- `experiments/faruq-v3-stb-capacity-control-v1/val_reports/stb_capacity_control_seed42_decision.json`
- `experiments/faruq-v3-stb-paired-confirmation-v1/val_reports/stb_capacity_paired_confirmation.json`
- `experiments/faruq-v3-af2-igem-paired-confirmation-v1/val_reports/af2_igem_paired_confirmation.json`

## Coffee Standard v8 external robustness update

The public Roboflow v8 split was rejected after audit because 14 augmented
parent identities crossed train, validation, and test. A leakage-safe external
diagnostic was constructed using one representative from each of 148
independent parent identities and 3,989 boxes from 18 directly equivalent
SNI-21 classes. No target-domain training was performed.

At seed 42, AF2 ranked first among all 11 retained models with 17.06% Macro
mAP50-95 versus 12.59% for D0FT (+4.47 points). Existing checkpoints then
confirmed the direction across three paired seeds:

| Metric | D0FT mean | AF2 mean | Mean paired delta | Improved seeds |
|---|---:|---:|---:|---:|
| Macro mAP50-95 | 11.43% | **15.51%** | **+4.08 points** | 3/3 |
| Bottom-3 | 0.25% | **0.35%** | +0.10 point | 2/3 |
| Worst class | 0.01% | **0.04%** | +0.02 point | 2/3 |

The external gate passed. This strengthens AF2's status from an in-domain
candidate to a target-free cross-dataset robustness direction. It does not
establish target-domain usability: absolute Bottom-3 and Worst AP remain near
zero. Full result:
`docs/COFFEE_STANDARD_V8_EXTERNAL_RESULT_2026-08-16.md`.

## AF2 and IGEM1 paired confirmation authorization

**Status: completed — AF2 PASS and IGEM1 PASS.** The paired confirmation reused
the completed three-seed D0FT controls and seed-42 breadth results, then trained
only AF2 and IGEM1 for seeds 123 and 2026. Test was not accessed.

| Model | Macro mean | Bottom-3 mean | Worst mean | Decision |
|---|---:|---:|---:|---|
| D0FT | 86.62% | 76.58% | 73.05% | control |
| **AF2** | **87.94%** | **79.37%** | **78.15%** | **PASS** |
| IGEM1 | 87.71% | 79.27% | 77.74% | **PASS** |

AF2 minus D0FT averaged +1.32 Macro, +2.80 Bottom-3, and +5.10
Worst-class points. Macro improved in 3/3 seeds, while Bottom-3 and Worst
improved in 2/3. IGEM1 minus D0FT averaged +1.10, +2.69, and +4.69 points
respectively, with the same improvement counts. Both candidates satisfied all
independently frozen gates.

AF2 is the descriptive lead between the two because its aggregate means are
slightly higher and its lower-tail dispersion is smaller. The protocol did not
freeze a direct AF2-versus-IGEM1 superiority test, so this is not a formal
pairwise superiority claim. No test reopening or post-result fusion/tuning is
authorized.

Frozen protocol:
`docs/FARUQ_V3_AF2_IGEM_PAIRED_CONFIRMATION_PROTOCOL.md`.

Full result:
`docs/FARUQ_V3_AF2_IGEM_PAIRED_CONFIRMATION_RESULT_2026-08-15.md`.

## DIDA-AF2 factorial study authorization

**Status: completed -- FAIL; stopped at seed 42 without test.** The static
audit was executed against the completed AF2 seed-42 checkpoint on 2026-08-17.
All implementation gates passed: exact factorial flags, identical model/state
schema and inference, safe paired views, finite auxiliary losses and
gradients, operational GT matching, classification-only auxiliary API, and no
test access. The subsequent training-only 2 x 2 decomposition produced:

| Arm | DG objective | FG margin | Macro | Bottom-3 | Worst |
|---|---:|---:|---:|---:|---:|
| AF2FT | no | no | 87.68% | 78.37% | 75.12% |
| AF2DG | yes | no | 87.05% | 76.31% | 73.13% |
| AF2FG | no | yes | 87.61% | 78.37% | 75.29% |
| AF2DGFG | yes | yes | 86.92% | 75.60% | 73.65% |

DG alone reduced Macro/Bottom-3/Worst by 0.62/2.06/1.99 points. FG alone was
approximately neutral, with -0.06/~0.00/+0.16 point changes. The joint arm
reduced the three metrics by 0.76/2.77/1.47 points relative to control and
failed all seven frozen criteria. Therefore seeds 123/2026 and test access are
not authorized. AF2's prior retained status is unchanged.

Protocol: `docs/FARUQ_V3_DIDA_AF2_FACTORIAL_PROTOCOL.md`.

Full result: `docs/FARUQ_V3_DIDA_AF2_FACTORIAL_RESULT_2026-08-17.md`.

## AF2 controlled illumination robustness

**Status: completed -- FAIL at seed 42; no confirmation seeds or test.** AF2's
mean Macro robustness advantage was +1.74 points, but it was positive in only
2/9 conditions. Mean Worst-class robustness was -2.32 points, below the frozen
-1 point tolerance. Warm (+5.66 Macro) and cool (+18.79) shifts drove the
positive mean; exposure, contrast, and localized shadow were predominantly
negative. Fixed AF2 therefore does not support a general illumination-
robustness claim. Its prior clean in-domain and Coffee Standard evidence is
unchanged.

Protocol: `docs/FARUQ_V3_AF2_ILLUMINATION_ROBUSTNESS_PROTOCOL.md`.
Result: `docs/FARUQ_V3_AF2_ILLUMINATION_ROBUSTNESS_RESULT_2026-08-17.md`.

## Raw-preserving adaptive AF2

**Status: completed -- FAIL at seed 42; stopped without illumination, extra
seeds, or test.** The static audit passed with 467 added parameters and exact
AF2 initialization. Both 30-epoch arms completed on grouped Faruq-v3
validation:

| Model | Macro | Bottom-3 | Worst |
|---|---:|---:|---:|
| Frozen AF2 | 88.20% | 80.04% | 79.35% |
| **AF2R0 control** | **89.55%** | **84.30%** | **83.97%** |
| AF2R1 illumination | 88.93% | 83.16% | 82.57% |

AF2R1 lost 0.62 Macro, 1.13 Bottom-3, and 1.40 Worst-class points to the
equal-parameter zero-information control. It preserved the original AF2
reference, but failed all three causal criteria against AF2R0. The adaptive
illumination mechanism is therefore rejected; fixed AF2's earlier retained
status is unchanged. Test was not accessed.

Protocol: `docs/FARUQ_V3_AF2_ADAPTIVE_RESIDUAL_GATE_PROTOCOL.md`.
Result: `docs/FARUQ_V3_AF2_ADAPTIVE_RESIDUAL_GATE_RESULT_2026-08-17.md`.

## AF2 channel-calibration factorization authorization

**Status: completed -- FAIL at seed 42; stopped without extra seeds or test.**
The static audit passed: `AF2FT30` retained AF2's 2,511,990 parameters, while
`AF2CAL3` added exactly three RGB residual-scale parameters and reproduced AF2
at initialization. Both matched 30-epoch arms completed:

| Model | Macro | Bottom-3 | Worst |
|---|---:|---:|---:|
| Frozen AF2 | 88.20% | 80.04% | 79.35% |
| AF2R0 reference | **89.55%** | **84.30%** | **83.97%** |
| `AF2FT30` control | 89.00% | 83.88% | 83.55% |
| `AF2CAL3` candidate | 88.77% | 83.72% | 83.00% |

AF2CAL3 lost 0.23 Macro, 0.17 Bottom-3, and 0.55 Worst-class points to
the matched continuation control. The AF2R0 gain is therefore not explained
by input-independent three-channel residual calibration. Seeds 123/2026 and
test access are not authorized. Fixed AF2's prior evidence is unchanged.

Protocol: `docs/FARUQ_V3_AF2_CHANNEL_CALIBRATION_PROTOCOL.md`.
Result: `docs/FARUQ_V3_AF2_CHANNEL_CALIBRATION_RESULT_2026-08-17.md`.

## AF2 spectral factorization

**Status: protocol frozen; implementation prepared; no training result.** The
next bounded study separates AF2's Fourier-window, orientation, radial-band,
threshold, and RGB/luminance assumptions into seven parameter-free one-stage
candidates: AF2WIN, AF2ORI, AF2POL, AF2SOFT, AF2LUM, PCG1, and WAV1. AF2C is
the historical bitwise control and is not retrained. Stage 1 screens the five
AF2 factorizations at seed 42; Stage 2 screens PCG1/WAV1; only a retained
global winner may receive paired seeds 123/2026. The locked Faruq test remains
closed. No outcome is asserted by this authorization entry.

Protocol: `docs/FARUQ_V3_AF2_SPECTRAL_FACTORIZATION_PROTOCOL.md`.

## Fresh detector-native low-rank bilinear residual

**Status: completed -- STOP_AFTER_SEED42; no test.** The matched linear and
quadratic arms each completed 50 epochs from the same official YOLO26n source
and identical initialized detector-state SHA. DLRBC improved Macro by 0.93
point and Bottom-3 by 1.86 points against `LRLIN_FRESH`, but Worst-class fell
10.45 points. The first frozen criterion (two metrics improve) passed, while
the maximum 0.5-point-drop criterion failed decisively. Seeds 123/2026 and an
AF2 factorial are not authorized.

`D0DIRECT` and `AF2DIRECT` are retained only as descriptive context because
their initialized detector-state SHA differs; they are not substituted for
the matched linear causal control or the missing `B0_FRESH` record.

Protocol: `docs/FARUQ_V3_DLRBC_FRESH_PROTOCOL_2026-08-26.md`.

Result: `docs/FARUQ_V3_DLRBC_FRESH_SEED42_RESULT_2026-08-26.md`.

## AF2 class-selective DLRBC residual

**Status: completed -- STOP_AFTER_SEED42; no extra seeds or test.** This
bounded follow-up did not combine the failed global
DLRBC head with AF2. It freezes the completed `AF2DIRECT_seed42` detector and
routes a zero-initialized rank-8 quadratic residual only to classes selected
by an AF2-versus-DLRBC train-only complementarity audit. Validation cannot
participate in class selection. Static verification against the real
AF2DIRECT checkpoint passed: initial raw boxes/scores are exactly AF2, active
residuals change only selected class scores, non-selected scores and raw boxes
remain bitwise equal, and gradients are finite. The candidate adds 44,796
parameters; only these residual/gate parameters trained for 20 epochs.

AF2CSD1 moved Macro from 80.79% to 80.86% (+0.07 point), Bottom-3 from
69.58% to 69.80% (+0.22 point), and left Worst-class unchanged at 66.95%.
Mean AP over train-selected classes improved only 0.23 point, below the frozen
0.50-point requirement. The direction therefore stopped without test or
extra seeds. The successful isolation but insufficient effect is retained as
a negative result; the AF2DIRECT baseline is not interchangeable with AF2
results from other initialization protocols.

Protocol: `docs/FARUQ_V3_AF2_CLASS_SELECTIVE_DLRBC_PROTOCOL_2026-08-26.md`.

Result: `docs/FARUQ_V3_AF2_CLASS_SELECTIVE_DLRBC_RESULT_2026-08-26.md`.
# AF2 uniform model soup — protocol frozen (2026-08-27)

- Status: `AUTHORIZED_VALIDATION_ONLY`; hasil belum tersedia.
- Tiga checkpoint AF2 terkonfirmasi (seed 42/123/2026) akan dirata-ratakan
  dengan koefisien uniform tanpa training atau pemilihan koefisien memakai val.
- Tujuan: menguji stabilisasi lower-tail tanpa perubahan arsitektur/parameter.
- Test tetap tertutup; jika gate gagal, AF2 asli dipertahankan.
- Protocol: `docs/FARUQ_V3_AF2_UNIFORM_MODEL_SOUP_PROTOCOL_2026-08-27.md`.

## AF2 box-score factorial

**Status: protocol frozen; validation-only implementation prepared; not yet
executed.** The study crosses raw one-to-one regression tensors and class-score
tensors from the optimization-matched D0FT seed-42 checkpoint and confirmed
AF2 seed-42 checkpoint. The four fixed arms are DD, DA, AD, and AA. Pure DD/AA
must reproduce both native post-processing exactly and their frozen historical
validation metrics within 0.2 point before either hybrid is interpreted.

The primary question is whether D0FT regression plus AF2 scores (`DA`) can
retain AF2 Macro while preserving or improving its lower tail. This is a
zero-training mechanistic diagnostic, not a new model result. Test remains
unavailable.

Protocol: `docs/FARUQ_V3_AF2_BOX_SCORE_FACTORIAL_PROTOCOL_2026-08-27.md`.

### AF2 box-score factorial result

**Status: completed -- AF2_BOX_SCORE_INTERACTION_NECESSARY.** DD and AA
reproduced the historical D0FT/AF2 metrics exactly and every validity gate
passed. D0FT boxes plus AF2 scores recovered most of AF2's gain but remained
0.12 Macro, 0.44 Bottom-3, and 0.62 Worst-class points below full AF2. AF2
scores are the dominant contributor, but AF2 regression remains necessary for
the complete lower-tail result. Naive branch replacement is stopped; test was
not accessed.

Result: `docs/FARUQ_V3_AF2_BOX_SCORE_FACTORIAL_RESULT_2026-08-27.md`.

## AF2 confidence-IoU alignment audit

**Status: protocol frozen; validation-only implementation prepared; not yet
executed.** The audit preserves each model's final candidate set, boxes, and
classes, then measures confidence alignment to same-class IoU and an oracle
confidence-only reranking upper bound. Native D0FT/AF2 metrics must exactly
reproduce the completed factorial before the audit is interpretable. A PASS
may authorize one matched AF2 quality-loss screen; no training or test is part
of this audit.

Protocol: `docs/FARUQ_V3_AF2_QUALITY_ALIGNMENT_PROTOCOL_2026-08-27.md`.

## AF2 complementary mechanisms

**Status: seed-42 completed -- PASS; `AF2SFS1` retained.** The static audit
passed and the four matched 30-epoch arms completed. Against `AF2CTRL`
(89.00% Macro, 83.88% Bottom-3, 83.55% Worst), the shared-P3
space/frequency selector reached 89.96% Macro, 84.19% Bottom-3, and 83.25%
Worst. Its deltas were +0.95, +0.31, and -0.29 points, respectively, so it
passed the prospectively frozen strict Macro route. `AF2FS1` and `AF2BHCL1`
were rejected.

This is a seed-42 screening success, not a confirmed three-seed or test result.
Only `AF2SFS1` may proceed after a paired seed-123/2026 confirmation protocol
is frozen. Test remains closed.

Protocol:
`docs/FARUQ_V3_AF2_COMPLEMENTARY_MECHANISMS_PROTOCOL_2026-08-28.md`.

Result:
`docs/FARUQ_V3_AF2_COMPLEMENTARY_MECHANISMS_RESULT_2026-08-28.md`.

### AF2SFS1 root-cause diagnostic authorization

**Status: completed -- INTERPRETABLE; active selector effect unsupported at
the fixed operating point.** Raw accessibility was unchanged at 99.81%.
AF2SFS1 gained 1.33 points final matched recall but lost 1.25 points
conditional top-1 accuracy, leaving correct-decision recall 0.19 point below
AF2CTRL. Normal and adapter-bypass AF2SFS1 had identical 66.35%
correct-decision recall. The completed +0.95-point Macro AP result therefore
cannot yet be attributed to active selector inference; it may reflect AP
ranking across thresholds or an optimization-mediated detector-weight change.
Test remained closed.

Protocol:
`docs/FARUQ_V3_AF2SFS1_ROOT_CAUSE_PROTOCOL_2026-08-28.md`.

Result:
`docs/FARUQ_V3_AF2SFS1_ROOT_CAUSE_RESULT_2026-08-28.md`.

### AF2SFS1 full-mAP intervention authorization

**Status: protocol frozen; validation-only implementation prepared; not yet
executed.** Native mAP is evaluated for AF2CTRL and four states of the same
AF2SFS1 checkpoint: normal, adapter bypass, spatial-only, and frequency-only.
The exact factorization separates direct selector AP effect from the
optimization-mediated AP effect retained after bypass. Per-class AP50/AP75 and
checkpoint tensor drift are reported. There is no training, post-hoc model
selection, or test access.

Protocol:
`docs/FARUQ_V3_AF2SFS1_MAP_INTERVENTION_PROTOCOL_2026-08-28.md`.

# 2026-08-29 — AF2-SPDS protocol frozen (not yet trained)

- Added a matched three-arm screen: `AF2BASE`, `AF2RGBDS`, and `AF2SPDS`.
- The design responds directly to the failed AF2MTS1 perturbation: auxiliary
  P3/P4/P5 decoders are read-only and removable; native Detect inputs and
  inference remain unchanged.
- Causal control separates generic RGB reconstruction from AF2-specific signal
  reconstruction (`AF2(x)-x`).
- Seed-42 development-only kill gate is frozen in
  `docs/FARUQ_V3_AF2_SIGNAL_PRESERVATION_DEEP_SUPERVISION_PROTOCOL_2026-08-29.md`.
- Test remains locked; no result claim exists until all three reports and the
  decision artifact are present.

## AF2-SPDS seed-42 completed

- `AF2BASE`: Macro/Bottom-3/Worst = 88.89/80.84/77.53%.
- `AF2RGBDS`: Macro/Bottom-3/Worst = 88.64/83.62/82.61%.
- `AF2SPDS`: Macro/Bottom-3/Worst = 88.65/84.67/83.44%.
- AF2-specific supervision added +1.05/+0.82 points Bottom-3/Worst over
  generic RGB deep supervision, but lost 0.24 point Macro versus the matched
  base and therefore failed the frozen gate.
- Result and diagnosis:
  `docs/FARUQ_V3_AF2_SPDS_RESULT_2026-08-29.md`.

# 2026-08-29 — AF2-SPDS refinement protocol frozen (not yet trained)

- Original AF2SPDS was cue-specific and strongly improved lower-tail metrics,
  but missed its Macro-preservation gate by 0.14 point beyond tolerance.
- Two isolated corrections are authorized: pure AF2-gate reconstruction
  (`AF2CUE1`) and late auxiliary-loss release (`AF2DECAY1`).
- The factorization, seed-42 gate, and test lock are documented in
  `docs/FARUQ_V3_AF2_SPDS_REFINEMENT_PROTOCOL_2026-08-29.md`.

## AF2-SPDS refinement seed-42 completed

- `AF2CUE1`: Macro/Bottom-3/Worst = 89.31/83.96/83.58%.
- `AF2DECAY1`: Macro/Bottom-3/Worst = 88.49/82.81/80.39%.
- AF2CUE1 improved matched AF2BASE by +0.42/+3.12/+6.05 points, but its
  Bottom-3 was 0.71 point below AF2SPDS. This exceeded the frozen 0.50-point
  tolerance by 0.21 point, so it is recorded as a rejected near-miss rather
  than retroactively retained.
- AF2DECAY1 was lower than AF2SPDS on all headline metrics.
- Formal aggregate decision: `FAIL_KILL_GATE`; retain original AF2; no test or
  confirmatory extra seeds. Because AF2CUE1 missed only one gate by 0.21 point
  while improving AF2BASE by +0.42/+3.12/+6.05 points, it is separately marked
  `RETAIN_PARETO_EXPLORATORY` for validation-only per-class analysis. This does
  not revise the frozen decision.
- Result: `docs/FARUQ_V3_AF2_SPDS_REFINEMENT_RESULT_2026-08-29.md`.

# 2026-08-29 — AF2 variant collision audit and AF2RN protocol

**Status: repository-wide collision audit completed; AF2RN protocol frozen;
training not yet authorized.** Completed and protocol-only AF2 work was
cross-checked across the current tree and the historical AF2, CAFR,
radial-wavelet, orientation-confirmation, feature-frequency-adapter, and
parent-combination branches.

The audit closes accidental repetition of AF1/AF12 high-pass masking,
Hann-windowing, 180-degree folding, fixed radial bands, soft thresholds,
luminance sharing, wavelet/phase-congruency fusion, trainable channel gates,
feature adapters, auxiliary losses, and tested AF2/STB/IGEM/SAF compositions.

One distinct hypothesis remains: `AF2RN` normalizes FFT magnitude by the
median magnitude within the same integer annulus **before** forming the legacy
360-bin angular density. Radius is used only as a nuisance-normalization index;
it never masks, removes, or separately thresholds a radial band. All remaining
AF2 behavior is held fixed.

Training is blocked until a static and train-only observability audit passes.
The collision evidence and frozen protocol are:

- `docs/FARUQ_V3_AF2_VARIANT_COLLISION_MATRIX_2026-08-29.md`;
- `docs/evidence/FARUQ_V3_AF2_VARIANT_COLLISION_MATRIX_2026-08-29.json`;
- `docs/FARUQ_V3_AF2_RADIAL_NORMALIZED_ANGULAR_DENSITY_PROTOCOL_2026-08-29.md`.
