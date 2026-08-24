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

## AF2 isolated radial/orientation factorization

**Status: completed at seed 42 -- AF2_ORIENT RETAIN; AF2_RADIAL REJECT; test
closed.** The study isolated the two CAFR factors directly against original
AF2, without the cumulative CAFR changes:

| Model | Macro | Bottom-3 | Worst | Delta vs AF2 (Macro / Bottom-3 / Worst) |
|---|---:|---:|---:|---:|
| AF2 reference | 88.20% | 80.04% | 79.35% | control |
| AF2_RADIAL | 86.57% | 78.21% | 75.41% | -1.62 / -1.83 / -3.94 points |
| **AF2_ORIENT** | **88.33%** | **81.38%** | **80.14%** | **+0.13 / +1.34 / +0.79 points** |

Fixed three-band radial separation reduced every primary metric and is
rejected. Unsigned 180-degree orientation folding improved all three metrics
and is retained for paired seeds 123/2026. The frozen isolation protocol did
not define a +0.5-point Macro threshold, so such a threshold is not introduced
after observing the results. The radial-plus-orientation combination is not
authorized because only the orientation factor was useful. Test was not
accessed.

Protocol: `docs/AF2_ISOLATED_RADIAL_ORIENTATION_PROTOCOL_2026-08-21.md`.

Full result:
`docs/FARUQ_V3_AF2_ISOLATED_RADIAL_ORIENTATION_RESULT_2026-08-21.md`.

## AF2_ORIENT paired confirmation

**Status: completed -- FAIL across seeds 42/123/2026; retain original AF2;
test closed.** The seed-42 lower-tail improvement repeated at seed 123 but not
at seed 2026:

| Metric | AF2 mean | AF2_ORIENT mean | Mean delta | Improved seeds |
|---|---:|---:|---:|---:|
| Macro | 87.94% | 88.26% | +0.32 point | 2/3 |
| Bottom-3 | 79.37% | 79.25% | -0.12 point | 2/3 |
| Worst class | 78.15% | 76.66% | -1.50 points | 2/3 |

The candidate passed both Macro criteria, but failed the frozen requirement of
at least +0.5-point mean Bottom-3 gain and failed non-negative mean Worst-class
delta. Seed 2026 was a class-redistribution failure rather than a global
collapse: `biji_muda` fell 12.82 points, while several other classes improved,
leaving Macro almost unchanged. Folding antipodal FFT directions therefore
does not provide a stable difficult-class advantage. No combined arm, extra
orientation tuning, or test access is authorized.

Protocol:
`docs/FARUQ_V3_AF2_ORIENT_PAIRED_CONFIRMATION_PROTOCOL_2026-08-21.md`.

Full result:
`docs/FARUQ_V3_AF2_ORIENT_PAIRED_CONFIRMATION_RESULT_2026-08-21.md`.

## AF2 mechanism diagnostic

**Status: completed -- CLASSIFICATION_DOMINANT across paired seeds
42/123/2026; no training or test access.** Raw top-500 proposal accessibility
was already saturated and did not improve (99.81% D0FT versus 99.75% AF2,
-0.06 point). In contrast, final proposal accessibility rose from 77.63% to
89.54% (+11.91 points), localization-conditioned Top-1 accuracy rose from
62.46% to 70.58% (+8.12), and correct-decision recall rose from 48.54% to
63.18% (+14.64). All final-output and conditional classification directions
improved in 3/3 seeds.

The final accessibility gain is attributed to improved class confidence and
candidate ranking because the underlying raw geometric candidate pool did not
improve. AF2 therefore supports a fine-grained discrimination/ranking claim,
not a raw-localization claim. Per-class effects remain heterogeneous, so the
result does not establish universal class improvement. This remains post-hoc
validation association rather than causal proof or a new selection gate.

Protocol:
`docs/FARUQ_V3_AF2_MECHANISM_DIAGNOSTIC_PROTOCOL_2026-08-21.md`.

Notebook:
`notebooks/Faruq_V3_AF2_Mechanism_Diagnostic_Colab.ipynb`.

Full result:
`docs/FARUQ_V3_AF2_MECHANISM_DIAGNOSTIC_RESULT_2026-08-21.md`.

## AF2 same-device efficiency audit

**Status: completed -- all validity gates passed; no training or data/test
access.** D0FT and original AF2 were measured in paired FP32 tensor-forward
runs at seeds 42/123/2026 on a Tesla T4. Both models retained exactly
2,511,990 parameters and identical 10,124,840-byte serialized tensor state.
AF2 median latency was 23.59 ms versus 13.52 ms (1.745x), p95 latency was
33.78 ms versus 19.15 ms (1.767x), and throughput was 39.96 versus 68.93
image/s (0.581x). Peak allocated CUDA memory rose from 75.2 MB to 127.6 MB
(1.696x). The checkpoint-file increase was only 11,254 bytes.

The result supports `parameter-free frontend`, but explicitly rejects any
interpretation of `compute-free` or `memory-free`. Combined with the completed
accuracy confirmation (+1.32 Macro, +2.80 Bottom-3, +5.10 Worst-class points),
AF2 provides a lower-tail accuracy benefit in exchange for substantial FFT
latency and temporary memory. Host-to-device transfer, postprocessing, I/O,
and standard YOLO FLOPs remain outside the claim.

Protocol:
`docs/FARUQ_V3_AF2_EFFICIENCY_AUDIT_PROTOCOL_2026-08-21.md`.

Full result:
`docs/FARUQ_V3_AF2_EFFICIENCY_AUDIT_RESULT_2026-08-21.md`.

## Thesis evidence freeze

**Status: completed -- model search closed; original AF2 selected as the thesis
model.** The repository evidence has been consolidated into a claim-level
matrix and thesis blueprint. The primary claim is the controlled adaptation of
the parameter-free AF2 frequency-angular frontend to an end-to-end YOLO26
detector for the grouped SNI-21 task. D0FT remains the mandatory optimization
control. Three-seed in-domain validation, classification-dominant mechanism
diagnosis, target-free Coffee Standard evaluation, negative stress tests, and
same-device efficiency jointly define the contribution and its limits.

The freeze explicitly forbids claims of a newly invented FFT algorithm,
cross-paper SOTA, general illumination robustness, real 300-gram readiness,
full-system 40 FPS, uniform per-class improvement, or untouched Faruq test
confirmation. No further training or test access is authorized by this entry.

Evidence matrix:
`docs/FARUQ_V3_AF2_THESIS_EVIDENCE_MATRIX_2026-08-21.md`.

Thesis blueprint:
`docs/FARUQ_V3_AF2_THESIS_BLUEPRINT_2026-08-21.md`.

## AF2 reused Faruq-test post-hoc evaluation

**Status: completed -- POSTHOC_DIRECTION_POSITIVE; reused test, not a new
locked-test confirmation.** The original AF2 checkpoints at seeds 42/123/2026
were evaluated against the historical D0FT reports on the identical
129-parent Faruq test package. Test-manifest SHA matched, no training or tuning
was performed, and AF2 improved every primary metric in all three seeds:

| Metric | D0FT mean | AF2 mean | Mean delta | Minimum delta | Improved seeds |
|---|---:|---:|---:|---:|---:|
| Macro | 86.31% | 88.33% | +2.02 points | +1.00 | 3/3 |
| Bottom-3 | 72.85% | 76.49% | +3.65 points | +1.44 | 3/3 |
| Worst class | 69.07% | 71.46% | +2.38 points | +1.05 | 3/3 |

The 1,000-iteration paired-parent bootstrap gave a +1.93-point custom Macro
delta, 95% interval +0.13 to +3.86 points, and 98.1% positive probability.
Five of 21 classes still had negative mean deltas, so universal class
improvement is not supported.

The output must remain labeled
`REUSED_TEST_POSTHOC_NOT_LOCKED_CONFIRMATION`. The positive result strengthens
the in-domain direction but does not replace the absence of a new untouched
AF2 test. No further tuning is authorized.

Protocol:
`docs/FARUQ_V3_AF2_REUSED_TEST_POSTHOC_PROTOCOL_2026-08-21.md`.

Notebook:
`notebooks/Faruq_V3_AF2_Reused_Test_Posthoc_Colab.ipynb`.

Full result:
`docs/FARUQ_V3_AF2_REUSED_TEST_POSTHOC_RESULT_2026-08-21.md`.

## AF2 feature-frequency classification adapter

**Status: protocol and implementation frozen before training on 2026-08-22;
seed-42 screening not yet executed; test remains locked.** This user-authorized
direction reopens validation-only model screening after the earlier thesis
evidence freeze. It follows the completed `CLASSIFICATION_DOMINANT` mechanism
diagnosis and does not modify the AF2 image frontend or YOLO26 box pathway.

Two capacity-matched arms are frozen. `AF2FFA0` receives a zero spectral
descriptor, while `AF2FFA1` receives the fixed high-frequency energy ratio of
each P3/P4/P5 channel. Both start as exact AF2 identities, add the same 1,344
parameters (about 0.054% for YOLO26n), and use identical 30-epoch continuation
schedules from the seed-matched AF2 checkpoint. Static audit, arm runner,
seed-42 decision, paired confirmation decision, unit tests, and a compact
validation-only Colab notebook are implemented.

Seed-42 PASS requires at least +0.5 Macro point, no Bottom-3 decrease, and no
more than 1-point Worst-class decrease versus the capacity control. FAIL stops
the direction without extra seeds or test access.

Protocol:
`docs/FARUQ_V3_AF2_FEATURE_FREQUENCY_ADAPTER_PROTOCOL.md`.

Notebook:
`notebooks/Faruq_V3_AF2_Feature_Frequency_Adapter_Colab.ipynb`.

## AF2-FFA bounded Pareto refinement

**Status: AF2FFAB1 completed and rejected, but marked
`INVALID_OPTIMIZATION_CONFOUND`; test remained locked.** The completed AF2FFA1 screen
improved Bottom-3/Worst by +0.82/+2.87 points while losing 0.33 Macro point to
its continuation control. Rather than paying immediately for two confirmation
seeds, one bounded candidate (`AF2FFAB1`) caps the learned spectral residual at
10% through `0.10*tanh(alpha)`. The completed AF2FFA0 and AF2FFA1 reports are
reused after exact seed/source-checkpoint validation, so only AF2FFAB1 trains.

The candidate is evaluated as a Pareto tail refinement: it must recover Macro
against AF2FFA1, remain within 0.1 point of AF2FFA0 Macro, retain meaningful
Bottom-3/Worst gains over AF2FFA0, and preserve the unbounded candidate's tail
signal. A retained result defers rather than automatically starts multiseed.

Protocol:
`docs/FARUQ_V3_AF2_FFA_BOUNDED_REFINEMENT_PROTOCOL_2026-08-22.md`.

Result: AF2FFAB1 produced 85.09% Macro, 73.33% Bottom-3, and 66.61% Worst.
Post-run inspection found that `0.10*tanh(alpha)` reduced the initial amplitude
gradient from 1.0 to 0.10, so the run did not isolate gain bounding from
optimization speed. The raw result is preserved but cannot decide the bounded
hypothesis.

Result document:
`docs/FARUQ_V3_AF2_FFA_BOUNDED_REFINEMENT_RESULT_2026-08-22.md`.

## AF2-FFA gradient-matched bound correction

**Status: protocol frozen before AF2FFAB2 seed-42 training on 2026-08-22; one
new arm authorized; test remains locked.** AF2FFAB2 replaces the confounded
parameterization with `0.10*tanh(alpha/0.10)`, retaining the ±10% bound while
restoring the same unit initial derivative as AF2FFA1. Static audit must report
the derivatives explicitly before training. Historical AF2FFA0/AF2FFA1 results
are reused; AF2FFAB1 is recorded but excluded as a scientific comparator.

Protocol:
`docs/FARUQ_V3_AF2_FFA_GRADIENT_MATCHED_BOUND_PROTOCOL_2026-08-22.md`.

Seed-42 result: AF2FFAB2 achieved 88.89% Macro, 82.11% Bottom-3, and 80.49%
Worst. Against AF2FFA0 it preserved Macro (+0.003 point) while improving
Bottom-3/Worst by +1.27/+2.96 points. It also exceeded AF2FFA1 by
+0.33/+0.45/+0.08 points. All frozen criteria passed and the decision was
`RETAIN_PARETO`; test remained locked.

## AF2FFAB2 paired three-seed confirmation

**Status: completed -- PASS across seeds 42/123/2026; retain AF2FFAB2 as a
validated Pareto refinement; test remained locked.** For each seed, AF2FFA0
and AF2FFAB2 started from the same seed-matched AF2 checkpoint and used the
same 30-epoch continuation schedule. This explicit control prevents
attributing continuation optimization to the frequency adapter.

| Metric | AF2FFA0 mean | AF2FFAB2 mean | Mean delta | Minimum delta | Improved seeds |
|---|---:|---:|---:|---:|---:|
| Macro | 87.10% | **88.54%** | **+1.44 points** | +0.003 | 3/3 |
| Bottom-3 | 77.68% | **80.52%** | **+2.83 points** | +1.27 | 3/3 |
| Worst class | 74.69% | **76.96%** | **+2.27 points** | +0.89 | 3/3 |

All frozen criteria passed. The decision is `PASS` and the recorded next
action is `RETAIN_AF2FFAB2_AS_VALIDATED_PARETO_REFINEMENT`. This result proves
repeatable benefit against the matched continuation control, not universal
superiority over original AF2. Descriptively, AF2FFAB2 exceeds original AF2
on Macro/Bottom-3 by about +0.60/+1.15 points but is about 1.19 points lower
on Worst class.

Protocol:
`docs/FARUQ_V3_AF2_FFA_GRADIENT_MATCHED_PAIRED_PROTOCOL_2026-08-22.md`.

Result:
`docs/FARUQ_V3_AF2_FFA_B2_PAIRED_CONFIRMATION_RESULT_2026-08-22.md`.

Evidence:
`docs/evidence/FARUQ_V3_AF2_FFA_B2_PAIRED_CONFIRMATION_2026-08-22.json`.

## AF2 recovered-cue class calibration

**Status: protocol and implementation frozen before seed-42 training on
2026-08-22; test remains locked.** The AF2FFAB2 analysis showed that its mean
Macro/Bottom-3 improvement over original AF2 came with a class-specific Worst
trade-off, while its extra P3/P4/P5 FFT increased rather than reduced AF2's
inference overhead. This follow-up therefore does not add another spectral
transform or continue the full detector.

`AF2RCC1` reuses the spatial RGB cue already recovered by the original AF2
frontend and applies three bounded `21 x 3` projections directly to P3/P4/P5
classification logits. All native AF2 detector parameters are frozen; only
189 zero-initialized calibration weights train for 20 epochs. `AF2RCC0` is a
schema-matched zero-cue identity used in static audit only. The audit requires
one AF2 recovery call, no additional FFT/ROI/decoded-box dependency, exact
initial AF2 identity, finite calibration gradients, and bitwise-invariant box
outputs under an active correction.

Seed-42 passes only if Macro remains within 0.1 point of original AF2,
Bottom-3 is not lower, Worst remains within 0.5 point, at least two headline
metrics improve, and `kulit_tanduk_ukuran_kecil` remains within 0.5 point.
Only PASS can authorize paired seed 123/2026 confirmation.

Protocol:
`docs/FARUQ_V3_AF2_RECOVERED_CUE_CALIBRATION_PROTOCOL_2026-08-22.md`.

Notebook:
`notebooks/Faruq_V3_AF2_Recovered_Cue_Calibration_Colab.ipynb`.

Seed-42 result: **completed -- FAIL; direction closed without test or extra
seeds.** AF2RCC1 achieved 88.1940% Macro, 80.0428% Bottom-3, and 79.3470%
Worst, compared with original AF2 at 88.1973%, 80.0428%, and 79.3470%.
The deltas were -0.0034/0.0000/0.0000 point, the target class delta was zero,
and zero of three headline metrics improved. All dataset and test-lock gates
passed, but the frozen improvement gate failed. Decision: `FAIL`; next:
`STOP_AF2_RCC`.

Result:
`docs/FARUQ_V3_AF2_RCC_SEED42_RESULT_2026-08-23.md`.

Evidence:
`docs/evidence/FARUQ_V3_AF2_RCC_SEED42_DECISION_2026-08-23.json`.

## Retained-candidate limitation consolidation and next analysis

**Status: completed on 2026-08-23; no training and no new test access.** The
retained candidates were reclassified by evidence level rather than ranked by
a mixture of seed-42 screens, multi-seed confirmations, and optimization
controls. Original AF2 remains the best-supported overall method; IGEM1 is the
confirmed non-frequency alternative; AF2FFAB2 is a validated Pareto refinement
against its matched continuation control but does not dominate original AF2 on
Worst-class AP. STB1, SAF1, AF1, CPE0/CPE7, HVIP1, PW1, LPS1, and FTIF arms
remain discovery-only unless separately confirmed. FCT0, AF2R0, AF2FT30/
AF2CT30, and AF2FFA0 are controls and must not be presented as architectural
methods.

The direct joint-from-D0 combinations also closed without a clean win:
AF2STB1 lost 1.45/4.78/4.99 Macro/Bottom-3/Worst points to STB1; AF2IGEM1 lost
0.55/1.76/1.93 points to IGEM1; AF2SAF1 gained 0.75 Macro point but lost
0.48/0.67 Bottom-3/Worst points to SAF1. These results reject naive joint
stacking, not every parent-preserving or routing strategy.

### 2026-08-23 — AF2 parent-preserving SAF/IGEM protocol frozen

The direct-pair evidence above was reclassified as a joint-from-D0 test rather
than a test of whether SAF or IGEM can improve a completed AF2 parent. Protocol
`FARUQ_V3_AF2_PARENT_RESIDUAL_PROTOCOL_2026-08-23.md` freezes four new seed-42
arms: `AF2SAF0/1` and `AF2IGEM0/1`. In every arm, the completed AF2 frontend,
backbone, neck, box path, and native class path remain frozen. Only a
zero-initialized classification residual is trainable. Each feature candidate
has an identical zero-information capacity/optimization control. Static audit,
per-arm resumable runners, and separate Colab notebooks were added before any
training result. Test remains locked. No numerical result is recorded yet.

The earlier analysis-only restriction was subsequently superseded for exactly
the four AF2 parent-residual seed-42 arms recorded above. It remains active for
all other new fusion ideas, additional seeds, and all test access.

Full consolidation:
`docs/FARUQ_V3_RETAINED_CANDIDATE_LIMITATIONS_2026-08-23.md`.
