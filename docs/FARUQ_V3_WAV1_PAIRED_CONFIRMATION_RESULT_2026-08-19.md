# Faruq-v3 WAV1 Paired Multiseed Confirmation Result

Date: 2026-08-19  
Protocol: `faruq-v3-wav1-paired-confirmation-v1`  
Decision: **PASS — WAV1 is validation-robust versus the seed-matched D0FT control across seeds 42/123/2026.**  
Evaluation: Faruq-v3 grouped development validation only  
Locked test accessed: **no**

## Frozen question

Does the standalone two-level Haar-wavelet frontend (`WAV1`) retain a positive validation advantage across seeds 42, 123, and 2026 when compared with the same seed-matched D0FT control used for AF2 and IGEM1 confirmation?

Seed 42 was reused from the completed spectral Stage-2 run. Only WAV1 seeds 123 and 2026 were newly trained. No retuning, fusion, or locked-test evaluation was performed.

## Frozen implementation contract

Static frontend checks passed:

- config SHA256: `70de7874371344c362d48d953e70fc9a99201ed2c0f88e063c8c4eadce0335b4`
- arm: `WAV1`
- wavelet levels: 2
- parameter-free: true
- no persistent state: true
- CPU bitwise repeatable: true
- finite output and finite gradient: true

Seed-42 batch-1 latency at 640 px was 15.922472 ms median and 17.801745 ms p95 on `cuda:0` under the frozen 10-warmup / 50-iteration measurement.

## Per-seed validation results

| Seed | Model | Macro mAP50-95 | Bottom-3 | Worst class |
|---:|---|---:|---:|---:|
| 42 | D0FT | 86.6887% | 74.9809% | 72.0224% |
| 42 | **WAV1** | **88.4105%** | **83.2761%** | **82.0349%** |
| 123 | D0FT | 86.3509% | 74.7514% | 67.8296% |
| 123 | **WAV1** | **87.8121%** | **77.5581%** | **75.1321%** |
| 2026 | **D0FT** | **86.8096%** | **79.9933%** | **79.3107%** |
| 2026 | WAV1 | 85.9802% | 78.5336% | 73.8722% |

### Paired WAV1 minus D0FT deltas

| Seed | Macro | Bottom-3 | Worst class |
|---:|---:|---:|---:|
| 42 | +1.7218 pp | +8.2952 pp | +10.0125 pp |
| 123 | +1.4612 pp | +2.8067 pp | +7.3026 pp |
| 2026 | -0.8294 pp | -1.4597 pp | -5.4385 pp |

The seed-2026 direction is explicitly negative on all three headline metrics and is retained as part of the result. WAV1 therefore does not win every seed.

## Three-seed aggregate

| Model | Macro mean ± SD | Bottom-3 mean ± SD | Worst mean ± SD |
|---|---:|---:|---:|
| D0FT | 86.6164 ± 0.2377% | 76.5752 ± 2.9624% | 73.0542 ± 5.8097% |
| **WAV1** | **87.4010 ± 1.2663%** | **79.7892 ± 3.0588%** | **77.0131 ± 4.3944%** |
| Paired delta | **+0.7845 pp** | **+3.2141 pp** | **+3.9588 pp** |

WAV1 improved Macro in 2/3 seeds, Bottom-3 in 2/3 seeds, and Worst class in 2/3 seeds.

## Frozen gate

| Criterion | Result |
|---|---|
| Mean Macro gain at least +0.5 pp | **PASS** (+0.7845 pp) |
| Macro improves in at least 2/3 seeds | **PASS** |
| Mean Bottom-3 not lower | **PASS** (+3.2141 pp) |
| Bottom-3 improves in at least 2/3 seeds | **PASS** |
| Mean Worst decline no greater than 1 pp | **PASS** (mean +3.9588 pp) |

Final decision: **PASS**.

## Frozen descriptive comparison with existing three-seed references

This table is contextual only; the confirmatory decision above is WAV1 versus D0FT.

| Model | Macro mean | Bottom-3 mean | Worst mean | Evidence status |
|---|---:|---:|---:|---|
| AF2 | **87.9377%** | 79.3704% | 78.1527% | PASS vs D0FT |
| STB1 | 87.8199% | **80.4954%** | **78.3596%** | paired validation; spatial-causal gate FAIL |
| IGEM1 | 87.7137% | 79.2657% | 77.7397% | PASS vs D0FT |
| ACMC1 | 87.62% | 79.13% | 76.30% | rounded contextual reference; locked-test NOT_CONFIRMED |
| **WAV1** | 87.4010% | 79.7892% | 77.0131% | **PASS vs D0FT** |

Descriptively, WAV1 ranks below AF2/STB1/IGEM1/ACMC1 on Macro mean, but its Bottom-3 mean exceeds AF2 and IGEM1. Direct superiority claims against those models are not authorized by this protocol.

## Interpretation boundary

1. WAV1 is now independently supported as a **validation-robust standalone spectral frontend** versus D0FT.
2. The result is not evidence that WAV1 is superior to AF2, IGEM1, STB1, ACMC1, or AF1 across seeds.
3. Seed 2026 shows substantial negative direction, so WAV1 has greater seed sensitivity than AF2 on the current three-seed sample.
4. The earlier spectral-factorization decision remains unchanged: WAV1 was REJECT under that study's seed-42 retain gate versus AF2C because its Macro gain was below +0.5 pp.
5. The failed AF2+WAV direct-fusion result also remains unchanged; standalone WAV1 PASS does not reopen that fusion branch.
6. Faruq locked test remains closed.

## Next action

`REPORT_WAV1_VALIDATION_ROBUSTNESS_WITHOUT_REOPENING_TEST`.

No post-result retuning, fourth seed, fusion, or locked-test evaluation is authorized by this result.
