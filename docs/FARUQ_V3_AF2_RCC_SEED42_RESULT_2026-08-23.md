# AF2 recovered-cue calibration seed-42 result

Status: **completed -- FAIL; stop AF2-RCC without test or extra seeds.**

## Frozen comparison

`AF2RCC1` freezes the original AF2 detector and trains only 189 bounded
P3/P4/P5 class-calibration weights for 20 epochs. The comparison uses the
original AF2 seed-42 validation result. All 21 validation classes were present
and test remained locked.

| Metric | Original AF2 | AF2RCC1 | Delta |
|---|---:|---:|---:|
| Macro mAP50-95 | 88.1973% | 88.1940% | -0.0034 point |
| Bottom-3 mAP50-95 | 80.0428% | 80.0428% | 0.0000 point |
| Worst-class mAP50-95 | 79.3470% | 79.3470% | 0.0000 point |

The target class `kulit_tanduk_ukuran_kecil` also changed by exactly zero.
None of the three headline metrics improved, so the frozen requirement that at
least two improve failed. The aggregate decision is `FAIL` and the next action
is `STOP_AF2_RCC`.

## Interpretation

The already-recovered AF2 RGB cue did not provide useful additional signal
through a 189-parameter frozen-detector logit calibration. This is a valid
null result: it does not invalidate original AF2, but it closes this specific
calibration direction. No seed 123/2026 confirmation and no test evaluation
are authorized.

Protocol:
`docs/FARUQ_V3_AF2_RECOVERED_CUE_CALIBRATION_PROTOCOL_2026-08-22.md`.

Raw evidence:
`docs/evidence/FARUQ_V3_AF2_RCC_SEED42_DECISION_2026-08-23.json`.

Drive source:
`experiments/faruq-v3-af2-recovered-cue-calibration-v1/val_reports/af2_rcc_seed42_decision.json`.
