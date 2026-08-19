# Faruq-v3 Model Experiment Master Log — WAV1 Addendum

Date: 2026-08-19

## WAV1 paired multiseed confirmation

**Status: PASS versus D0FT on grouped Faruq-v3 validation; locked test remains closed.**

Standalone `WAV1` uses the frozen two-level Haar wavelet frontend from the AF2 spectral-factorization study. Seed 42 was reused from the completed Stage-2 run and only seeds 123 and 2026 were newly trained from their corresponding D0 checkpoints.

| Model | Macro mean | Bottom-3 mean | Worst mean | Evidence status |
|---|---:|---:|---:|---|
| D0FT | 86.62% | 76.58% | 73.05% | paired control |
| **WAV1** | **87.40%** | **79.79%** | **77.01%** | **PASS vs D0FT** |

Mean paired WAV1-minus-D0FT gains were **+0.78 Macro**, **+3.21 Bottom-3**, and **+3.96 Worst-class points**. WAV1 improved all three headline metrics in 2/3 seeds. The frozen five-part confirmation gate passed completely.

Per-seed paired deltas (Macro / Bottom-3 / Worst):

- seed 42: `+1.72 / +8.30 / +10.01` points;
- seed 123: `+1.46 / +2.81 / +7.30` points;
- seed 2026: `-0.83 / -1.46 / -5.44` points.

The negative seed-2026 direction is important. WAV1 is therefore validation-robust under the frozen aggregate gate, but it is more seed-sensitive than AF2 on the current three-seed evidence.

Contextual three-seed means after this completion:

| Model | Macro | Bottom-3 | Worst | Status |
|---|---:|---:|---:|---|
| AF2 | **87.94%** | 79.37% | 78.15% | PASS vs D0FT |
| STB1 | 87.82% | **80.50%** | **78.36%** | paired validation; spatial-causal gate FAIL |
| IGEM1 | 87.71% | 79.27% | 77.74% | PASS vs D0FT |
| ACMC1 | 87.62% | 79.13% | 76.30% | paired validation; locked-test NOT_CONFIRMED |
| WAV1 | 87.40% | **79.79%** | 77.01% | **PASS vs D0FT** |

This comparison is descriptive only. The WAV1 protocol did not freeze direct superiority tests against AF2, IGEM1, STB1, ACMC1, or AF1.

The earlier spectral-factorization decision is not rewritten: WAV1 remained REJECT in that study because its seed-42 Macro gain over AF2C was only +0.213 point, below the frozen +0.5-point retain threshold. Likewise, the later AF2+WAV fusion failure remains closed. The new result establishes a different point: **standalone WAV1 is independently validation-robust relative to D0FT across the frozen three seeds.**

Authoritative result:
`docs/FARUQ_V3_WAV1_PAIRED_CONFIRMATION_RESULT_2026-08-19.md`

Machine-readable evidence:
`docs/evidence/FARUQ_V3_WAV1_PAIRED_CONFIRMATION_2026-08-19.json`
