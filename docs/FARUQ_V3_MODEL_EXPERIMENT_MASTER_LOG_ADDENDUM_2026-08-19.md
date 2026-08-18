# Faruq-v3 Model Experiment Master Log — Addendum 2026-08-19

This addendum extends `docs/FARUQ_V3_MODEL_EXPERIMENT_MASTER_LOG_2026-08-13.md`, whose embedded snapshot predates the completed AF2 spectral-factorization and radial-wavelet follow-up results. For AF2 spectral/refinement status, this addendum supersedes the earlier authorization-only wording until the consolidated master log is regenerated.

## AF2 spectral factorization — completed

AF2C remained the historical seed-42 control at 88.197% Macro / 80.043% Bottom-3 / 79.347% Worst mAP50-95.

Stage-1 candidates all failed the frozen +0.50-point Macro retention gate. Notable mechanistic observations were:

- `AF2POL`: 87.968% Macro / 83.260% Bottom-3 / 83.138% Worst. Relative to AF2C: -0.229 / +3.217 / +3.791 points. REJECT because Macro did not improve by +0.50 point.
- `AF2ORI`: 88.000 / 81.813 / 80.394. REJECT.
- `AF2SOFT`: 87.909 / 82.614 / 80.091. REJECT.
- `AF2WIN`: 88.138 / 80.682 / 78.216. REJECT.
- `AF2LUM`: 87.392 / 81.717 / 81.040. REJECT.

Stage-2 controls:

- `PCG1`: 86.867 / 74.301 / 68.407, median latency about 43.15 ms. REJECT.
- `WAV1`: 88.411 / 83.276 / 82.035, median latency about 15.92 ms. Relative to AF2C: +0.213 / +3.233 / +2.688 points. REJECT because Macro gain was below the frozen +0.50-point threshold.

No spectral-factorization arm was retained. AF2C stayed selected; no extra seeds or test were authorized by this study.

## AF2 radial-wavelet refinement — completed

A follow-up protocol tested whether the two most interesting observations could improve legacy AF2 directly:

- `AF2RAD`: add only 3 radial bands to legacy 360-bin AF2.
- `AF2WAV`: fuse legacy AF2 cue with the exact WAV1 Haar cue by parameter-free pointwise max.
- `AF2RADWAV`: combine the two changes.

The pre-training static audit passed, including bitwise AF2C legacy equivalence, unchanged legacy AF2 angle map for AF2RAD, WAV1 cue bitwise equivalence, deterministic/finite forward-backward behavior, no trainable frontend parameters, and no test access.

| Arm | Macro | Bottom-3 | Worst | Median latency | Decision |
|---|---:|---:|---:|---:|---|
| AF2C | **88.197%** | 80.043% | 79.347% | historical control | KEEP |
| AF2RAD | 86.776% | 76.954% | 72.090% | 24.524 ms | REJECT |
| AF2WAV | 87.725% | **81.707%** | **80.817%** | 25.445 ms | REJECT |
| AF2RADWAV | 87.101% | 79.347% | 77.133% | 27.265 ms | REJECT |

Deltas versus AF2C:

- AF2RAD: -1.421 Macro / -3.089 Bottom-3 / -7.257 Worst points.
- AF2WAV: -0.472 / +1.664 / +1.470 points.
- AF2RADWAV: -1.096 / -0.695 / -2.214 points.

Final decision: **FAIL → `KEEP_AF2C_AND_STOP`**. No seed 123/2026, retuning, or locked-test access is authorized.

Mechanistic conclusion:

1. Radial decomposition alone does **not** explain AF2POL's lower-tail gain. The AF2POL observation must be treated as a cumulative `WIN + ORI + POL` interaction rather than an independently validated radial effect.
2. Standalone WAV1's useful result does **not** transfer through the tested AF2 max-cue fusion. AF2WAV is lower than both AF2C in Macro and standalone WAV1 across the headline metrics.
3. Combining radial and wavelet cues does not rescue either mechanism.

Canonical result report: `docs/FARUQ_V3_AF2_RAD_WAVELET_REFINEMENT_RESULT_2026-08-19.md`.

Machine-readable evidence: `docs/evidence/FARUQ_V3_AF2_RAD_WAVELET_REFINEMENT_RESULT_2026-08-19.json`.

Training notebook commit: `35ec68dfc1af5fa27ee29ee2acc86b6b4aa914c2`.

Faruq locked test remained closed.
