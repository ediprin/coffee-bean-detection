# Faruq-v3 WAV1 Paired Multiseed Confirmation Protocol

Date frozen: 2026-08-19
Status: **frozen before seed-123/2026 WAV1 training**

## Question

Does the seed-42 WAV1 result remain a validation-robust improvement over the optimization-matched D0FT control across seeds 42, 123, and 2026, and where does its three-seed aggregate sit descriptively relative to AF2, IGEM1, STB1, and ACMC1?

This is a follow-up to the completed AF2 spectral-factorization study. WAV1 was not retained by the old seed-42 gate because its Macro gain over AF2C was only +0.213 point, below that study's frozen +0.5-point requirement. That rejection is not changed here. The present study asks a different question: whether standalone WAV1 is stable across seeds when compared with the same D0FT control used to confirm AF2 and IGEM1.

## Frozen data and seeds

- Dataset: Faruq-v3 grouped development split only.
- Train: 1,665 images.
- Validation: 294 images.
- Classes: 21, all required in validation.
- Seeds: 42, 123, 2026.
- Seed 42 WAV1 is reused from the completed spectral Stage-2 run; it is not retrained.
- The seed-42 metrics are frozen in repository evidence `docs/evidence/FARUQ_V3_WAV1_SEED42_RESULT_2026-08-19.json`; a Kaggle Saved Version of seed42 is not required for this confirmation.
- Only WAV1 seeds 123 and 2026 are newly trained.
- Each new seed starts from the existing seed-matched D0 checkpoint.
- Faruq locked test is not restored, read, or reopened.

## Frozen WAV1 seed-42 observation

Repository evidence snapshot: `docs/evidence/FARUQ_V3_WAV1_SEED42_RESULT_2026-08-19.json`, transcribed from the completed Kaggle WAV1 seed-42 result before any seed-123/2026 training.

- Macro mAP50-95: 0.8841052369918866
- Bottom-3 mAP50-95: 0.8327607439278027
- Worst-class mAP50-95: 0.8203489485589485
- Median batch-1 latency at 640 px: 15.922472000056587 ms
- WAV1 checkpoint SHA256: `ff8d06f2f9b98ae005c1b60d67613e3397eb541dc6420a3d2b069f8cd56ac426`
- Initial D0 seed42 SHA256: `0c458841b84bedce4e0ddada6a5773f6a5ac8a91dad084a4a5f24e89f04e6367`
- evaluation split: validation
- test accessed: false

The seed-42 checkpoint itself is not needed because this confirmation uses only its already completed validation metrics. The seed-42 result is never retrained or recomputed.

## Frozen implementation

WAV1 remains exactly the spectral-factorization implementation already audited and run at seed 42:

- two-level orthonormal Haar decomposition;
- luminance input;
- LH/HL/HH detail energy at each level;
- each level resized to input resolution and min-max normalized;
- mean detail-energy cue across the two levels;
- cue expanded across RGB and applied through the same AFAB residual gate;
- parameter-free frontend;
- native YOLO26n-P3 detector otherwise unchanged.

Configuration: `configs/af2_spectral/WAV1_yolo26n.yaml`.

Frozen training schedule is exactly the existing WAV1 schedule: 50 epochs, imgsz 640, batch 16, workers 2, patience 15, optimizer auto, no pretrained model replacement, close_mosaic 10, max_det 500, deterministic seed-specific training.

No wavelet level, fusion, scale, threshold, color conversion, or training hyperparameter may be tuned after seeing seeds 123/2026.

## Primary paired control

The primary control is the already completed seed-matched D0FT result from the AF2/IGEM paired-confirmation evidence. The three D0FT seed rows are reused without retraining.

The WAV1 validation-robustness gate is deliberately identical to the gate previously used for AF2 and IGEM1:

1. Mean paired Macro gain versus D0FT >= +0.5 percentage point.
2. Macro improves versus D0FT in at least 2/3 seeds.
3. Mean Bottom-3 versus D0FT is not lower.
4. Bottom-3 improves versus D0FT in at least 2/3 seeds.
5. Mean Worst-class decline versus D0FT is no greater than 1.0 percentage point.

All five criteria must pass.

## Frozen secondary comparison set

After the primary decision is computed, report three-seed aggregate means for WAV1 beside these already completed references:

- AF2: Macro 0.8793765273831853; Bottom-3 0.7937036279638393; Worst 0.7815268371194097. Status: PASS vs D0FT.
- IGEM1: Macro 0.8771367301594344; Bottom-3 0.7926573397931215; Worst 0.7773973469475308. Status: PASS vs D0FT.
- STB1: Macro 0.8781987645071222; Bottom-3 0.8049539441492847; Worst 0.7835956356579805. Status: paired validation reference; spatial-causal gate vs CMC0 FAIL.
- ACMC1: Macro 0.8762; Bottom-3 0.7913; Worst 0.7630. Status: paired validation candidate; later locked-test confirmation NOT_CONFIRMED.

AF2/IGEM1 and STB1 values come from repository machine-readable evidence. ACMC1 is retained only as the existing rounded three-seed contextual reference from the model master log; therefore direct sub-basis-point comparison with ACMC1 is not allowed.

This secondary table is descriptive. It does not replace the primary D0FT gate and does not create a post-hoc superiority test among mechanisms with different original confirmatory questions.

## Decision and stopping rule

- Primary PASS: WAV1 may be described as validation-robust versus D0FT across the frozen three seeds. Report its descriptive rank among the frozen reference set. Do not reopen the locked test.
- Primary FAIL: stop WAV1 as a multiseed candidate. No retuning, fourth seed, fusion, or locked-test evaluation is authorized.

Regardless of PASS/FAIL, preserve all per-seed deltas and lower-tail results. A high seed-42 result alone is not sufficient evidence of robustness.
