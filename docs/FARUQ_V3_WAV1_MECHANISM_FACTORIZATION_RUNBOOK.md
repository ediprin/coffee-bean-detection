# Faruq-v3 WAV1 Mechanism Factorization — Execution Runbook

Status: execution companion to the frozen protocol. This document does not change the scientific contract in `FARUQ_V3_WAV1_MECHANISM_FACTORIZATION_PROTOCOL.md`.

## 1. Build the small Kaggle add-on

Open and run all cells in:

`notebooks/Faruq_V3_WAV1_Factorization_Kaggle_Addon_Colab.ipynb`

The notebook:

1. mounts the project Drive;
2. clones `agent/wav1-mechanism-factorization`;
3. locates the already-frozen `WAV1_seed42_result.json` by its exact validation contract and frozen headline metrics;
4. builds `wav1_factorization_kaggle_manifest.json` with byte count and SHA256;
5. uploads/versions the private Kaggle dataset `faruq-v3-wav1-factorization-addon-v1`.

The add-on contains no D0 checkpoint, no Faruq image archive, and no locked-test data. Those remain in the existing private core.

## 2. Prepare the Kaggle notebook

Open:

`notebooks/Faruq_V3_WAV1_Factorization_Stage1_Sequential_Kaggle.ipynb`

Attach exactly the required experiment inputs:

- existing private core: `faruq-v3-experiment-core-v1`;
- add-on: `faruq-v3-wav1-factorization-addon-v1`.

A prior `wav1-factorization-stage1-output.zip` may also be attached for exact-contract resume. Resume is accepted only when arm, seed, config SHA, D0 SHA, epoch contract, and test-lock flag match.

Use one GPU. The screen is frozen to seed 42.

## 3. What Run All does before training

The Kaggle notebook stops before training unless all checks pass:

- both core and add-on manifests exist exactly once;
- the existing AF2 spectral core passes its original SHA/load-test contract;
- the add-on WAV1 reference matches the frozen seed-42 result;
- unit/contract tests for the mechanism operator pass;
- `WAV1_REF` is bitwise equal on CPU to the confirmed WAV1 operator;
- all new frontends are finite, active, state-free, differentiable, and shape/dtype preserving;
- the seed-42 D0 SHA matches the static audit;
- deterministic CUDA smoke checks pass;
- the development root does not expose a `test` split.

## 4. Frozen execution order

The notebook runs:

1. `HP1`
2. `WAV_L1`
3. `WAV_L2`
4. `WAV_RAWFUSE`

Each arm starts from the same D0 seed-42 checkpoint and uses the same 50-epoch schedule. `WAV1_REF` is never retrained.

After each arm, the notebook writes a fresh `wav1-factorization-stage1-output.zip` snapshot so an interrupted Kaggle session can be resumed without generic checkpoint discovery.

## 5. Final seed-42 report

After all four arms complete, the notebook creates:

`val_reports/wav1_factorization_seed42_report.json`

The report compares each causal arm with the frozen D0FT and WAV1 seed-42 references and reports:

- Macro mAP50-95;
- Bottom-3 mAP50-95;
- Worst-class mAP50-95;
- gain versus D0FT;
- fraction of the WAV1 gain retained for each headline metric;
- Pearson correlation between per-class AP-delta patterns when full classwise metrics are available.

The report intentionally returns `MECHANISTIC_REVIEW_REQUIRED`. It does not select a winner and does not authorize additional seeds.

## 6. Stop boundary

After the seed-42 report:

- do not tune HP1, wavelet levels, normalization, interpolation, RGB/luminance conversion, gating, or training schedule;
- do not run seed 123 or 2026 from this protocol;
- do not reopen Faruq locked test;
- do not create LH/HL/HH arms unless the result first supports one wavelet level as the dominant mechanism.

If one mechanistic explanation survives, freeze a separate paired-confirmation protocol before any additional seed training.
