# Faruq-v3 STB-guided WAV-L1 Kaggle runbook

Branch: `agent/stb-guided-robust-wav-yolo`

Current authorization: **S2 seed42 only**.

Locked test: **closed**.

## 1. Build the STB1 teacher add-on once

Open in Colab:

`notebooks/Faruq_V3_STB_Guided_Teacher_Addon_Colab.ipynb`

Run all cells. It resolves the existing Drive project, validates the frozen
STB1 seed42 checkpoint, and creates/updates the private Kaggle dataset:

`faruq-v3-stb1-teacher-addon-v1`

The add-on contains only the STB1 teacher checkpoint and its integrity
manifest; it contains no test images.

## 2. Attach exactly two private inputs in Kaggle

For the S2 training notebook attach:

1. existing `faruq-v3-experiment-core-v1`;
2. new `faruq-v3-stb1-teacher-addon-v1`.

Enable GPU.

## 3. Run the S2 notebook

Open:

`notebooks/Faruq_V3_STB_Guided_WAVL1_S2_Seed42_Kaggle.ipynb`

Use **Run All**.

The notebook will stop before training unless all of the following pass:

- pinned `ultralytics==8.4.96` install;
- `tests/test_stb_guided.py`;
- Faruq-v3 grouped-development input contract;
- STB1 teacher SHA/type/parameter contract;
- STB-guided static architecture audit;
- no-test contract.

If authorized, it trains only:

`S2_STB_CROSSKD_seed42`

for the frozen 50-epoch configuration.

After training, the runner performs a second deployment checkpoint audit. It
must prove that the serialized candidate still has exactly the WAV-L1 student
parameter count/architecture and contains no teacher, AF2, or training
criterion state before validation is evaluated.

## 4. Expected output

Working directory:

`/kaggle/working/stb-guided-s2-seed42-v1`

Decision JSON:

`/kaggle/working/stb-guided-s2-seed42-v1/val_reports/stb_guided_s2_seed42_decision.json`

Log:

`/kaggle/working/stb-guided-s2-seed42-v1/S2_STB_CROSSKD_seed42.log`

ZIP:

`/kaggle/working/stb-guided-s2-seed42-output.zip`

The final screen prints the frozen WAV-L1 reference, frozen STB1 teacher
reference, S2 candidate, delta versus WAV-L1, retention gates, advancement
gates, and final `PASS`/`FAIL`.

## 5. Stop rule

If `FAIL`, stop S2. Do not tune KD temperature/weight/threshold, do not run
extra seeds, and do not open test.

If `PASS`, stop after seed42 and freeze a separate paired 42/123/2026 S2
confirmation protocol before any additional training.

S3 (`crosskd_af2`) is implemented in code only. It is not authorized by this
runbook even if it is visible in the repository.
