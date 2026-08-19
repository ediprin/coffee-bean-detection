# Faruq-v3 WAV1 Factorization — Parallel Colab Runbook

Purpose: run the four frozen seed-42 mechanism-factorization arms in parallel across separate Google Colab accounts while sharing one canonical `Coffee_Bean_Detection` Drive folder.

## Shared Drive rule

Every account must have Editor access to the same shared project folder and add a shortcut under My Drive so that the logical path is:

```text
/content/drive/MyDrive/Coffee_Bean_Detection
```

Do not create copies of the project, dataset archive, checkpoints, or experiments folder.

Required shared artifacts:

```text
bundles/faruq-development-v3-grouped.tar
experiments/faruq-v3-yolo26n-baseline-v1/D0_seed42/weights/best.pt
```

The notebooks extract the development dataset into each account's local `/content`, so parallel accounts do not share temporary dataset files.

## One account = one arm

| Account | Notebook | Arm | Shared Drive output |
|---|---|---|---|
| A | `Faruq_V3_WAV1_Factorization_HP1_Colab.ipynb` | `HP1` | `experiments/faruq-v3-wav1-mechanism-factorization-v1/parallel/HP1` |
| B | `Faruq_V3_WAV1_Factorization_WAV_L1_Colab.ipynb` | `WAV_L1` | `experiments/faruq-v3-wav1-mechanism-factorization-v1/parallel/WAV_L1` |
| C | `Faruq_V3_WAV1_Factorization_WAV_L2_Colab.ipynb` | `WAV_L2` | `experiments/faruq-v3-wav1-mechanism-factorization-v1/parallel/WAV_L2` |
| D | `Faruq_V3_WAV1_Factorization_WAV_RAWFUSE_Colab.ipynb` | `WAV_RAWFUSE` | `experiments/faruq-v3-wav1-mechanism-factorization-v1/parallel/WAV_RAWFUSE` |

Each arm has a separate output root, static audit, log, lock, checkpoint directory, and validation report. This prevents cross-account writes from colliding.

## Per-notebook behavior

Each model notebook:

1. mounts Google Drive;
2. clones `agent/wav1-mechanism-factorization`;
3. installs the project and pinned Ultralytics dependency contract;
4. resolves the canonical shared project root;
5. verifies the Faruq development archive and D0 seed-42 checkpoint;
6. extracts the development dataset locally;
7. asserts that no `test` directory is exposed;
8. runs an independent static factorization audit;
9. verifies the frozen WAV1 reference remains bitwise-equivalent to the confirmed implementation;
10. runs a CUDA forward/backward smoke test for the selected arm;
11. trains only that arm at seed 42;
12. evaluates only `val`;
13. stores checkpoints, logs, latency, and result JSON directly in the shared Drive output shown above.

The runner is resume-aware. If an account disconnects, rerun the same notebook from the same account or another account with access to the shared folder; the existing run contract/checkpoint will be reused when compatible.

## After all four finish

Run the CPU-only notebook:

```text
Faruq_V3_WAV1_Factorization_Parallel_Report_Colab.ipynb
```

It reads all four result JSONs, D0FT seed-42 evidence, and the frozen WAV1 seed-42 headline reference, then writes:

```text
experiments/faruq-v3-wav1-mechanism-factorization-v1/parallel_seed42_report.json
```

The report computes each arm's gain versus D0FT and the fraction of the frozen WAV1 gain retained for Macro, Bottom-3, and Worst-class mAP50-95. It does not authorize extra seeds or select a winner automatically.

## Stop rule

After the parallel report is produced, stop training. Do not launch seed 123/2026 or reopen the locked test until the mechanism results are reviewed and a separate confirmation protocol is frozen.
