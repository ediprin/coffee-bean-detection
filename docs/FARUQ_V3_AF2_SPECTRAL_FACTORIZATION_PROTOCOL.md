# AF2 Spectral Factorization Protocol

Status: **frozen before training** (2026-08-17)

Repository branch: `agent/af2-spectral-factorization`

Dataset: Faruq-v3 grouped development split only (1,665 train / 294 validation; all 21 labels in validation)
Backbone: YOLO26n P3, initialized from the seed-matched D0 checkpoint

## Question

AF2 has retained in-domain and target-free Coffee Standard evidence, but its
fixed patchwise Fourier implementation can conflate five independent choices:
rectangular-window spectral leakage, 360 redundant orientation bins,
discarded radial frequency structure, hard entropy thresholding, and
channel-independent RGB gates. This study changes exactly one factor at a
time while preserving a raw residual path and one-stage detection.

No arm uses ROI Align, candidate Top-K selection, decoded boxes before
classification, a second crop, a trainable frontend parameter, or test data.
Each frontend is applied directly to the RGB image before unchanged YOLO26n:

\[
x' = x + x \odot \operatorname{normalize}(c(x)).
\]

`AF2C` is the bitwise legacy AF2 control and is reused rather than retrained.

## Frozen arms

| Arm | Change from preceding AF2 path | Fixed setting |
|---|---|---|
| `AF2C` | Legacy control | patch 32, 50% overlap, 360 directions, hard threshold |
| `AF2WIN` | Analysis/synthesis window | periodic square-root Hann plus normalized overlap-add |
| `AF2ORI` | Orientation factorization | 16 bins modulo \(\pi\) |
| `AF2POL` | Radial factorization | 3 geometric-grid radial bands × 16 orientations |
| `AF2SOFT` | Threshold factorization | \(d\,\sigma((d-\tau)/0.02)\) |
| `AF2LUM` | Channel factorization | Rec.709 luminance gate shared by RGB |
| `PCG1` | Mechanistic alternative | log-Gabor phase congruency: 4 scales, 6 orientations |
| `WAV1` | Mechanistic alternative | two-level Haar LH/HL/HH luminance energy |

The PCG constants are wavelength 3, multiplier 2.1, `sigmaOnf=0.65`,
angular ratio 1.5, low-pass cutoff/order 0.4/10, and noise multiplier 2.
Radial boundaries come only from the FFT grid's geometric radius quantiles;
they do not use train or validation image statistics.

Relevant method rationale is fixed from LFDet, Kovesi's phase-congruency
formulation, FADC (CVPR 2024), and WTConv (ECCV 2024). These references
motivate alternatives; they do not imply their efficacy for coffee defects.

## Pre-training gates

The static audit must PASS on CPU and GPU before any arm trains. It verifies:

- bitwise `AF2C` equivalence to legacy AF2;
- deterministic, finite, active frontends and finite gradients to input;
- raw residual preservation and full angular/radial coverage;
- unchanged detector parameter count and state-dict schema;
- all D0 weights transferred, identical YAML/schedule, no persistent frontend
  state, no ROI/decoded-box dependency, and no test access.

The train-only observability audit covers every 1,665 train image. It reports
angular occupancy, entropy and threshold distribution, radial energy,
retained spectral mass, rectangular-versus-Hann leakage proxy, RGB/luminance
disagreement, and cue stability under fixed photometric transforms. It does
not choose a hyperparameter.

## Screening and gates

All seven new arms use seed 42, 50 epochs, identical YOLO26n P3 schedule,
the D0 seed-42 checkpoint, and validation only. `last.pt` is saved each epoch.
Latency is batch-1 at 640 on the same GPU and is reported, not a scientific
rejection criterion.

Stage 1 runs `AF2WIN`, `AF2ORI`, `AF2POL`, `AF2SOFT`, `AF2LUM`. Stage 2 runs
`PCG1`, `WAV1` only after the Stage 1 report has been saved. A candidate is
`RETAIN` versus `AF2C` only when all are true:

- Macro mAP50-95 gain >= 0.5 point;
- Bottom-3 class mAP50-95 is not lower;
- Worst-class mAP50-95 drops <= 1 point;
- validation has all 21 ground-truth classes and test is not accessed.

The global winner is highest Macro. Ties within 0.2 Macro point are broken by
Bottom-3, then Worst class, then lower batch-1 latency. If none retain, AF2C
remains the selected model and no fusion is tried.

## Paired confirmation and post-hoc scope

Only the global winner trains seeds 123 and 2026 from D0 checkpoints having the
same recorded seed. Paired three-seed PASS requires mean Macro gain >= 0.5
point, Macro improvement in >=2/3 seeds, mean Bottom-3 no lower and improved
in >=2/3 seeds, and mean Worst no more than 1 point lower.

After a PASS only, no-training evaluations may run on Coffee Standard
leakage-safe development data, Faruq synthetic-density B0--B3, and the
isolated illumination diagnostic. The illumination report is a paired,
clean-normalized degradation diagnostic on validation identities, **not**
controlled real-lux evidence. Faruq locked test is not reopened.

## Artifact and Kaggle contract

The private Kaggle input contains exactly one verified archive plus explicitly
named `D0_seed42_best.pt`, `D0_seed123_best.pt`, `D0_seed2026_best.pt`, AF2
seed-42 evidence, AF2 three-seed evidence, and `af2_spectral_kaggle_manifest.json`.
Each filename, byte count, and SHA256 is checked before import. Generic search
for `best.pt` is prohibited. A Kaggle arm writes a `run_contract.json` with
arm, seed, config SHA, D0 SHA, epoch count, and test lock; only an identical
contract can restore a previous output. Each arm notebook exports a ZIP with
checkpoints, logs, result JSON, and contract before the session ends.

## Reporting rule

Raw reports and checkpoints remain external artifacts. This repository records
the static/observability reports and, after execution, a result document and
master-log entry for either PASS or FAIL. No performance result is implied by
this frozen protocol.
