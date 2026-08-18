# Faruq-v3 AF2 radial-wavelet refinement protocol

Date frozen: **2026-08-18**  
Status: **frozen before follow-up training**

## Motivation from completed AF2 factorization

The completed seed-42 factorization established two specific signals without
changing the original retention rule:

| Arm | Macro mAP50-95 | Bottom-3 | Worst |
|---|---:|---:|---:|
| AF2C | 88.1973% | 80.0428% | 79.3470% |
| AF2POL | 87.9684% | 83.2599% | 83.1377% |
| WAV1 | 88.4105% | 83.2761% | 82.0349% |

AF2POL improved the lower tail but was cumulative (`WIN + ORI + POL`), so the
radial effect has not yet been inserted directly into the legacy 360-bin AF2
operator. WAV1 improved Macro and the lower tail but remained below the frozen
+0.5-point Macro retention threshold. PCG1 degraded all headline metrics and is
not continued.

This follow-up therefore asks one narrow question:

> Can the radial structure suggested by AF2POL and/or the already successful
> WAV1 cue be integrated directly with legacy AF2 while preserving AF2's Macro
> performance?

This is a new protocol. It does not reinterpret the completed spectral
factorization or retroactively change any of its decisions.

## Shared control and training contract

- Dataset: Faruq-v3 grouped development only, 1665 train / 294 validation.
- Validation must contain all 21 classes.
- Initial detector: seed-matched D0 YOLO26n P3 checkpoint.
- Discovery seed: 42 only.
- 50 epochs, 640 px, batch 16, workers 2, patience 15, optimizer `auto`,
  `close_mosaic=10`, `max_det=500`, `pretrained=false`.
- Frontends are parameter-free and are applied before the native YOLO26 model.
- No trainable fusion coefficient and no hyperparameter search are allowed.
- Faruq locked test remains closed.

## Arms

### AF2C — unchanged legacy reference

Canonical AF2: rectangular 32x32 patches, 50% overlap, 360-degree directional
bins, independent RGB transforms, hard entropy threshold (`gamma=0.1`), inverse
FFT, overlap averaging, min-max cue, and residual multiplicative gate.

The static audit must prove AF2C is bitwise equal to the existing
`AFABInputEnhancer(mode="af2")` implementation.

### AF2RAD — direct radial insertion into legacy AF2

AF2RAD changes only the directional-density index. The FFT coefficient at
location `(r, theta)` is accumulated into a joint radial-direction cell:

`cell = radial_bin(r) * 360 + angle_bin(theta)`.

Three geometric radial bands are formed from fixed quantiles of the 32x32 FFT
radius grid. Within each radial band, probability, entropy threshold, and
normalized directional density are computed over the same 360 directions used
by legacy AF2.

AF2RAD deliberately does **not** inherit the Stage-1 Hann window, 16-bin
orientation reduction, soft threshold, or luminance-only gate. All other AF2
operations are inherited unchanged from the legacy implementation.

### AF2WAV — legacy AF2 plus the frozen WAV1 cue

The AF2 spectral cue is the legacy AF2 recovered image after per-channel
min-max normalization. The wavelet cue is exactly the completed WAV1 operator:
Rec.709 luminance, two-level orthonormal Haar DWT, LH/HL/HH detail-energy at
each level, bilinear return to the input resolution, per-level min-max
normalization, and mean across the two levels.

The cues are fused without a learned or tuned coefficient:

`cue = max(AF2_cue, WAV1_cue)`.

The input gate is applied once:

`x' = x + x * cue`.

The static audit must prove that the wavelet cue is bitwise equal to the
already-tested WAV1 implementation and that max fusion never numerically
attenuates the AF2 cue.

### AF2RADWAV — radial AF2 plus the frozen WAV1 cue

Identical to AF2WAV except that the AF2 branch is AF2RAD. The same coefficient-
free max fusion and single residual gate are used.

## Static gates before training

Training is forbidden unless all of the following pass:

1. D0 is loadable and has `nc=21`.
2. AF2C is bitwise equal to legacy AF2 on CPU.
3. AF2RAD uses exactly three radial bands and 360 angular bins.
4. The follow-up wavelet cue is bitwise equal to WAV1.
5. Max fusion never attenuates the AF2 cue.
6. Every train arm has finite forward/backward behavior under deterministic
   algorithms, repeatability within `atol=rtol=1e-6`, no trainable frontend
   parameters, and no persistent frontend state.
7. No test access is authorized.

## Frozen seed-42 retention rule

The primary retention gate is intentionally unchanged from the original
spectral-factorization protocol. Relative to AF2C, an arm is `RETAIN` iff:

- Macro mAP50-95 gain is at least **+0.5 percentage point**;
- Bottom-3 mAP50-95 is not lower;
- Worst-class mAP50-95 does not drop by more than **1.0 percentage point**;
- validation contains all 21 classes and test is not accessed.

Lower-tail improvements are recorded descriptively but do not relax this gate.

If multiple arms retain, choose highest Macro. A Macro tie within 0.2 point is
broken by Bottom-3, then Worst, then lower batch-1 median latency.

If no arm retains, keep AF2C and stop this refinement branch. Do not create a
post-hoc fusion, retune thresholds, add seeds, or open test.

## After a seed-42 PASS

Only the single frozen winner may proceed to paired seeds 123 and 2026 against
seed-matched AF2/D0 controls under a separately frozen confirmation contract.
The Faruq locked test remains closed until a later protocol explicitly and
independently authorizes access.

## Evidence inputs

The private Kaggle core bundle remains the authoritative source for:

- `D0_seed42_best.pt`;
- seed-matched D0 checkpoints for any later paired confirmation;
- `lfdet_afab_seed42_screening.json` containing the AF2C reference;
- AF2 paired-confirmation evidence.

The completed Stage-1/PCG1/WAV1 Saved Versions are historical evidence for the
motivation above; they are not training inputs to AF2RAD/AF2WAV/AF2RADWAV.
