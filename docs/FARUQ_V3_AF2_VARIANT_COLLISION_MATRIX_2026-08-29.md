# Faruq-v3 AF2 Variant Collision Matrix — 2026-08-29

## Decision

The repository-wide audit supports exactly one new frontend hypothesis:
`AF2RN`, or **radially normalized angular density**. It is not the already
failed radial-band route under a new name.

No training is authorized by this document. `AF2RN` must first pass a static
gate and a train-only observability audit. The machine-readable inventory is
`docs/evidence/FARUQ_V3_AF2_VARIANT_COLLISION_MATRIX_2026-08-29.json`.

## Why this audit was necessary

AF2 has accumulated several classes of experiments: direct modifications of
the Fourier frontend, trainable feature adapters, auxiliary losses,
continuation controls, and combinations with other retained detectors. A new
name alone is not a new mechanism. This matrix separates those axes so a new
experiment cannot silently repeat a failed one.

The authoritative original AF2 mean across seeds 42/123/2026 remains 87.94%
Macro, 79.37% Bottom-3, and 78.15% Worst-class mAP50–95. AF2 remains the
confirmed frequency frontend. `AF2FFAB2` is a separately validated Pareto
refinement versus its matched continuation control, not proof that the
original AF2 frontend has been replaced.

## Direct spectral/frontend collision matrix

| Prior arm | Factor changed | Scientific status | What AF2RN must not repeat |
|---|---|---|---|
| AF1 | Adaptive circular high-pass | Seed-42 discovery only | No radial-frequency removal |
| AF12 | AF1 mask then AF2 mask | Rejected | No sequential or multiplied masks |
| AF2WIN | Hann window/overlap-add | Rejected | Keep legacy rectangular window |
| AF2_ORIENT | 180° orientation folding | Failed three-seed confirmation | Keep 360 signed direction bins |
| AF2POL | Hann + orientation + three radial bands | Rejected by Macro gate; tail improved | No independent radial-band threshold |
| AF2SOFT | Soft threshold | Rejected | Keep legacy hard threshold |
| AF2LUM / CAFR-C1 | Shared luminance gate | Rejected | Keep RGB-independent processing |
| PCG1 | Phase congruency | Rejected | No alternate transform |
| WAV1 | Haar detail energy | Rejected by Macro gate; tail improved | No wavelet path |
| AF2_RADIAL / AF2RAD | Isolated fixed radial bands | Rejected twice | No radial bands or radial masks |
| AF2WAV | AF2/WAV cue max fusion | Rejected | No cue fusion |
| AF2RADWAV | Radial AF2 plus WAV cue | Rejected | No radial masking or cue fusion |
| CAFR C2/C3/C4 | Radial × direction, soft gate, orientation folding | Full ladder rejected | None of those cumulative factors |
| AF2_ORIENT + CMC0 | Orientation frontend plus pointwise head | Rejected | No head-capacity change |

The two radial results are decisive. `AF2_RADIAL` reached 86.57/78.21/75.41%
and `AF2RAD` reached 86.78/76.95/72.09%, both below AF2. `AF2POL` reached
87.97/83.26/83.14%, but it was cumulative `WIN + ORI + POL`; the later
isolated radial study showed that radial partitioning alone did not cause its
tail gain.

## Adjacent mechanisms already tested

These are important but do not collide with a parameter-free input statistic:

| Family | Examples | Status |
|---|---|---|
| Residual/channel conditioning | AF2R1, AF2CAL3, AF2RCC1 | Rejected against matched controls |
| Feature-frequency adapter | AF2FFA1, AF2FFAB2 | AF2FFAB2 passed three-seed causal comparison versus AF2FFA0; descriptive Worst remains below original AF2 |
| Auxiliary objectives | DG, FG, DG+FG, BHCL, AF2SPDS | DG/FG/BHCL rejected; SPDS missed Macro gate despite strong tail gain |
| Shared feature selector | AF2SFS1 | Retained at seed 42 only; not confirmed, and diagnostics indicate optimization-mediated gain |
| Other-detector composition | AF2+STB, AF2+IGEM, AF2+SAF | Tested formulations rejected |
| Class-selective residual | AF2CSD1 | Effect too small; rejected |
| Multilevel scaffold | AF2MTS1 | Rejected with severe lower-tail loss |

The previously pending SPDS refinements are now complete. `AF2CUE1` was a
near-miss: it improved the matched base by +0.42/+3.12/+6.05 points, but lost
0.71 Bottom-3 point versus AF2SPDS and exceeded the frozen 0.50-point tolerance.
`AF2DECAY1` was lower than AF2SPDS on all three metrics. Both are rejected;
neither collides with AF2RN because they change a removable training-only
objective rather than the input spectral statistic.

## The one non-duplicated hypothesis

For each patch and RGB channel, let `A(r, theta)` be FFT magnitude. Legacy AF2
immediately sums magnitude over radius to obtain angular density. AF2RN first
computes a geometry-only integer annulus:

```text
rho(u, v) = floor(sqrt((u - c)^2 + (v - c)^2))
```

The single outer Nyquist-corner member created by the even 32×32 grid is
merged into its immediately inner annulus. This fixed geometry rule prevents
a singleton median from suppressing that coefficient automatically.

It then normalizes every coefficient by the median magnitude of its own
annulus:

```text
Z(u, v) = max(A(u, v) / (median_ring(A) + eps) - 1, 0)
D(theta) = sum Z(u, v) for coordinates assigned to theta
```

Everything after `D(theta)` stays legacy AF2: 360 signed bins, entropy
threshold with `gamma = 0.10`, hard suppression, original phase, inverse FFT,
per-channel processing, and raw-preserving residual addition.

The radial coordinate therefore acts only as a **nuisance normalization
index**. It never selects, removes, partitions for separate decisions, or
reweights a frequency band. This is the exact boundary separating AF2RN from
AF2POL, AF2_RADIAL, AF2RAD, AF1, and AF12.

## Mechanistic rationale

Natural patches often have a dominant radial spectral fall-off: low
frequencies carry much more energy than high frequencies. Legacy AF2 sums this
unequal baseline into the angular statistic. A weak but class-relevant scratch,
hole rim, crack, or spotted boundary can therefore be directionally coherent
yet numerically small compared with the radial background.

AF2RN asks a narrower question: *is this direction unusually strong relative
to other coefficients at the same spatial frequency?* It keeps low-frequency
colour/shape and high-frequency texture simultaneously, avoiding the failure
mode of AF1/AF12 and fixed radial bands.

This rationale is plausible, not a result claim. A training result is required
before any performance statement.

## No-repeat ruling

The following are closed and may not be silently added to AF2RN during this
study: Hann windows, 180° folding, fixed radial bands, soft thresholds,
luminance sharing, wavelet/phase-congruency fusion, learned channel scales,
feature adapters, auxiliary losses, ROI processing, or detector-head changes.

If AF2RN fails, the result closes this statistic. It does not authorize an
`AF2RN + previous module` rescue run.

## Evidence provenance

The audit used the current master log plus result documents stored on the
historical branches that produced the experiments, notably
`agent/af2-continuation-confirmation`, `agent/af2-rad-wavelet-refinement`,
`agent/cafr-yolo`, `codex/af2-feature-frequency-adapter`, and
`codex/af2-igem-parent-confirmation`. Branch-qualified paths are preserved in
the JSON evidence rather than pretending every historical result file is
present on the current branch.
