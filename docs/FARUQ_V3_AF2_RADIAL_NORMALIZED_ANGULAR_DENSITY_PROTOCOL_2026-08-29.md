# Faruq-v3 AF2RN Protocol — Radially Normalized Angular Density

Status: **frozen before implementation, observability audit, or training**
Date: 2026-08-29
Test: locked

## Research question

Can AF2 improve fine-grained coffee-defect discrimination when its angular
density is normalized against the natural radial spectral baseline, without
removing any radial frequency and without changing the rest of AF2?

The no-repeat basis is frozen in
`docs/FARUQ_V3_AF2_VARIANT_COLLISION_MATRIX_2026-08-29.md`.

## Arms

| Arm | Role | Only treatment difference |
|---|---|---|
| `AF2C` | historical control | original AF2 angular density |
| `AF2RN` | candidate | annulus-median normalization before angular accumulation |

`AF2C` reuses the completed seed-42 result and checkpoint. `AF2RN` starts from
the same D0 seed-42 checkpoint and uses the same YOLO26n-P3 config, grouped
Faruq-v3 development split, seed 42, and 50-epoch schedule.

## Frozen operator

For each 32×32 patch and each RGB channel independently:

1. compute the orthonormal 2-D FFT and shift the origin;
2. assign each frequency coordinate to integer annulus
   `floor(sqrt(dx**2 + dy**2))`;
   merge a non-DC singleton outer annulus into its immediately inner annulus;
3. divide each magnitude by the median magnitude in its annulus;
4. subtract one and clamp below at zero;
5. sum the result into the original 360 signed direction bins;
6. apply the original AF2 entropy-conditioned hard threshold with `gamma=0.10`;
7. multiply the original complex spectrum by that angular weight, preserving
   phase;
8. inverse FFT, overlap-average with the original rectangular patches, and
   apply the original raw-preserving residual gate.

The DC annulus contributes zero after self-normalization. `eps=1e-8` is used
only for numerical safety. The fixed singleton merge prevents the even-grid
Nyquist corner from being suppressed merely because it is alone. There are no
learned frontend parameters and no data-derived radial boundaries.

## Prohibited changes

This experiment must not add or alter:

- AF1/high-pass filtering;
- fixed radial bands or per-band decisions;
- Hann windows;
- 180° orientation folding;
- soft thresholding;
- shared luminance gating;
- wavelet or phase-congruency cues;
- detector architecture, loss, augmentation, optimizer, or schedule;
- feature adapters, auxiliary losses, ROI/crop stages, or decoded-box inputs.

## Static and train-only observability gate

Before training, the implementation must verify:

- AF2C remains bitwise equal to the legacy CPU operator and numerically equal
  within `atol=rtol=1e-6` on CUDA;
- AF2RN is deterministic within the same CUDA tolerance, finite in forward and
  backward, and preserves dtype under AMP;
- detector parameter count and state-dict schema equal AF2C;
- every FFT coordinate belongs to exactly one annulus and one angular bin;
- no coefficient is removed merely because of radius;
- the recovered cue is non-degenerate on at least 95% of all 1,665 train
  images (`spatial_std > 1e-6`);
- median retained spectral mass is strictly between 2% and 98%;
- AF2RN differs numerically from AF2C on at least 95% of train images;
- train-only reporting includes retained mass, angular occupancy, entropy,
  threshold, AF2/AF2RN mask Jaccard, low/mid/high radial retention, spatial
  ringing proxy, and per-class GT-box cue strength;
- validation labels, validation metrics, and test files are not read by this
  observability audit.

Failure of any structural/non-degeneracy gate blocks training. The reported
distributions are diagnostic and cannot be used to tune annulus definition,
threshold, gamma, patch size, or overlap.

## Seed-42 kill gate

AF2RN is retained only if all conditions hold against the original AF2 seed-42
result under the same validation evaluator:

- Macro mAP50–95 gain is at least +0.50 percentage point;
- Bottom-3 mAP50–95 is not lower;
- Worst-class mAP50–95 drops by no more than 1.00 point;
- all 21 validation classes have ground truth;
- test is not accessed.

If the gate fails, stop AF2RN without tuning, fusion, extra seeds, or test.

## Confirmation boundary

Only a seed-42 PASS authorizes a separately frozen paired confirmation for
seeds 123 and 2026 using seed-matched D0 and original-AF2 controls. This
protocol does not itself authorize those runs or reopen the already reused
Faruq test.

## Claims boundary

Until training passes, the only valid claim is that AF2RN is mechanistically
distinct from the repository's completed AF2 variants. No accuracy,
robustness, efficiency, or thesis-superiority claim is authorized.
