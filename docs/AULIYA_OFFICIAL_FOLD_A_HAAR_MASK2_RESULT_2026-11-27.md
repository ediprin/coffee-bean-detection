# Auliya fold A: HAAR_MASK2 validation result

Decision: **retain HAAR_MASK2 as a fine-grid mask candidate, not as a
general replacement for HAAR on the standard grid.** This is a single-seed
development screen. The held-out group A was not opened.

## Comparison

The [frozen MASK2 protocol](AULIYA_OFFICIAL_FOLD_A_HAAR_MASK2_PROTOCOL_2026-11-27.md)
compares native `HAAR` with `HAAR_MASK2`, which adds a 1,152-parameter
stride-2 prototype residual and trains it using stride-2 polygon targets.
The box, classification, and mask-coefficient graph wiring is unchanged.
Training still updates shared features, so prediction equality after
optimization is not assumed. The experiment tests the combined learned
stride-2 prototype and matching targets, not their separate contributions.

Both arms started from the same official YOLO26s-seg asset, used seed 0, and
completed 80/80 epochs with the official 676-image train and 119-image
validation lists for fold A. The runs used Tesla T4 and Ultralytics 8.4.210.
The paired comparison reused both `best.pt` checkpoints, verified their hashes,
and performed no training. Ground-truth polygons are rasterized directly
onto each shared evaluation grid. On stride 2, HAAR logits are interpolated
before mask decoding; on stride 4, candidate logits are reduced to the
canonical raster. Compare models **within** each grid, not across grids.

## Results

AP is reported in percent; differences are in percentage points.

| Metric | Stride 4 HAAR | Stride 4 MASK2 | Δ | Stride 2 HAAR | Stride 2 MASK2 | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mask mAP50–95, 18 classes | 82.91 | 83.04 | +0.13 | 86.17 | 90.51 | **+4.35** |
| Bottom-3 mask AP | 74.93 | 74.47 | −0.46 | 79.94 | 83.09 | **+3.15** |
| Worst mask AP | 72.90 | 72.77 | −0.13 | 79.01 | 80.55 | **+1.54** |
| Mask AP90, 18 classes | 61.36 | 62.30 | +0.94 | 88.06 | 91.50 | **+3.44** |
| Mask AP95, 18 classes | 15.69 | 17.39 | +1.69 | 17.83 | 53.50 | **+35.67** |
| Mask mAP50–95, all 21 classes | 83.35 | 82.99 | −0.36 | 86.32 | 90.16 | **+3.85** |
| Box mAP50–95, all 21 classes | 93.99 | 93.64 | −0.35 | 93.90 | 93.71 | −0.19 |

For **Biji Hitam Pecah**, stride-2 mask AP90 improved from 44.47% to 49.13%
(+4.66 points), and AP95 improved from 0.43% to 9.84% (+9.41 points).
Strict overlap is still difficult for this class. The prior contour audit
identified undercoverage in 13 of 15 correctly classified, localized targets
across seven images; these examples should not be counted as independent.
On stride 4, its AP90 was unchanged and AP95 fell from 0.65% to zero.
Not all classes improved on the fine grid: Biji Muda lost 3.37 points.

Paired validation inference time per image was 11.65 → 12.50 ms on stride 4
(batch 16), and 17.24 → 18.26 ms on stride 2 (batch 4). These are validator
timings, not synchronized batch-1 deployment measurements.

## Claim boundary

MASK2 improves fine-grid mask precision in this run, especially at IoU 0.95.
Standard-grid scores show a trade-off: strict AP rises, but Bottom-3 and Worst
fall slightly. The fine-grid numbers are not the canonical Auliya test
protocol and do not demonstrate held-out generalization. This comparison
does not establish that HAAR is required for MASK2. A future matched
`RGB + MASK2` control, evaluated on the same two grids, would test that
question; it is not an experiment completed or authorized by this record.

Paired evidence: shared Drive project
`experiments/auliya-official-fold-a-haar-mask2-v1/paired_validation.json`.
The candidate's run contract and training artifacts are in the `HAAR_MASK2/`
subdirectory. The control artifacts remain in
`experiments/auliya-official-fold-a-frequency-v1/HAAR/`.
