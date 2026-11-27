# Auliya fold A: HAAR with learned stride-2 mask prototypes

The previous validation audit found good box localization but undercoverage at
the contour of small black broken beans. This screen asks whether a finer mask
grid helps without modifying the detector's box, class, or mask-coefficient
branches. It does not assume that label noise is absent or that finer masks will
generalize. Group A remains held out and inaccessible.

## One treatment against an existing control

Compare `HAAR_MASK2` against the completed `HAAR` checkpoint from the
[official fold-A screen](AULIYA_OFFICIAL_FOLD_A_FREQUENCY_PROTOCOL_2026-11-26.md).
Train only the new arm from the same official `yolo26s-seg.pt`, not from HAAR's
trained checkpoint. Seed 0, the ordered 676/119 official lists, augmentation,
optimizer, 640 resolution, batch 16, maximum 80 epochs, patience 25, checkpoint
selection metric, and software/GPU contract stay the same. Set training mask
raster stride to 2 so the instance loss actually supervises the added grid.
The control uses the stock training stride 4. HAAR runs unchanged in both arms.

At zero residual, new prototype logits are a 2x nearest-neighbor expansion of
the original 32 channels. A grouped 3x3 convolution followed by pixel shuffle
learns four subpixel residuals per prototype channel (1,152 added weights).
There is no new P2 information in this first test; it isolates the learned
subpixel grid from extra backbone feature fusion. Box and class graph wiring
is unchanged. Joint training can still change the shared features and outputs.

## Comparable scoring

1. The selection and primary reporting grid stays stride 4. Candidate fine
   logits are reduced by a 2x2 mean before box crop and binarization. Polygon
   targets are rasterized directly at stride 4, including during training's
   internal validation. The old HAAR report must be reproducible on this grid.
2. The diagnostic grid is stride 2 for both arms: redraw polygon targets at
   stride 2 and bilinearly interpolate the original HAAR logits before crop
   and binarization. Use batch 4 for both arms on this diagnostic grid.
3. Report in each grid: 18-class mask mAP50–95, Bottom-3, Worst, AP90/AP95,
   classwise mask AP, all-21 mask AP, box AP guardrail, and Biji Hitam Pecah
   AP90/AP95 as a small-sample diagnostic. Report per-image validation timing
   on the same runtime as context, not as a deployment benchmark.

The useful signal is a gain at strict mask IoU without sacrificing the normal
macro score or lower tail. If the gain appears only on one scoring grid, report
that limit. This is one fixed validation screen, not permission to open test,
tune against test, or run more seeds automatically.

## Execution

`Auliya_Official_Fold_A_HAAR_MASK2_Colab.ipynb` uses the existing verified
pilot image bundle, plus the new source ZIP in `Coffee_Bean_Detection/bundles/`.
Audit is the default. Explicit training authorization runs only `HAAR_MASK2`.
The runtime checks the existing HAAR report and SHA-256 before training and
binds the new arm to that exact checkpoint. The checkpoint and paired report
have a separate output directory; the existing HAAR artifacts are read-only.
