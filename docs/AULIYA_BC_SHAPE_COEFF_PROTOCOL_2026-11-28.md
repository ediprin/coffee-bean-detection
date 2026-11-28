# Auliya B→C — SHAPECOEF1

Research question: does supervision of object shape during detector training improve *Biji Hitam Sebagian* discrimination on group C? Earlier frozen-detector corrections after and before top-k did not recover enough targets at the operating threshold. The candidate is untrained; the hypothesis is open.

## Intervention

Read a scalar from the native 32-dimensional mask coefficients at each anchor. An auxiliary smooth-L1 term teaches the scalar to predict the solidity of the matched, augmented ground-truth mask for classes 7 and 12–14. Solidity is contour area divided by convex-hull area, computed on the native training raster. Tiny or disconnected masks are skipped. Normalize as `(solidity - 0.96)/0.04`, clipped to [-1,1]. The auxiliary loss is 0.10× native classification gain; both native E2E branches and the existing assignment method remain in use.

The scalar can adjust **only** class-7 logits through a learned, zero-initialized gain bounded by `log(4)`. Mask coefficients are detached before this projection: the extra loss has no direct gradient to the mask/box heads. Shared detector training and assignment can still change those predictions empirically. The two branches add 68 parameters while training, 34 in the fused inference graph. No second crop, no extra mask decode, no spatial transform at inference. The runtime cost has not been benchmarked.

## Matched screen

Run a fresh RGB arm and SHAPECOEF1, starting from the same official pretrained YOLO26n-seg asset (not a coffee checkpoint), same runtime and same GPU. Use 375 group-B training images and 420 group-C validation images, seed 42, 50 fixed epochs, 640 px, batch 8, 2 workers, and the same Ultralytics 8.4.96 schedule as the RGB pilot. Use fixed epoch-50 `last.pt`, not an epoch picked on C. Group A stays closed. This is an exploratory development screen because C has already informed the research direction.

Report Biji Hitam Sebagian box AP50–95, correct class on matched boxes and target false positives at fixed confidence 0.25/IoU 0.50, and 18-class box/mask AP with tail metrics. A gain in the target class is not enough if it mostly produces extra false positives or visibly degrades the control's detection quality. An improvement in the diagnostic is not a claim of generalization on A. Do not tune auxiliary weight, residual bound or threshold using C in this screen.

## Run

Use the prepared B/C development directory and the SHA-pinned `yolo26n-seg.pt` already used by the prior RGB pilot. From an installed checkout, run the CPU preflight first (default, no training):

```sh
python -m coffee_detector.experiments.run_auliya_shape_coeff \
  --development /path/to/auliya-development \
  --weights /path/to/yolo26n-seg.pt \
  --output /path/to/auliya-bc-shape-coeff-v1
```

Only after explicit training authorization, append `--device 0 --arm RGB --authorize-training` or `--device 0 --arm SHAPECOEF1 --authorize-training` to that command. The candidate automatically runs its GPU preflight before training. Use a separate session per arm and the same persistent output root. Each arm maintains its own exclusive lock, contract, progress marker, optimizer-bearing `last.pt`, and validation report. A lock left by a hard interruption must only be removed after verifying the original session is no longer active. Do not use a legacy RGB pilot checkpoint as an initializer.

### Kaggle

Import `notebooks/Auliya_BC_SHAPECOEF1_Kaggle.ipynb`, attach the private
`auliya-shape-coeff-kaggle-code-v1.zip` input and the existing B/C development
input (extracted directory or `auliya-robusta-sni-b-train-c-val.zip`). The code
input includes pinned official `yolo26n-seg.pt`, but no dataset or trial
checkpoint. The notebook verifies the code manifest and the SHA256 of the
individual dataset files, so it can reuse the same B/C dataset as Semantic
Transfer when the contents match exactly. Open a GPU T4 session with Internet
enabled for the pinned dependencies and Ultralytics runtime assets. GitHub
access is not needed. The packaged run keeps the 8.4.96, YOLO26n-seg, 50-epoch
protocol above, separate from the Semantic Transfer run.

Default execution only audits. After a GPU PASS and authorization, set
`AUTHORIZE_TRAINING=True`. `RUN_ARMS` defaults to RGB then SHAPECOEF1. For one
arm per Notebook set `RUN_ARMS` to one item; attach the saved outputs from
the other arm to collect the paired report. To resume, attach a previously
saved full Notebook Output; the runner checks the arm, bundle, dataset,
official weight, runtime/GPU, and `last.pt` hash. The small `*-reports.zip`
is for analysis and does not contain checkpoint weights. Unsaved Kaggle
session state cannot be restored by this notebook.

Rebuild the code input using `python tools/build_auliya_shape_coeff_kaggle.py
--weights PATH --development-zip PATH --output
outputs/auliya-shape-coeff-kaggle-code-v1.zip`. If the payload changes, pin the
new manifest SHA in the notebook and keep checkpoints from the old payload
separate.
