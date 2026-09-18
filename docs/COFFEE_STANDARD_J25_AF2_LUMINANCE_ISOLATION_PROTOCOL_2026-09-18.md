# Coffee Standard J25 AF2 Luminance-Isolation Protocol

Status: **frozen before AF2LUMDIRECT training**

Date: 2026-09-18

## Question

Does the failure of RGB-independent AF2 on J25 arise from channel-specific
color-frequency coupling rather than a transferable texture cue?

## Controlled intervention

The completed `D0DIRECT` and `AF2DIRECT` seed-42 reports from
`coffee-standard-j25-train-siblings-af2-direct-seed42-v2` are frozen
references. Exactly one new arm is trained:

- `AF2LUMDIRECT`: compute the unchanged legacy AF2 operator on Rec.709
  luminance, min-max normalize the recovered one-channel signal, and apply the
  same shared gate to all untouched RGB channels using `x + x * gate`.

The arm starts afresh from the same official `yolo26n.pt` initialization. It
uses the same model YAML, 695-image train partition, 68-image validation
partition, seed 42, 50 epochs, image size 640, batch 16, optimizer `auto`, and
all other settings. The frontend remains parameter-free. No AF2 checkpoint is
reused.

## Static gates

- detector state keys, tensors, and parameter count equal native D0 before
  training;
- AF2 hyperparameters exactly match RGB AF2;
- an isoluminant chroma perturbation leaves the luminance-shared gate unchanged
  while changing the legacy RGB-independent gate;
- finite, active train/inference frontend;
- test not accessed.

## Frozen interpretation

Relative to `AF2DIRECT`:

- `SUPPORTS_RGB_CHANNEL_INTERFERENCE` when Macro gains at least 0.2 points
  without lower-tail loss, or when Macro stays within 0.2 points while both
  Bottom-3 and Worst improve by at least 1 point;
- `RGB_CHANNEL_SPECTRA_BENEFICIAL` when luminance loses at least 0.2 Macro
  points and does not improve Bottom-3;
- otherwise `CHANNEL_COLOR_CAUSE_NOT_ISOLATED`.

This is a one-seed mechanistic diagnostic, not a promotion or robustness
confirmation. It stops after seed 42 in every outcome. Locked test remains
closed.

Output root: `experiments/coffee-standard-j25-af2-luminance-v1`.

